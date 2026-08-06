"""Preprocessing pipeline, vocabulary, and DataLoader factory.

The vocabulary is fit on the training split only; fitting on val/test would leak
information about held-out examples into the model's token coverage.
"""
from __future__ import annotations

import html
import re
import unicodedata

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import config

PAD_TOKEN, UNK_TOKEN = "<pad>", "<unk>"
PAD_IDX, UNK_IDX = 0, 1
CTX_TOKEN, MSG_TOKEN, TGT_TOKEN = "<ctx>", "<msg>", "<tgt>"
CONTEXT_SPECIAL_TOKENS = (CTX_TOKEN, MSG_TOKEN, TGT_TOKEN)

# Slang is expanded so semantically identical phrases collapse to shared tokens
# the model already has signal for (e.g. "jg" and "jungle" stop competing for
# separate embeddings). Multi-word values are split downstream during tokenize.
SLANG_MAP = {
    "jg": "jungle",
    "ff": "forfeit",
    "ff15": "forfeit",
    "ff20": "forfeit",
    "inting": "intentional feeding",
    "int": "intentional feeding",
    "gg": "good game",
    "wp": "well played",
    "gj": "good job",
    "kys": "kill yourself",
    "adc": "attack damage carry",
    "mid": "middle",
    "bot": "bottom",
    "afk": "away from keyboard",
    "diff": "difference",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def clean_text(text: str) -> list[str]:
    """Normalize HTML/Unicode/whitespace, tokenize, then expand LoL slang.

    NFKD transliteration preserves ASCII equivalents such as ``é`` -> ``e``;
    symbols without an ASCII equivalent (including emoji) are removed.
    """
    text = html.unescape(str(text))
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = " ".join(text.split())
    tokens: list[str] = []
    for tok in _TOKEN_RE.findall(text):
        # Expand slang first; a mapped value may itself be multi-word.
        tokens.extend(SLANG_MAP.get(tok, tok).split())
    return tokens


class Vocab:
    """Maps tokens to integer ids and encodes token lists to padded id arrays."""

    def __init__(self, stoi: dict[str, int]):
        self.stoi = stoi
        self.itos = {i: s for s, i in stoi.items()}

    def __len__(self) -> int:
        return len(self.stoi)

    @classmethod
    def build(
        cls,
        token_lists,
        max_size: int,
        min_freq: int,
        *,
        reserved_tokens: tuple[str, ...] = (),
    ) -> "Vocab":
        counts: dict[str, int] = {}
        for tokens in token_lists:
            for tok in tokens:
                counts[tok] = counts.get(tok, 0) + 1
        # Sort by frequency (desc) then token for deterministic ids across runs.
        ordered = sorted(
            (t for t, c in counts.items() if c >= min_freq),
            key=lambda t: (-counts[t], t),
        )
        stoi = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        for tok in reserved_tokens:
            if tok not in stoi:
                stoi[tok] = len(stoi)
        reserved = set(stoi)
        for tok in ordered:
            if tok in reserved:
                continue
            if len(stoi) >= max_size:
                break
            stoi[tok] = len(stoi)
        return cls(stoi)

    def encode(self, tokens, max_len: int) -> np.ndarray:
        ids = [self.stoi.get(tok, UNK_IDX) for tok in tokens[:max_len]]
        # Right-pad so all sequences share a length; PAD_IDX is masked by the
        # embedding's padding_idx so it contributes no gradient.
        ids += [PAD_IDX] * (max_len - len(ids))
        return np.asarray(ids, dtype=np.int64)


def build_context_tokens(
    group_texts: list[str],
    target_pos: int,
    k: int,
) -> list[str]:
    """Build a same-match context token sequence centered on ``target_pos``.

    Uses up to ``k`` previous and ``k`` next messages inside the group only.
    Markers: ``<ctx>`` before the target, ``<tgt>`` on the target, ``<msg>`` after.
    """
    if k < 1:
        raise ValueError(f"context_k must be >= 1, got {k}")
    if not 0 <= target_pos < len(group_texts):
        raise IndexError(
            f"target_pos {target_pos} out of range for group length {len(group_texts)}"
        )
    start = max(0, target_pos - k)
    end = min(len(group_texts), target_pos + k + 1)
    tokens: list[str] = []
    for index in range(start, end):
        if index == target_pos:
            tokens.append(TGT_TOKEN)
        elif index < target_pos:
            tokens.append(CTX_TOKEN)
        else:
            tokens.append(MSG_TOKEN)
        tokens.extend(clean_text(group_texts[index]))
    return tokens


def add_context_column(df: pd.DataFrame, context_k: int) -> pd.DataFrame:
    """Return a copy of ``df`` with a ``context_text`` column of joined tokens.

    Requires ``group_id`` and prefers ``msg_index`` for within-match order.
    Neighbors are drawn only from rows that share the same ``group_id``.
    """
    if "group_id" not in df.columns:
        raise ValueError("context windows require a group_id column.")
    ordered = df.copy()
    if "msg_index" in ordered.columns:
        ordered = ordered.sort_values(["group_id", "msg_index"], kind="stable")
    else:
        ordered = ordered.sort_values(["group_id"], kind="stable")
    ordered = ordered.reset_index(drop=True)

    context_texts: list[str] = [""] * len(ordered)
    for _, group in ordered.groupby("group_id", sort=False):
        positions = group.index.to_numpy()
        texts = group["text"].astype(str).tolist()
        group_ids = group["group_id"].to_numpy()
        if len(set(group_ids)) != 1:
            raise AssertionError("Context builder mixed group_id values.")
        for local_pos, frame_pos in enumerate(positions):
            tokens = build_context_tokens(texts, local_pos, context_k)
            context_texts[frame_pos] = " ".join(tokens)
    ordered["context_text"] = context_texts
    return ordered


class ToxicChatDataset(Dataset):
    """Wraps a DataFrame of (text, label) rows as encoded tensors."""

    def __init__(
        self,
        df: pd.DataFrame,
        vocab: Vocab,
        max_len: int = config.MAX_LEN,
        *,
        text_column: str = "text",
    ):
        self.vocab = vocab
        self.max_len = max_len
        if text_column == "context_text":
            # Context strings already include special markers as whitespace tokens.
            token_lists = [str(text).split() for text in df[text_column]]
        else:
            token_lists = [clean_text(t) for t in df[text_column]]
        self.lengths = np.asarray(
            [max(1, min(len(tokens), max_len)) for tokens in token_lists],
            dtype=np.int64,
        )
        self.encoded = [vocab.encode(tokens, max_len) for tokens in token_lists]
        self.labels = df["label"].to_numpy(dtype=np.float32)
        self.texts = (
            df["raw_text"].astype(str).tolist()
            if "raw_text" in df.columns
            else df["text"].astype(str).tolist()
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.encoded[idx])
        length = torch.tensor(self.lengths[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, length, y


def _split(df: pd.DataFrame):
    """Stratified train/val/test split so class balance is preserved per split."""
    rng = np.random.default_rng(config.SEED)
    train_parts, val_parts, test_parts = [], [], []
    for _, group in df.groupby("label"):
        idx = rng.permutation(len(group))
        g = group.iloc[idx].reset_index(drop=True)
        n_train = int(len(g) * config.TRAIN_FRAC)
        n_val = int(len(g) * config.VAL_FRAC)
        train_parts.append(g.iloc[:n_train])
        val_parts.append(g.iloc[n_train : n_train + n_val])
        test_parts.append(g.iloc[n_train + n_val :])

    def _concat(parts):
        return pd.concat(parts).sample(frac=1, random_state=config.SEED).reset_index(drop=True)

    return _concat(train_parts), _concat(val_parts), _concat(test_parts)


def load_splits(csv_path=config.DATA_PATH):
    """Return shared train/validation/test DataFrames.

    Prepared L2DTnH data carries case-grouped split assignments. The legacy
    role-proxy Tribunal file has no split column and falls back to the original
    row-level stratified split so the first experiment remains reproducible.
    """
    df = pd.read_csv(csv_path)
    required = {"text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(
            "Expected a CSV with text,label columns. Run prepare_tribunal.py first."
        )
    if "split" in df.columns:
        expected = {"train", "val", "test"}
        observed = set(df["split"].dropna().unique())
        if observed != expected:
            raise ValueError(f"Expected split values {expected}, got {observed}.")
        splits = tuple(
            df[df["split"] == name].reset_index(drop=True)
            for name in ("train", "val", "test")
        )
        if "group_id" in df.columns:
            group_sets = [set(part["group_id"]) for part in splits]
            if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i + 1, 3)):
                raise ValueError("A group_id appears in more than one data split.")
        return splits
    return _split(df)


def get_dataloaders(
    csv_path=config.DATA_PATH,
    batch_size: int = config.BATCH_SIZE,
    *,
    seed: int = config.SEED,
    include_test: bool = True,
    context_k: int | None = None,
    max_len: int | None = None,
    vocab: Vocab | None = None,
):
    """Build deterministic DataLoaders and a training-only vocabulary.

    ``include_test=False`` is the tuning path: the held-out frame is neither
    wrapped in a Dataset nor exposed to the caller. The returned test loader is
    therefore ``None`` until a configuration has been frozen.

    When ``context_k`` is set, each example concatenates same-``group_id``
    neighbors and the vocabulary reserves context marker tokens.

    Pass ``vocab`` to restore a frozen checkpoint vocabulary instead of rebuilding
    from the current ``MIN_FREQ`` / train tokens.
    """
    train_df, val_df, test_df = load_splits(csv_path)
    resolved_max_len = config.MAX_LEN if max_len is None else max_len
    text_column = "text"
    reserved: tuple[str, ...] = ()
    if context_k is not None:
        train_df = add_context_column(train_df, context_k)
        val_df = add_context_column(val_df, context_k)
        if include_test:
            test_df = add_context_column(test_df, context_k)
        text_column = "context_text"
        reserved = CONTEXT_SPECIAL_TOKENS
        if max_len is None:
            resolved_max_len = config.CONTEXT_MAX_LEN

    def _token_lists(frame: pd.DataFrame):
        if text_column == "context_text":
            return (str(text).split() for text in frame[text_column])
        return (clean_text(t) for t in frame["text"])

    if vocab is None:
        vocab = Vocab.build(
            _token_lists(train_df),
            max_size=config.MAX_VOCAB_SIZE,
            min_freq=config.MIN_FREQ,
            reserved_tokens=reserved,
        )
    train_ds = ToxicChatDataset(
        train_df, vocab, resolved_max_len, text_column=text_column
    )
    val_ds = ToxicChatDataset(
        val_df, vocab, resolved_max_len, text_column=text_column
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    test_dl = None
    if include_test:
        test_ds = ToxicChatDataset(
            test_df, vocab, resolved_max_len, text_column=text_column
        )
        test_dl = DataLoader(test_ds, batch_size=batch_size)
    return train_dl, val_dl, test_dl, vocab
