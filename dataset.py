"""Preprocessing pipeline, vocabulary, and DataLoader factory.

The vocabulary is fit on the training split only; fitting on val/test would leak
information about held-out examples into the model's token coverage.
"""
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import config

PAD_TOKEN, UNK_TOKEN = "<pad>", "<unk>"
PAD_IDX, UNK_IDX = 0, 1

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
    """Lowercase, drop non-ASCII, then tokenize with slang expansion.

    Non-ASCII is stripped via encode/decode rather than regex so emoji and
    other unicode noise vanish before tokenization instead of fragmenting it.
    """
    text = str(text).lower()
    text = text.encode("ascii", "ignore").decode("ascii")
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
    def build(cls, token_lists, max_size: int, min_freq: int) -> "Vocab":
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
        for tok in ordered[: max(0, max_size - len(stoi))]:
            stoi[tok] = len(stoi)
        return cls(stoi)

    def encode(self, tokens, max_len: int) -> np.ndarray:
        ids = [self.stoi.get(tok, UNK_IDX) for tok in tokens[:max_len]]
        # Right-pad so all sequences share a length; PAD_IDX is masked by the
        # embedding's padding_idx so it contributes no gradient.
        ids += [PAD_IDX] * (max_len - len(ids))
        return np.asarray(ids, dtype=np.int64)


class ToxicChatDataset(Dataset):
    """Wraps a DataFrame of (text, label) rows as encoded tensors."""

    def __init__(self, df: pd.DataFrame, vocab: Vocab, max_len: int = config.MAX_LEN):
        self.vocab = vocab
        self.max_len = max_len
        self.encoded = [vocab.encode(clean_text(t), max_len) for t in df["text"]]
        self.labels = df["label"].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.encoded[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


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
    """Return raw (train, val, test) DataFrames; shared by LSTM and baseline."""
    df = pd.read_csv(csv_path)
    required = {"text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(
            "Expected a CSV with text,label columns. Run prepare_tribunal.py first."
        )
    return _split(df)


def get_dataloaders(csv_path=config.DATA_PATH, batch_size: int = config.BATCH_SIZE):
    """Build the train/val/test DataLoaders plus the fitted vocab.

    Returns (train_dl, val_dl, test_dl, vocab) so callers can size the embedding
    table from len(vocab).
    """
    train_df, val_df, test_df = load_splits(csv_path)
    vocab = Vocab.build(
        (clean_text(t) for t in train_df["text"]),
        max_size=config.MAX_VOCAB_SIZE,
        min_freq=config.MIN_FREQ,
    )
    train_ds = ToxicChatDataset(train_df, vocab)
    val_ds = ToxicChatDataset(val_df, vocab)
    test_ds = ToxicChatDataset(test_df, vocab)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    test_dl = DataLoader(test_ds, batch_size=batch_size)
    return train_dl, val_dl, test_dl, vocab
