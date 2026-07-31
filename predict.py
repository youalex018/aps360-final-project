"""Interactive terminal scorer for the frozen primary hybrid model.

Loads the seed-42 ``weight7_hybrid_late`` assets: single-message ``weight_7``
LSTM checkpoint plus train-only TF-IDF LinearSVC late fusion (alpha / threshold
from the hybrid run JSON). Type a chat line to get the predicted label and
probabilities.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

import config
from dataset import Vocab, clean_text, load_splits
from hybrid import fit_lexical_scorer, fuse, svm_probabilities
from train import build_model, experiment_config_from_dict

DEFAULT_CHECKPOINT = config.EXPERIMENTS_DIR / "weight_7_seed42.pt"
DEFAULT_HYBRID_JSON = config.EXPERIMENTS_DIR / "weight7_hybrid_late_seed42.json"


@dataclass
class Predictor:
    """Loaded hybrid scorer ready for single-message inference."""

    model: torch.nn.Module
    vocab: Vocab
    max_len: int
    device: torch.device
    clf: object
    vectorizer: object
    alpha: float
    threshold: float
    configuration_id: str
    checkpoint_path: Path


def resolve_checkpoint(path: Path | None = None) -> Path:
    """Prefer the named seed-42 checkpoint; fall back to ``config.CKPT_PATH``."""
    if path is not None:
        return path
    if DEFAULT_CHECKPOINT.is_file():
        return DEFAULT_CHECKPOINT
    if config.CKPT_PATH.is_file():
        return config.CKPT_PATH
    raise FileNotFoundError(
        f"No LSTM checkpoint found at {DEFAULT_CHECKPOINT} or {config.CKPT_PATH}. "
        "Train or copy weight_7_seed42.pt under artifacts/experiments/ first."
    )


def load_fusion_params(hybrid_json: Path) -> tuple[float, float, str]:
    """Read alpha, threshold, and configuration_id from a hybrid run JSON."""
    if not hybrid_json.is_file():
        raise FileNotFoundError(
            f"Hybrid run JSON not found: {hybrid_json}. "
            "Expected weight7_hybrid_late_seed42.json under artifacts/experiments/."
        )
    with hybrid_json.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    fusion = payload.get("fusion")
    if not isinstance(fusion, dict):
        raise ValueError(f"{hybrid_json} is missing a fusion block")
    if "alpha" not in fusion or "threshold" not in fusion:
        raise ValueError(f"{hybrid_json} fusion block must include alpha and threshold")
    configuration_id = str(payload.get("configuration_id", hybrid_json.stem))
    return float(fusion["alpha"]), float(fusion["threshold"]), configuration_id


def load_predictor(
    *,
    checkpoint_path: Path | None = None,
    hybrid_json: Path = DEFAULT_HYBRID_JSON,
    device_name: str | None = None,
) -> Predictor:
    """Load LSTM weights, restore vocab, and refit the train-only lexical scorer."""
    ckpt_path = resolve_checkpoint(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"LSTM checkpoint not found: {ckpt_path}")
    if not config.DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Prepared dataset not found: {config.DATA_PATH}. "
            "Run: python prepare_l2dtnh.py"
        )

    alpha, threshold, configuration_id = load_fusion_params(hybrid_json)
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    for key in ("model_state", "vocab_stoi", "experiment_config"):
        if key not in checkpoint:
            raise KeyError(f"Checkpoint {ckpt_path} is missing required key {key!r}")

    spec = experiment_config_from_dict(checkpoint["experiment_config"])
    vocab = Vocab(checkpoint["vocab_stoi"])
    model = build_model(spec, len(vocab)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    train_df, _, _ = load_splits()
    clf, vectorizer = fit_lexical_scorer(train_df["text"], train_df["label"])
    max_len = int(spec.max_len) if spec.max_len is not None else config.MAX_LEN

    return Predictor(
        model=model,
        vocab=vocab,
        max_len=max_len,
        device=device,
        clf=clf,
        vectorizer=vectorizer,
        alpha=alpha,
        threshold=threshold,
        configuration_id=configuration_id,
        checkpoint_path=ckpt_path,
    )


@torch.inference_mode()
def score_message(predictor: Predictor, text: str) -> dict:
    """Score one raw chat line; returns fused and component probabilities."""
    tokens = clean_text(text)
    length = max(1, min(len(tokens), predictor.max_len))
    ids = torch.from_numpy(predictor.vocab.encode(tokens, predictor.max_len)).unsqueeze(0)
    lengths = torch.tensor([length], dtype=torch.long)
    ids = ids.to(predictor.device)
    lengths = lengths.to(predictor.device)

    logits = predictor.model(ids, lengths)
    p_lstm = float(torch.sigmoid(logits).item())
    p_svm = float(svm_probabilities(predictor.clf, predictor.vectorizer, [text])[0])
    p_fused = float(fuse([p_lstm], [p_svm], predictor.alpha)[0])
    toxic = p_fused >= predictor.threshold
    return {
        "text": text,
        "label": "toxic" if toxic else "non-toxic",
        "prediction": int(toxic),
        "probability": p_fused,
        "p_lstm": p_lstm,
        "p_svm": p_svm,
        "alpha": predictor.alpha,
        "threshold": predictor.threshold,
        "tokens": tokens,
    }


def format_score(result: dict) -> str:
    """Human-readable one-line score summary."""
    return (
        f"{result['label']}  "
        f"p={result['probability']:.4f}  "
        f"(lstm={result['p_lstm']:.4f}, svm={result['p_svm']:.4f}, "
        f"α={result['alpha']:.1f}, thr={result['threshold']:.2f})"
    )


def run_repl(predictor: Predictor) -> None:
    """Interactive loop: type a chat line, get a prediction."""
    print(
        f"Loaded {predictor.configuration_id} from {predictor.checkpoint_path.name} "
        f"on {predictor.device}  (α={predictor.alpha:.1f}, thr={predictor.threshold:.2f})"
    )
    print("Type a chat message. Commands: quit / exit (or empty line / Ctrl+D).")
    while True:
        try:
            line = input("chat> ")
        except EOFError:
            print()
            break
        text = line.strip()
        if not text or text.lower() in {"quit", "exit"}:
            break
        print(format_score(score_message(predictor, text)))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=f"LSTM checkpoint (default: {DEFAULT_CHECKPOINT} or {config.CKPT_PATH})",
    )
    parser.add_argument(
        "--hybrid-json",
        type=Path,
        default=DEFAULT_HYBRID_JSON,
        help="Hybrid run JSON providing fusion alpha and threshold",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
        help="Inference device (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--once",
        type=str,
        default=None,
        help="Score a single message and exit (non-interactive)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        predictor = load_predictor(
            checkpoint_path=args.checkpoint,
            hybrid_json=args.hybrid_json,
            device_name=args.device,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.once is not None:
        print(format_score(score_message(predictor, args.once)))
        return 0

    run_repl(predictor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
