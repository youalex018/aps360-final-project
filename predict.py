"""Terminal scorer for the frozen primary hybrid model.

Resolves the validation-winning hybrid from
``artifacts/frozen_context_hybrid_config.json`` (seed-42 checkpoint + fusion
alpha/threshold), falling back to ``artifacts/best_model.pt``. Score one line
interactively, with ``--once``, or batch-score ``data/final_test/final_chat.csv``.

python predict.py --once "gg ez mid diff" --show-tokens
python predict.py --csv
python predict.py --csv data/final_test/final_chat.csv --output artifacts/final_test/final_hybrid_predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

import config
from dataset import Vocab, clean_text, load_splits
from hybrid import fit_lexical_scorer, fuse, svm_probabilities
from train import build_model, experiment_config_from_dict

FROZEN_HYBRID_CONFIG = config.ARTIFACTS_DIR / "frozen_context_hybrid_config.json"
FROZEN_IMPROVED_HYBRID_CONFIG = (
    config.ARTIFACTS_DIR / "frozen_improved_hybrid_config.json"
)
DEFAULT_HYBRID_JSON = config.EXPERIMENTS_DIR / "weight7_hybrid_late_seed42.json"
DEFAULT_FINAL_TEST_CSV = config.ROOT / "data" / "final_test" / "final_chat.csv"
DEFAULT_FINAL_TEST_OUTPUT = config.FINAL_TEST_ARTIFACTS_DIR / "final_hybrid_predictions.csv"
DEFAULT_BATCH_B_CSV = (
    config.ROOT / "data" / "final_test" / "batch_b" / "final_chat.csv"
)
DEFAULT_BATCH_B_OUTPUT = (
    config.FINAL_TEST_ARTIFACTS_DIR / "batch_b" / "final_hybrid_predictions.csv"
)
FALLBACK_CHECKPOINT = config.EXPERIMENTS_DIR / "weight_7_seed42.pt"


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


@dataclass(frozen=True)
class ResolvedAssets:
    """Checkpoint and fusion parameters for the frozen primary hybrid."""

    checkpoint_path: Path
    alpha: float
    threshold: float
    configuration_id: str
    hybrid_json: Path | None = None


def _repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return config.ROOT / path


def resolve_frozen_assets(
    *,
    frozen_config: Path = FROZEN_HYBRID_CONFIG,
    hybrid_json: Path | None = None,
    checkpoint_path: Path | None = None,
) -> ResolvedAssets:
    """Prefer the frozen winner; allow explicit overrides."""

    if checkpoint_path is not None and hybrid_json is not None:
        alpha, threshold, configuration_id = load_fusion_params(hybrid_json)
        return ResolvedAssets(
            checkpoint_path=checkpoint_path,
            alpha=alpha,
            threshold=threshold,
            configuration_id=configuration_id,
            hybrid_json=hybrid_json,
        )

    if frozen_config.is_file() and checkpoint_path is None and hybrid_json is None:
        with frozen_config.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        top_two = payload.get("top_two")
        if not isinstance(top_two, list) or not top_two:
            raise ValueError(f"{frozen_config} is missing a non-empty top_two list")
        winner = top_two[0]
        configuration_id = str(winner.get("configuration_id", "unknown"))
        runs = winner.get("runs")
        if not isinstance(runs, list) or not runs:
            raise ValueError(
                f"{frozen_config} winner {configuration_id!r} has no runs"
            )
        seed42 = next((run for run in runs if int(run.get("seed", -1)) == 42), runs[0])
        ckpt = _repo_path(seed42["checkpoint_path"])
        fusion = seed42.get("fusion")
        if not isinstance(fusion, dict):
            raise ValueError(
                f"{frozen_config} seed run is missing a fusion block for {configuration_id}"
            )
        if "alpha" not in fusion or "threshold" not in fusion:
            raise ValueError(
                f"{frozen_config} fusion block must include alpha and threshold"
            )
        if not ckpt.is_file():
            # Canonical copy written by the LSTM screen / train entrypoint.
            if config.CKPT_PATH.is_file():
                ckpt = config.CKPT_PATH
            else:
                raise FileNotFoundError(
                    f"Frozen checkpoint missing at {ckpt} and {config.CKPT_PATH}."
                )
        return ResolvedAssets(
            checkpoint_path=ckpt,
            alpha=float(fusion["alpha"]),
            threshold=float(fusion["threshold"]),
            configuration_id=configuration_id,
            hybrid_json=None,
        )

    resolved_hybrid = hybrid_json if hybrid_json is not None else DEFAULT_HYBRID_JSON
    alpha, threshold, configuration_id = load_fusion_params(resolved_hybrid)
    ckpt = resolve_checkpoint(checkpoint_path)
    return ResolvedAssets(
        checkpoint_path=ckpt,
        alpha=alpha,
        threshold=threshold,
        configuration_id=configuration_id,
        hybrid_json=resolved_hybrid,
    )


def resolve_checkpoint(path: Path | None = None) -> Path:
    """Prefer ``best_model.pt``, then the seed-42 weight_7 experiment checkpoint."""
    if path is not None:
        return path
    if config.CKPT_PATH.is_file():
        return config.CKPT_PATH
    if FALLBACK_CHECKPOINT.is_file():
        return FALLBACK_CHECKPOINT
    raise FileNotFoundError(
        f"No LSTM checkpoint found at {config.CKPT_PATH} or {FALLBACK_CHECKPOINT}. "
        "Train or copy the frozen weight_7 checkpoint under artifacts/ first."
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
    hybrid_json: Path | None = None,
    frozen_config: Path = FROZEN_HYBRID_CONFIG,
    device_name: str | None = None,
) -> Predictor:
    """Load LSTM weights, restore vocab, and refit the train-only lexical scorer."""
    assets = resolve_frozen_assets(
        frozen_config=frozen_config,
        hybrid_json=hybrid_json,
        checkpoint_path=checkpoint_path,
    )
    ckpt_path = assets.checkpoint_path
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"LSTM checkpoint not found: {ckpt_path}")
    if not config.DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Prepared dataset not found: {config.DATA_PATH}. "
            "Run: python prepare_l2dtnh.py"
        )

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
        alpha=assets.alpha,
        threshold=assets.threshold,
        configuration_id=assets.configuration_id,
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
        f"alpha={result['alpha']:.1f}, thr={result['threshold']:.2f})"
    )


def load_final_test_messages(csv_path: Path) -> list[dict[str, str]]:
    """Load anonymized final-test rows; labels may be blank."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Final-test CSV not found: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"match_id", "message_order", "text"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"{csv_path} must include columns match_id, message_order, text"
            )
        rows = []
        for row in reader:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            rows.append(
                {
                    "match_id": row["match_id"],
                    "message_order": row["message_order"],
                    "text": text,
                    "label": (row.get("label") or "").strip(),
                    "notes": (row.get("notes") or "").strip(),
                }
            )
    return rows


def score_csv(
    predictor: Predictor,
    csv_path: Path,
    output_path: Path | None = None,
) -> Path:
    """Score every non-empty final-test message and write a predictions CSV."""
    rows = load_final_test_messages(csv_path)
    if output_path is None:
        if csv_path.resolve() == DEFAULT_FINAL_TEST_CSV.resolve():
            output_path = DEFAULT_FINAL_TEST_OUTPUT
        else:
            output_path = csv_path.with_name(f"{csv_path.stem}_predictions.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "match_id",
        "message_order",
        "text",
        "human_label",
        "prediction",
        "label",
        "probability",
        "p_lstm",
        "p_svm",
        "tokens",
        "notes",
    ]
    toxic_count = 0
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            result = score_message(predictor, row["text"])
            toxic_count += int(result["prediction"])
            writer.writerow(
                {
                    "match_id": row["match_id"],
                    "message_order": row["message_order"],
                    "text": row["text"],
                    "human_label": row["label"],
                    "prediction": result["prediction"],
                    "label": result["label"],
                    "probability": f"{result['probability']:.6f}",
                    "p_lstm": f"{result['p_lstm']:.6f}",
                    "p_svm": f"{result['p_svm']:.6f}",
                    "tokens": " ".join(result["tokens"]),
                    "notes": row["notes"],
                }
            )

    print(
        f"Scored {len(rows)} messages from {csv_path} -> {output_path} "
        f"({toxic_count} predicted toxic / {len(rows) - toxic_count} non-toxic)"
    )
    return output_path


def run_repl(predictor: Predictor) -> None:
    """Interactive loop: type a chat line, get a prediction."""
    print(
        f"Loaded {predictor.configuration_id} from {predictor.checkpoint_path.name} "
        f"on {predictor.device}  (alpha={predictor.alpha:.1f}, thr={predictor.threshold:.2f})"
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
        help=(
            "LSTM checkpoint override (default: frozen winner checkpoint, else "
            f"{config.CKPT_PATH})"
        ),
    )
    parser.add_argument(
        "--hybrid-json",
        type=Path,
        default=None,
        help=(
            "Hybrid run JSON override for fusion alpha/threshold "
            f"(default: read from {FROZEN_HYBRID_CONFIG.name})"
        ),
    )
    parser.add_argument(
        "--frozen-config",
        type=Path,
        default=None,
        help="Frozen hybrid selection JSON used to pick the primary checkpoint",
    )
    parser.add_argument(
        "--use-improved",
        action="store_true",
        help=(
            f"Prefer {FROZEN_IMPROVED_HYBRID_CONFIG.name} when present "
            "(used automatically with --batch-b)"
        ),
    )
    parser.add_argument(
        "--batch-b",
        action="store_true",
        help=(
            "Score Batch B CSV "
            f"({DEFAULT_BATCH_B_CSV}) into {DEFAULT_BATCH_B_OUTPUT} "
            "using the improved freeze when available"
        ),
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
    parser.add_argument(
        "--csv",
        type=Path,
        nargs="?",
        const=DEFAULT_FINAL_TEST_CSV,
        default=None,
        help=(
            "Score messages from a final-test CSV "
            f"(default path if flag alone: {DEFAULT_FINAL_TEST_CSV})"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Predictions CSV path (default: artifacts/final_test/final_hybrid_predictions.csv "
            "for the canonical final-test CSV, else <csv_stem>_predictions.csv beside the input)"
        ),
    )
    parser.add_argument(
        "--show-tokens",
        action="store_true",
        help="With --once, also print the cleaned token list after slang expansion",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_b:
        args.csv = args.csv or DEFAULT_BATCH_B_CSV
        args.output = args.output or DEFAULT_BATCH_B_OUTPUT
        args.use_improved = True
    frozen_config = args.frozen_config
    if frozen_config is None:
        if args.use_improved and FROZEN_IMPROVED_HYBRID_CONFIG.is_file():
            frozen_config = FROZEN_IMPROVED_HYBRID_CONFIG
        else:
            frozen_config = FROZEN_HYBRID_CONFIG
    try:
        predictor = load_predictor(
            checkpoint_path=args.checkpoint,
            hybrid_json=args.hybrid_json,
            frozen_config=frozen_config,
            device_name=args.device,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Using {predictor.configuration_id} <- {predictor.checkpoint_path} "
        f"(alpha={predictor.alpha:.1f}, thr={predictor.threshold:.2f})",
        file=sys.stderr,
    )

    if args.csv is not None:
        try:
            score_csv(predictor, args.csv, args.output)
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.once is not None:
        result = score_message(predictor, args.once)
        print(format_score(result))
        if args.show_tokens:
            print(f"tokens: {result['tokens']}")
        return 0

    run_repl(predictor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
