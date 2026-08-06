"""Train and evaluate LSTM configurations without test-set tuning."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

import config
from dataset import Vocab, get_dataloaders, load_splits
from model import ToxicChatLSTM


@dataclass(frozen=True)
class ExperimentConfig:
    config_id: str = "corrected_current"
    seed: int = config.SEED
    embed_dim: int = config.EMBED_DIM
    hidden_dim: int = config.HIDDEN_DIM
    num_layers: int = config.NUM_LAYERS
    dropout: float = config.DROPOUT
    bidirectional: bool = False
    pooling: str = "last"
    positive_weight: float | None = None
    optimizer: str = "adam"
    weight_decay: float = config.WEIGHT_DECAY
    learning_rate: float = config.LR
    batch_size: int = config.BATCH_SIZE
    max_epochs: int = config.EPOCHS
    patience: int = config.EARLY_STOP_PATIENCE
    embedding_path: str | None = None
    freeze_embeddings: bool = False
    context_k: int | None = None
    max_len: int | None = None
    hybrid: bool = False


def set_seed(seed: int) -> None:
    """Seed all RNGs and request deterministic CPU/CUDA kernels."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metric_summary(
    labels,
    probabilities,
    loss: float,
    threshold: float = 0.5,
) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    return {
        "loss": float(loss),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
    }


def select_validation_threshold(
    labels,
    probabilities,
    loss: float,
    threshold_grid=config.THRESHOLD_GRID,
) -> dict:
    """Select the predeclared threshold with best toxic-class F1."""
    candidates = [
        metric_summary(labels, probabilities, loss, threshold)
        for threshold in threshold_grid
    ]
    return max(
        candidates,
        key=lambda item: (
            item["f1"],
            item["precision"],
            item["balanced_accuracy"],
            -abs(item["threshold"] - 0.5),
            -item["threshold"],
        ),
    )


def checkpoint_is_better(candidate: dict, incumbent: dict | None) -> bool:
    """Use validation F1 first and lower validation loss only to break ties."""
    if incumbent is None:
        return True
    if candidate["f1"] > incumbent["f1"] + 1e-12:
        return True
    return (
        abs(candidate["f1"] - incumbent["f1"]) <= 1e-12
        and candidate["loss"] < incumbent["loss"] - 1e-12
    )


def run_epoch(model, loader, criterion, device, optimizer=None, threshold=0.5):
    """Run one train/evaluation pass and return metrics plus predictions."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, total = 0.0, 0
    all_labels, all_probabilities = [], []
    with torch.set_grad_enabled(is_train):
        for x, lengths, y in loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            logits = model(x, lengths)
            loss = criterion(logits, y)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            total += y.size(0)
            all_labels.extend(y.detach().cpu().numpy().tolist())
            all_probabilities.extend(
                torch.sigmoid(logits).detach().cpu().numpy().tolist()
            )
    average_loss = total_loss / max(total, 1)
    metrics = metric_summary(
        all_labels,
        all_probabilities,
        average_loss,
        threshold,
    )
    return metrics, all_labels, all_probabilities


def _load_pretrained(spec: ExperimentConfig):
    if spec.embedding_path is None:
        return None, None
    payload = torch.load(spec.embedding_path, map_location="cpu", weights_only=True)
    return payload["matrix"], payload.get("coverage")


def build_model(spec: ExperimentConfig, vocab_size: int) -> ToxicChatLSTM:
    pretrained, _ = _load_pretrained(spec)
    return ToxicChatLSTM(
        vocab_size=vocab_size,
        embed_dim=spec.embed_dim,
        hidden_dim=spec.hidden_dim,
        num_layers=spec.num_layers,
        dropout=spec.dropout,
        bidirectional=spec.bidirectional,
        pooling=spec.pooling,
        pretrained_embeddings=pretrained,
        freeze_embeddings=spec.freeze_embeddings,
    )


def _environment_metadata(device: torch.device) -> dict:
    packages = {}
    for package in ("numpy", "pandas", "scikit-learn", "torch"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = "unknown"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "packages": packages,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def _criterion(train_loader, positive_weight, device):
    positives = float(train_loader.dataset.labels.sum())
    negatives = float(len(train_loader.dataset.labels) - positives)
    resolved_weight = (
        negatives / positives if positive_weight is None else positive_weight
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            resolved_weight,
            dtype=torch.float32,
            device=device,
        )
    )
    return criterion, float(resolved_weight)


def _optimizer(model, spec: ExperimentConfig):
    if spec.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=spec.learning_rate,
            weight_decay=spec.weight_decay,
        )
    if spec.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=spec.learning_rate,
            weight_decay=spec.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {spec.optimizer}")


def save_predictions(path, texts, labels, probabilities, threshold) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["text", "label", "prediction", "probability", "threshold"],
        )
        writer.writeheader()
        for text, label, probability in zip(texts, labels, probabilities):
            writer.writerow(
                {
                    "text": text,
                    "label": int(label),
                    "prediction": int(probability >= threshold),
                    "probability": f"{probability:.6f}",
                    "threshold": f"{threshold:.2f}",
                }
            )


def experiment_config_from_dict(data: dict) -> ExperimentConfig:
    """Build an ExperimentConfig, ignoring unknown keys from older artifacts."""
    allowed = set(ExperimentConfig.__dataclass_fields__)
    return ExperimentConfig(**{key: value for key, value in data.items() if key in allowed})


def run_experiment(
    spec: ExperimentConfig,
    *,
    evaluate_test: bool = False,
    canonical: bool = False,
    device_name: str | None = None,
) -> dict:
    """Train from train/validation only; optionally evaluate after freezing."""
    set_seed(spec.seed)
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    train_dl, val_dl, test_dl, vocab = get_dataloaders(
        batch_size=spec.batch_size,
        seed=spec.seed,
        include_test=evaluate_test,
        context_k=spec.context_k,
        max_len=spec.max_len,
    )
    model = build_model(spec, len(vocab)).to(device)
    criterion, positive_weight = _criterion(
        train_dl, spec.positive_weight, device
    )
    optimizer = _optimizer(model, spec)

    output_dir = config.EXPERIMENTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{spec.config_id}_seed{spec.seed}"
    checkpoint_path = output_dir / f"{run_name}.pt"
    metrics_path = output_dir / f"{run_name}.json"
    best_val = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    print(f"Run: {run_name}; device: {device}; test enabled: {evaluate_test}")
    for epoch in range(1, spec.max_epochs + 1):
        train_metrics, _, _ = run_epoch(
            model, train_dl, criterion, device, optimizer
        )
        raw_val, val_labels, val_probabilities = run_epoch(
            model, val_dl, criterion, device
        )
        val_metrics = select_validation_threshold(
            val_labels,
            val_probabilities,
            raw_val["loss"],
        )
        history.append(
            {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        )
        marker = ""
        if checkpoint_is_better(val_metrics, best_val):
            best_val = val_metrics
            best_epoch = epoch
            epochs_without_improvement = 0
            _, coverage = _load_pretrained(spec)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocab_stoi": vocab.stoi,
                    "best_epoch": best_epoch,
                    "selected_threshold": val_metrics["threshold"],
                    "validation": val_metrics,
                    "positive_weight": positive_weight,
                    "experiment_config": asdict(spec),
                    "embedding_coverage": coverage,
                },
                checkpoint_path,
            )
            marker = " <- saved"
        else:
            epochs_without_improvement += 1
        print(
            f"Epoch {epoch:02d}/{spec.max_epochs} | "
            f"train loss {train_metrics['loss']:.4f} f1@.50 "
            f"{train_metrics['f1']:.3f} | val loss {val_metrics['loss']:.4f} "
            f"p {val_metrics['precision']:.3f} r {val_metrics['recall']:.3f} "
            f"f1 {val_metrics['f1']:.3f} @ {val_metrics['threshold']:.2f}"
            f"{marker}"
        )
        if epochs_without_improvement >= spec.patience:
            print(f"Early stopping after {epoch} epochs.")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    selected_threshold = float(checkpoint["selected_threshold"])
    best_raw, best_labels, best_probabilities = run_epoch(
        model,
        val_dl,
        criterion,
        device,
        threshold=selected_threshold,
    )
    best_val = metric_summary(
        best_labels,
        best_probabilities,
        best_raw["loss"],
        selected_threshold,
    )
    result = {
        "schema_version": 2,
        "dataset": str(config.DATA_PATH.relative_to(config.ROOT)),
        "run_name": run_name,
        "configuration_id": spec.config_id,
        "seed": spec.seed,
        "checkpoint_path": str(checkpoint_path.relative_to(config.ROOT)),
        "selection_metric": "validation toxic-class F1",
        "threshold_grid": list(config.THRESHOLD_GRID),
        "best_epoch": best_epoch,
        "selected_threshold": selected_threshold,
        "positive_weight": positive_weight,
        "vocab_size": len(vocab),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "experiment_config": asdict(spec),
        "environment": _environment_metadata(device),
        "history": history,
        "validation": best_val,
        "test_accessed": bool(evaluate_test),
    }
    if evaluate_test:
        assert test_dl is not None
        test_metrics, test_labels, test_probabilities = run_epoch(
            model,
            test_dl,
            criterion,
            device,
            threshold=selected_threshold,
        )
        result["test"] = test_metrics
        _, _, test_df = load_splits()
        prediction_path = (
            config.ARTIFACTS_DIR / "lstm_predictions.csv"
            if canonical
            else output_dir / f"{run_name}_test_predictions.csv"
        )
        save_predictions(
            prediction_path,
            test_df["raw_text"] if "raw_text" in test_df else test_df["text"],
            test_labels,
            test_probabilities,
            selected_threshold,
        )
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if canonical:
        config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        (config.ARTIFACTS_DIR / "lstm_metrics.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
        torch.save(checkpoint, config.CKPT_PATH)
    return result


def collect_split_probabilities(
    checkpoint_path: Path,
    *,
    split: str,
    device_name: str | None = None,
) -> tuple[list[float], list[int], ExperimentConfig]:
    """Score one split with a frozen LSTM checkpoint (no threshold retuning)."""
    if split not in {"val", "test"}:
        raise ValueError(f"split must be 'val' or 'test', got {split!r}")
    preview = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    spec = experiment_config_from_dict(preview["experiment_config"])
    set_seed(spec.seed)
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    frozen_vocab = Vocab(preview["vocab_stoi"])
    train_dl, val_dl, test_dl, vocab = get_dataloaders(
        batch_size=spec.batch_size,
        seed=spec.seed,
        include_test=split == "test",
        context_k=spec.context_k,
        max_len=spec.max_len,
        vocab=frozen_vocab,
    )
    model = build_model(spec, len(vocab)).to(device)
    model.load_state_dict(preview["model_state"])
    criterion, _ = _criterion(
        train_dl,
        float(preview["positive_weight"]),
        device,
    )
    loader = test_dl if split == "test" else val_dl
    assert loader is not None
    _, labels, probabilities = run_epoch(model, loader, criterion, device)
    return probabilities, [int(label) for label in labels], spec


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    device_name: str | None = None,
    prediction_path: Path | None = None,
) -> dict:
    """Evaluate one already-frozen checkpoint on the grouped test split."""
    preview = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    spec = experiment_config_from_dict(preview["experiment_config"])
    set_seed(spec.seed)
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    frozen_vocab = Vocab(preview["vocab_stoi"])
    train_dl, _, test_dl, vocab = get_dataloaders(
        batch_size=spec.batch_size,
        seed=spec.seed,
        include_test=True,
        context_k=spec.context_k,
        max_len=spec.max_len,
        vocab=frozen_vocab,
    )
    model = build_model(spec, len(vocab)).to(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state"])
    criterion, _ = _criterion(
        train_dl,
        float(checkpoint["positive_weight"]),
        device,
    )
    threshold = float(checkpoint["selected_threshold"])
    assert test_dl is not None
    metrics, labels, probabilities = run_epoch(
        model,
        test_dl,
        criterion,
        device,
        threshold=threshold,
    )
    if prediction_path is not None:
        _, _, test_df = load_splits()
        save_predictions(
            prediction_path,
            test_df["raw_text"] if "raw_text" in test_df else test_df["text"],
            labels,
            probabilities,
            threshold,
        )
    return {
        "configuration_id": spec.config_id,
        "seed": spec.seed,
        "checkpoint_path": str(checkpoint_path.relative_to(config.ROOT)),
        "selected_threshold": threshold,
        "validation": checkpoint["validation"],
        "test": metrics,
        "probabilities": probabilities,
        "labels": labels,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-id", default="corrected_current")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--embed-dim", type=int, default=config.EMBED_DIM)
    parser.add_argument("--hidden-dim", type=int, default=config.HIDDEN_DIM)
    parser.add_argument("--num-layers", type=int, default=config.NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=config.DROPOUT)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--pooling", choices=["last", "mean_max"], default="last")
    parser.add_argument("--positive-weight", type=float)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    parser.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--learning-rate", type=float, default=config.LR)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--patience", type=int, default=config.EARLY_STOP_PATIENCE)
    parser.add_argument("--embedding-path")
    parser.add_argument("--freeze-embeddings", action="store_true")
    parser.add_argument("--context-k", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Mark this config as hybrid (fusion is applied by hybrid.py / the context runner).",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Only use after the configuration and threshold are frozen.",
    )
    parser.add_argument("--canonical", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = ExperimentConfig(
        config_id=args.config_id,
        seed=args.seed,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        pooling=args.pooling,
        positive_weight=args.positive_weight,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        embedding_path=args.embedding_path,
        freeze_embeddings=args.freeze_embeddings,
        context_k=args.context_k,
        max_len=args.max_len,
        hybrid=args.hybrid,
    )
    result = run_experiment(
        spec,
        evaluate_test=args.evaluate_test,
        canonical=args.canonical,
        device_name=args.device,
    )
    val = result["validation"]
    print(
        f"Best validation: F1 {val['f1']:.3f}, precision "
        f"{val['precision']:.3f}, recall {val['recall']:.3f}, "
        f"threshold {result['selected_threshold']:.2f}"
    )
    if "test" in result:
        test = result["test"]
        print(
            f"Frozen test: F1 {test['f1']:.3f}, balanced accuracy "
            f"{test['balanced_accuracy']:.3f}"
        )


if __name__ == "__main__":
    main()
