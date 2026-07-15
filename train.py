"""Train the LSTM and save complete, reproducible evaluation evidence."""
import csv
import json
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
from dataset import get_dataloaders, load_splits
from model import ToxicChatLSTM


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metric_summary(labels, probabilities, loss: float) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities > 0.5).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Run one train/evaluation pass and return metrics plus predictions."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total = 0.0, 0
    all_labels, all_probabilities = [], []
    with torch.set_grad_enabled(is_train):
        for x, lengths, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, lengths)
            loss = criterion(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * y.size(0)
            total += y.size(0)
            all_labels.extend(y.detach().cpu().numpy().tolist())
            all_probabilities.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())

    metrics = metric_summary(all_labels, all_probabilities, total_loss / total)
    return metrics, all_labels, all_probabilities


def save_predictions(path, texts, labels, probabilities) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["text", "label", "prediction", "probability"],
        )
        writer.writeheader()
        for text, label, probability in zip(texts, labels, probabilities):
            writer.writerow(
                {
                    "text": text,
                    "label": int(label),
                    "prediction": int(probability > 0.5),
                    "probability": f"{probability:.6f}",
                }
            )


def main():
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_dl, val_dl, test_dl, vocab = get_dataloaders()
    model = ToxicChatLSTM(vocab_size=len(vocab)).to(device)

    positives = float(train_dl.dataset.labels.sum())
    negatives = float(len(train_dl.dataset.labels) - positives)
    positive_weight = negatives / positives
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    config.CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = 0
    history = []

    for epoch in range(1, config.EPOCHS + 1):
        train_metrics, _, _ = run_epoch(model, train_dl, criterion, device, optimizer)
        val_metrics, _, _ = run_epoch(model, val_dl, criterion, device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        marker = ""
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocab_stoi": vocab.stoi,
                    "best_epoch": best_epoch,
                    "positive_weight": positive_weight,
                },
                config.CKPT_PATH,
            )
            marker = "  <- saved"

        print(
            f"Epoch {epoch:02d}/{config.EPOCHS} | "
            f"train loss {train_metrics['loss']:.4f} "
            f"bal_acc {train_metrics['balanced_accuracy']:.3f} "
            f"f1 {train_metrics['f1']:.3f} | "
            f"val loss {val_metrics['loss']:.4f} "
            f"bal_acc {val_metrics['balanced_accuracy']:.3f} "
            f"f1 {val_metrics['f1']:.3f}{marker}"
        )

    checkpoint = torch.load(
        config.CKPT_PATH,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state"])
    test_metrics, test_labels, test_probabilities = run_epoch(
        model, test_dl, criterion, device
    )
    _, _, test_df = load_splits()

    result = {
        "dataset": str(config.DATA_PATH.relative_to(config.ROOT)),
        "device": str(device),
        "best_epoch": best_epoch,
        "positive_weight": positive_weight,
        "vocab_size": len(vocab),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "hyperparameters": {
            "max_len": config.MAX_LEN,
            "embedding_dim": config.EMBED_DIM,
            "hidden_dim": config.HIDDEN_DIM,
            "num_layers": config.NUM_LAYERS,
            "dropout": config.DROPOUT,
            "batch_size": config.BATCH_SIZE,
            "epochs": config.EPOCHS,
            "learning_rate": config.LR,
        },
        "history": history,
        "test": test_metrics,
    }
    (config.RESULTS_DIR / "lstm_metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    save_predictions(
        config.RESULTS_DIR / "lstm_predictions.csv",
        test_df["raw_text"] if "raw_text" in test_df else test_df["text"],
        test_labels,
        test_probabilities,
    )
    print(
        "\nBest model test metrics: "
        f"accuracy {test_metrics['accuracy']:.3f}, "
        f"balanced accuracy {test_metrics['balanced_accuracy']:.3f}, "
        f"F1 {test_metrics['f1']:.3f}"
    )


if __name__ == "__main__":
    main()
