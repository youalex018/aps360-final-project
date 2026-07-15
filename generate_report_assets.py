"""Generate progress-report figures and qualitative examples from saved runs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

import config

FIRST_ITERATION = {
    "train_loss": [0.6886, 0.6794, 0.6646, 0.6392, 0.6048, 0.5648, 0.5285, 0.4979, 0.4736, 0.4553],
    "val_loss": [0.6856, 0.6858, 0.6941, 0.7138, 0.7367, 0.7900, 0.8743, 0.9291, 0.9868, 1.0391],
    "train_accuracy": [0.537, 0.570, 0.597, 0.628, 0.656, 0.684, 0.706, 0.724, 0.738, 0.749],
    "val_accuracy": [0.547, 0.550, 0.551, 0.548, 0.549, 0.547, 0.544, 0.542, 0.544, 0.542],
    "lstm_test_accuracy": 0.545,
    "svm_test_accuracy": 0.552,
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def add_box(ax, xy, width, height, text, fontsize=9):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02",
        linewidth=1.2,
        edgecolor="#333333",
        facecolor="#f2f4f7",
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
    )


def add_arrow(ax, start, end):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.3},
    )


def create_data_pipeline(audit: dict, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 2.5))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 2.5)
    ax.axis("off")
    boxes = [
        (0.1, "L2DTnH\n15,171 expert-\nlabelled messages"),
        (
            2.2,
            "Required fields\n5 missing text\n116 missing ID\n−116 rows",
        ),
        (
            4.3,
            "Normalize\nHTML + NFKD\nlowercase + slang\n−341 empty",
        ),
        (
            6.4,
            "Quality filter\n42 contradictory\nnormalized texts\n−421 rows",
        ),
        (
            8.5,
            "Grouped split\n14,293 rows\n70/15/15 by match\n0 group leakage",
        ),
    ]
    for x, text in boxes:
        add_box(ax, (x, 0.65), 1.75, 1.2, text)
    for x in [1.85, 3.95, 6.05, 8.15]:
        add_arrow(ax, (x, 1.25), (x + 0.3, 1.25))
    fig.suptitle(
        "Revised message-level data pipeline",
        fontsize=13,
        fontweight="bold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.04,
        (
            f"Prepared labels: {audit['prepared_label_counts']['0']} non-toxic / "
            f"{audit['prepared_label_counts']['1']} toxic; "
            f"median {audit['token_length']['median']:.0f} tokens."
        ),
        ha="center",
        fontsize=9,
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_learning_curves(lstm: dict, output: Path) -> None:
    history = lstm["history"]
    epochs = [item["epoch"] for item in history]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))

    axes[0].plot(range(1, 11), FIRST_ITERATION["train_loss"], label="Train")
    axes[0].plot(range(1, 11), FIRST_ITERATION["val_loss"], label="Validation")
    axes[0].axvline(1, color="#555555", linestyle=":", linewidth=1, label="Best epoch")
    axes[0].set_title("First iteration: role-proxy labels")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary cross-entropy loss")
    axes[0].legend(fontsize=8)

    axes[1].plot(
        epochs,
        [item["train"]["loss"] for item in history],
        label="Train",
    )
    axes[1].plot(
        epochs,
        [item["val"]["loss"] for item in history],
        label="Validation",
    )
    axes[1].axvline(
        lstm["best_epoch"],
        color="#555555",
        linestyle=":",
        linewidth=1,
        label=f"Best epoch {lstm['best_epoch']}",
    )
    axes[1].set_title("Revised: expert message labels")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Weighted BCE loss")
    axes[1].legend(fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_xticks(range(1, 11))
    fig.suptitle("LSTM learning curves", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_confusion_matrices(baseline: dict, lstm: dict, output: Path) -> None:
    matrices = [
        np.asarray(baseline["test"]["confusion_matrix"]),
        np.asarray(lstm["test"]["confusion_matrix"]),
    ]
    titles = ["TF-IDF + LinearSVC", "Packed LSTM"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, matrix, title in zip(axes, matrices, titles):
        image = ax.imshow(matrix, cmap="Blues")
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(col, row, str(value), ha="center", va="center", fontsize=11)
        ax.set_title(title)
        ax.set_xticks([0, 1], ["Non-toxic", "Toxic"])
        ax.set_yticks([0, 1], ["Non-toxic", "Toxic"])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Expert-labelled grouped test set (n=2,134)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_architecture_comparison(baseline: dict, lstm: dict, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.4, 3.4))
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    ax.text(0.1, 2.85, "Baseline", fontsize=12, fontweight="bold")
    ax.text(0.1, 1.25, "Primary model", fontsize=12, fontweight="bold")

    baseline_boxes = [
        (1.5, "Normalized\nmessage"),
        (3.5, f"TF-IDF\n{baseline['vocabulary_size']:,} terms"),
        (5.5, "LinearSVC\nbalanced classes"),
        (
            7.5,
            f"Toxic / non-toxic\nF1={baseline['test']['f1']:.3f}\n"
            f"Bal. acc={baseline['test']['balanced_accuracy']:.3f}",
        ),
    ]
    lstm_boxes = [
        (1.5, "≤50 token IDs"),
        (3.1, "Embedding\n100 dimensions"),
        (4.7, "Packed LSTM\n128 hidden × 2"),
        (6.3, "Dropout 0.3\n+ Linear(1)"),
        (
            7.9,
            f"Sigmoid output\nF1={lstm['test']['f1']:.3f}\n"
            f"Bal. acc={lstm['test']['balanced_accuracy']:.3f}",
        ),
    ]
    for x, text in baseline_boxes:
        add_box(ax, (x, 2.35), 1.45, 0.8, text, fontsize=8)
    for x, text in lstm_boxes:
        add_box(ax, (x, 0.65), 1.3, 0.9, text, fontsize=8)
    for x1, x2 in zip([2.95, 4.95, 6.95], [3.5, 5.5, 7.5]):
        add_arrow(ax, (x1, 2.75), (x2, 2.75))
    for x1, x2 in zip([2.8, 4.4, 6.0, 7.6], [3.1, 4.7, 6.3, 7.9]):
        add_arrow(ax, (x1, 1.1), (x2, 1.1))
    ax.text(
        9.35,
        0.2,
        f"{lstm['parameter_count']:,} trainable parameters",
        ha="center",
        fontsize=8,
    )
    fig.suptitle(
        "Comparable models: identical normalization and grouped splits",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_qualitative_examples(output: Path) -> None:
    lstm = pd.read_csv(config.RESULTS_DIR / "lstm_predictions.csv")
    baseline = pd.read_csv(config.RESULTS_DIR / "baseline_predictions.csv")
    combined = lstm.rename(
        columns={
            "prediction": "lstm_prediction",
            "probability": "lstm_probability",
        }
    )
    combined["svm_prediction"] = baseline["prediction"]
    combined["svm_margin"] = baseline["margin"]

    selections: list[tuple[str, pd.Series, str]] = []
    slang_mask = combined["text"].str.lower().str.contains(
        r"\b(?:jg|ff|int|kys|noob|afk|bot lane|diff)\b",
        regex=True,
    )
    correct_both = (combined["lstm_prediction"] == combined["label"]) & (
        combined["svm_prediction"] == combined["label"]
    )
    if (slang_mask & correct_both).any():
        row = combined[slang_mask & correct_both].iloc[0]
        selections.append(("slang success", row, "Both models handle game-specific language."))

    cases = [
        (
            "LSTM false positive",
            (combined["label"] == 0) & (combined["lstm_prediction"] == 1),
            "A benign message receives a high toxicity probability.",
            "lstm_probability",
            False,
        ),
        (
            "LSTM false negative",
            (combined["label"] == 1) & (combined["lstm_prediction"] == 0),
            "The sequential model misses an expert-labelled toxic message.",
            "lstm_probability",
            True,
        ),
        (
            "SVM-only success",
            (combined["svm_prediction"] == combined["label"])
            & (combined["lstm_prediction"] != combined["label"]),
            "Bag-of-words succeeds where the LSTM fails.",
            "svm_margin",
            False,
        ),
    ]
    for category, mask, comment, column, ascending in cases:
        candidates = combined[mask].sort_values(column, ascending=ascending)
        if not candidates.empty:
            selections.append((category, candidates.iloc[0], comment))

    fields = [
        "category",
        "text",
        "label",
        "lstm_probability",
        "lstm_prediction",
        "svm_margin",
        "svm_prediction",
        "comment",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for category, row, comment in selections:
            writer.writerow(
                {
                    "category": category,
                    "text": row["text"],
                    "label": int(row["label"]),
                    "lstm_probability": f"{row['lstm_probability']:.3f}",
                    "lstm_prediction": int(row["lstm_prediction"]),
                    "svm_margin": f"{row['svm_margin']:.3f}",
                    "svm_prediction": int(row["svm_prediction"]),
                    "comment": comment,
                }
            )


def main() -> None:
    audit = read_json(config.RESULTS_DIR / "data_audit.json")
    baseline = read_json(config.RESULTS_DIR / "baseline_metrics.json")
    lstm = read_json(config.RESULTS_DIR / "lstm_metrics.json")
    config.REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    create_data_pipeline(audit, config.REPORT_FIGURES_DIR / "data_pipeline.png")
    create_learning_curves(lstm, config.REPORT_FIGURES_DIR / "learning_curves.png")
    create_confusion_matrices(
        baseline,
        lstm,
        config.REPORT_FIGURES_DIR / "confusion_matrices.png",
    )
    create_architecture_comparison(
        baseline,
        lstm,
        config.REPORT_FIGURES_DIR / "architecture_comparison.png",
    )
    build_qualitative_examples(config.RESULTS_DIR / "qualitative_examples.csv")
    print(f"Wrote report assets to {config.REPORT_FIGURES_DIR}")


if __name__ == "__main__":
    main()
