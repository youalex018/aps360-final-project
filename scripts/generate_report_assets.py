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

from toxic_chat import config

ARCHIVE_DIR = config.ARTIFACTS_DIR / "archive" / "single_message_cuda_frozen"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_artifact(*relative_parts: str) -> Path:
    """Prefer live artifacts/, then the frozen single-message archive."""
    live = config.ARTIFACTS_DIR.joinpath(*relative_parts)
    if live.exists():
        return live
    archived = ARCHIVE_DIR.joinpath(*relative_parts)
    if archived.exists():
        return archived
    raise FileNotFoundError(
        f"Missing artifact {'/'.join(relative_parts)} in artifacts/ or archive."
    )


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
        (
            0.1,
            f"L2DTnH\n{audit['source_rows']:,} expert-\nlabelled messages",
        ),
        (
            2.2,
            "Required fields\n"
            f"{audit['missing_message_rows']} missing text\n"
            f"{audit['missing_group_rows']} missing ID\n"
            f"−{audit['missing_required_rows']} rows",
        ),
        (
            4.3,
            "Normalize\nHTML + NFKD\nlowercase + slang\n"
            f"−{audit['empty_after_normalization_rows']} empty",
        ),
        (
            6.4,
            "Quality filter\n"
            f"{audit['conflicting_normalized_texts']} contradictory\n"
            "normalized texts\n"
            f"−{audit['conflicting_rows_removed']} rows",
        ),
        (
            8.5,
            f"Grouped split\n{audit['prepared_rows']:,} rows\n"
            "70/15/15 by match\n0 group leakage",
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


def create_learning_curves(backbone: dict, output: Path, *, title: str) -> None:
    history = backbone["history"]
    epochs = [item["epoch"] for item in history]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))

    axes[0].plot(
        epochs,
        [item["train"]["loss"] for item in history],
        label="Train",
    )
    axes[0].plot(
        epochs,
        [item["val"]["loss"] for item in history],
        label="Validation",
    )
    axes[0].axvline(
        backbone["best_epoch"],
        color="#555555",
        linestyle=":",
        linewidth=1,
        label=f"Selected epoch {backbone['best_epoch']}",
    )
    axes[0].set_title("Weighted loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCEWithLogits loss")
    axes[0].legend(fontsize=8)

    axes[1].plot(
        epochs,
        [item["train"]["f1"] for item in history],
        label="Train F1 @ 0.50",
    )
    axes[1].plot(
        epochs,
        [item["val"]["f1"] for item in history],
        label="Validation F1 @ tuned threshold",
    )
    axes[1].axvline(
        backbone["best_epoch"],
        color="#555555",
        linestyle=":",
        linewidth=1,
        label=f"Best epoch {backbone['best_epoch']}",
    )
    aggregate = backbone["validation_aggregate"]["f1"]
    axes[1].set_title(
        f"Backbone val F1: {aggregate['mean']:.3f} ± {aggregate['std']:.3f}"
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Toxic-class F1")
    axes[1].legend(fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_xticks(epochs)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_confusion_matrices(
    baseline: dict,
    primary: dict,
    output: Path,
    *,
    primary_title: str,
) -> None:
    matrices = [
        np.asarray(baseline["test"]["confusion_matrix"]),
        np.asarray(primary["test"]["confusion_matrix"]),
    ]
    titles = ["TF-IDF + LinearSVC", primary_title]
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
    fig.suptitle(
        f"Expert-labelled grouped test set (n={matrices[0].sum():,})",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_architecture_comparison(
    baseline: dict,
    hybrid: dict,
    output: Path,
) -> None:
    fig_w, fig_h = 12.6, 4.1
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    # Same box width + same gap on both rows (baseline ends earlier).
    label_x = 0.15
    x_start = 2.4
    box_w, gap, box_h = 1.72, 0.40, 0.98
    baseline_y, hybrid_y = 2.45, 0.72
    fontsize = 8.5

    baseline_texts = [
        "Normalized\nmessage",
        f"TF-IDF\n{baseline['vocabulary_size']:,} terms",
        "LinearSVC\nbalanced classes",
        (
            f"Toxic / non-toxic\nF1={baseline['test']['f1']:.3f}\n"
            f"Bal. acc={baseline['test']['balanced_accuracy']:.3f}"
        ),
    ]
    spec = hybrid["experiment_config"]
    aggregate = hybrid["test_aggregate"]
    alphas = hybrid["validation_aggregate"].get("alphas", [])
    alpha_text = ", ".join(f"{value:.1f}" for value in alphas) if alphas else "val-tuned"
    hybrid_texts = [
        "Normalized\nmessage",
        f"LSTM\n{spec['hidden_dim']}×{spec['num_layers']}",
        "SVM\nmargin → σ",
        f"Late blend\nα ∈ {{{alpha_text}}}",
        (
            "Toxic / non-toxic\n"
            f"F1={aggregate['f1']['mean']:.3f} ± {aggregate['f1']['std']:.3f}\n"
            f"Bal. acc={aggregate['balanced_accuracy']['mean']:.3f}"
        ),
    ]

    def box_xs(n: int) -> list[float]:
        return [x_start + i * (box_w + gap) for i in range(n)]

    baseline_xs = box_xs(len(baseline_texts))
    hybrid_xs = box_xs(len(hybrid_texts))

    ax.text(
        label_x,
        baseline_y + box_h / 2,
        "Baseline",
        fontsize=12,
        fontweight="bold",
        va="center",
        ha="left",
    )
    ax.text(
        label_x,
        hybrid_y + box_h / 2,
        "Primary hybrid",
        fontsize=12,
        fontweight="bold",
        va="center",
        ha="left",
    )

    for x, text in zip(baseline_xs, baseline_texts):
        add_box(ax, (x, baseline_y), box_w, box_h, text, fontsize=fontsize)
    for x, text in zip(hybrid_xs, hybrid_texts):
        add_box(ax, (x, hybrid_y), box_w, box_h, text, fontsize=fontsize)

    for xs, y in ((baseline_xs, baseline_y), (hybrid_xs, hybrid_y)):
        for x1, x2 in zip(xs, xs[1:]):
            add_arrow(
                ax,
                (x1 + box_w + 0.03, y + box_h / 2),
                (x2 - 0.03, y + box_h / 2),
            )

    last_right = hybrid_xs[-1] + box_w
    ax.text(
        last_right,
        0.22,
        f"LSTM backbone: {hybrid['parameter_count']:,} parameters",
        ha="right",
        va="center",
        fontsize=8,
    )
    fig.suptitle(
        "Comparable models: identical normalization and grouped splits",
        fontsize=13,
        fontweight="bold",
        y=0.97,
    )
    fig.savefig(output, dpi=220, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def build_qualitative_examples(
    primary_predictions: Path,
    baseline_predictions: Path,
    output: Path,
) -> None:
    primary = pd.read_csv(primary_predictions)
    baseline = pd.read_csv(baseline_predictions)
    combined = primary.rename(
        columns={
            "prediction": "hybrid_prediction",
            "probability": "hybrid_probability",
        }
    )
    combined["svm_prediction"] = baseline["prediction"]
    combined["svm_margin"] = baseline["margin"]

    selections: list[tuple[str, pd.Series, str]] = []
    slang_mask = combined["text"].str.lower().str.contains(
        r"\b(?:jg|ff|int|kys|noob|afk|bot lane|diff)\b",
        regex=True,
    )
    correct_both = (combined["hybrid_prediction"] == combined["label"]) & (
        combined["svm_prediction"] == combined["label"]
    )
    if (slang_mask & correct_both).any():
        row = combined[slang_mask & correct_both].iloc[0]
        selections.append(
            ("slang success", row, "Both models handle game-specific language.")
        )

    cases = [
        (
            "Hybrid false positive",
            (combined["label"] == 0) & (combined["hybrid_prediction"] == 1),
            "A benign message receives a high toxicity probability.",
            "hybrid_probability",
            False,
        ),
        (
            "Hybrid false negative",
            (combined["label"] == 1) & (combined["hybrid_prediction"] == 0),
            "The hybrid misses an expert-labelled toxic message.",
            "hybrid_probability",
            True,
        ),
        (
            "SVM-only success",
            (combined["svm_prediction"] == combined["label"])
            & (combined["hybrid_prediction"] != combined["label"]),
            "Bag-of-words succeeds where the hybrid fails.",
            "svm_margin",
            False,
        ),
        (
            "Hybrid-only success",
            (combined["hybrid_prediction"] == combined["label"])
            & (combined["svm_prediction"] != combined["label"]),
            "Late fusion corrects an SVM error.",
            "hybrid_probability",
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
        "hybrid_probability",
        "hybrid_prediction",
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
                    "hybrid_probability": f"{row['hybrid_probability']:.3f}",
                    "hybrid_prediction": int(row["hybrid_prediction"]),
                    "svm_margin": f"{row['svm_margin']:.3f}",
                    "svm_prediction": int(row["svm_prediction"]),
                    "comment": comment,
                }
            )


def _tex_escape_id(value: str) -> str:
    return value.replace("_", r"\_")


def write_generated_metrics(
    audit: dict,
    baseline: dict,
    lstm: dict,
    hybrid: dict,
    summary: dict,
) -> None:
    baseline_test = baseline["test"]
    lstm_test = lstm["test_aggregate"]
    lstm_validation = lstm["validation_aggregate"]
    hybrid_test = hybrid["test_aggregate"]
    hybrid_validation = hybrid["validation_aggregate"]
    baseline_matrix = np.asarray(baseline_test["confusion_matrix"])
    lstm_matrix = np.asarray(lstm["test"]["confusion_matrix"])
    hybrid_matrix = np.asarray(hybrid["test"]["confusion_matrix"])
    lstm_spec = lstm["experiment_config"]
    hybrid_spec = hybrid["experiment_config"]

    frozen_lstm = read_json(resolve_artifact("frozen_lstm_config.json"))
    lstm_alternate = next(
        candidate
        for candidate in frozen_lstm["top_two"]
        if candidate["configuration_id"] != lstm["frozen_configuration_id"]
    )
    frozen_hybrid = read_json(resolve_artifact("frozen_context_hybrid_config.json"))
    hybrid_runner_up = next(
        candidate
        for candidate in frozen_hybrid["top_two"]
        if candidate["configuration_id"] != hybrid["frozen_configuration_id"]
    )
    context_screen = summary["context_screen"]
    best_context = max(context_screen, key=lambda run: run["validation"]["f1"])
    alphas = hybrid_validation.get("alphas", [])
    thresholds = hybrid_validation.get("thresholds", [])

    values = {
        "BaselineValidationFOne": f"{baseline['validation']['f1']:.3f}",
        "BaselineAccuracy": f"{baseline_test['accuracy']:.3f}",
        "BaselineBalancedAccuracy": f"{baseline_test['balanced_accuracy']:.3f}",
        "BaselinePrecision": f"{baseline_test['precision']:.3f}",
        "BaselineRecall": f"{baseline_test['recall']:.3f}",
        "BaselineFOne": f"{baseline_test['f1']:.3f}",
        "BaselineTruePositives": str(int(baseline_matrix[1, 1])),
        "BaselineFalsePositives": str(int(baseline_matrix[0, 1])),
        # Single-message LSTM ablation (failed attempt).
        "LSTMAccuracy": f"{lstm_test['accuracy']['mean']:.3f}",
        "LSTMBalancedAccuracy": (
            f"{lstm_test['balanced_accuracy']['mean']:.3f}"
        ),
        "LSTMPrecision": f"{lstm_test['precision']['mean']:.3f}",
        "LSTMRecall": f"{lstm_test['recall']['mean']:.3f}",
        "LSTMFOne": f"{lstm_test['f1']['mean']:.3f}",
        "LSTMFOneStd": f"{lstm_test['f1']['std']:.3f}",
        "BaselineMinusLSTMFOne": (
            f"{baseline_test['f1'] - lstm_test['f1']['mean']:.3f}"
        ),
        "LSTMEnsembleFOne": f"{lstm['ensemble_secondary']['f1']:.3f}",
        "LSTMValidationFOne": f"{lstm_validation['f1']['mean']:.3f}",
        "LSTMValidationFOneStd": f"{lstm_validation['f1']['std']:.3f}",
        "AlternateConfiguration": _tex_escape_id(
            lstm_alternate["configuration_id"]
        ),
        "AlternateValidationFOne": (
            f"{lstm_alternate['validation_aggregate']['f1']['mean']:.3f}"
        ),
        "AlternateValidationFOneStd": (
            f"{lstm_alternate['validation_aggregate']['f1']['std']:.3f}"
        ),
        "LSTMSeedFortyTwoFOne": f"{lstm['test']['f1']:.3f}",
        "LSTMTruePositives": str(int(lstm_matrix[1, 1])),
        "LSTMFalsePositives": str(int(lstm_matrix[0, 1])),
        "LSTMThreshold": f"{lstm['selected_threshold']:.2f}",
        "LSTMBestEpoch": str(lstm["best_epoch"]),
        "LSTMParameters": f"{lstm['parameter_count']:,}",
        "LSTMVocabSize": f"{lstm['vocab_size']:,}",
        "LSTMConfiguration": _tex_escape_id(lstm["frozen_configuration_id"]),
        "LSTMHiddenDimension": str(lstm_spec["hidden_dim"]),
        "LSTMLayers": str(lstm_spec["num_layers"]),
        "LSTMDropout": f"{lstm_spec['dropout']:.1f}",
        "PreparedRows": f"{audit['prepared_rows']:,}",
        # Frozen hybrid primary model.
        "HybridConfiguration": _tex_escape_id(
            hybrid["frozen_configuration_id"]
        ),
        "HybridAccuracy": f"{hybrid_test['accuracy']['mean']:.3f}",
        "HybridBalancedAccuracy": (
            f"{hybrid_test['balanced_accuracy']['mean']:.3f}"
        ),
        "HybridPrecision": f"{hybrid_test['precision']['mean']:.3f}",
        "HybridRecall": f"{hybrid_test['recall']['mean']:.3f}",
        "HybridFOne": f"{hybrid_test['f1']['mean']:.3f}",
        "HybridFOneStd": f"{hybrid_test['f1']['std']:.3f}",
        "HybridMinusBaselineFOne": (
            f"{hybrid_test['f1']['mean'] - baseline_test['f1']:.3f}"
        ),
        "HybridValidationFOne": f"{hybrid_validation['f1']['mean']:.3f}",
        "HybridValidationFOneStd": f"{hybrid_validation['f1']['std']:.3f}",
        "HybridSeedFortyTwoFOne": f"{hybrid['test']['f1']:.3f}",
        "HybridTruePositives": str(int(hybrid_matrix[1, 1])),
        "HybridFalsePositives": str(int(hybrid_matrix[0, 1])),
        "HybridThreshold": f"{hybrid['selected_threshold']:.2f}",
        "HybridAlphaSeedFortyTwo": f"{hybrid['fusion']['alpha']:.1f}",
        "HybridAlphas": ", ".join(f"{value:.1f}" for value in alphas),
        "HybridThresholds": ", ".join(f"{value:.2f}" for value in thresholds),
        "HybridBestEpoch": str(hybrid["best_epoch"]),
        "HybridParameters": f"{hybrid['parameter_count']:,}",
        "HybridVocabSize": f"{hybrid['vocab_size']:,}",
        "HybridHiddenDimension": str(hybrid_spec["hidden_dim"]),
        "HybridLayers": str(hybrid_spec["num_layers"]),
        "HybridDropout": f"{hybrid_spec['dropout']:.1f}",
        "HybridRunnerUpConfiguration": _tex_escape_id(
            hybrid_runner_up["configuration_id"]
        ),
        "HybridRunnerUpValidationFOne": (
            f"{hybrid_runner_up['validation_aggregate']['f1']['mean']:.3f}"
        ),
        "HybridRunnerUpValidationFOneStd": (
            f"{hybrid_runner_up['validation_aggregate']['f1']['std']:.3f}"
        ),
        "ContextBestConfiguration": _tex_escape_id(
            best_context["configuration_id"]
        ),
        "ContextBestValidationFOne": f"{best_context['validation']['f1']:.3f}",
        "ContextKOneValidationFOne": (
            f"{context_screen[0]['validation']['f1']:.3f}"
        ),
        "ContextKTwoValidationFOne": (
            f"{context_screen[1]['validation']['f1']:.3f}"
        ),
        "ContextKThreeValidationFOne": (
            f"{context_screen[2]['validation']['f1']:.3f}"
        ),
    }
    output = config.ROOT / "reports" / "progress" / "generated_metrics.tex"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(
            rf"\newcommand{{\{name}}}{{{value}}}"
            for name, value in values.items()
        )
        + "\n",
        encoding="utf-8",
    )
    (config.ARTIFACTS_DIR / "report_metric_manifest.json").write_text(
        json.dumps(values, indent=2),
        encoding="utf-8",
    )


def load_hybrid_bundle() -> tuple[dict, dict]:
    hybrid = read_json(resolve_artifact("context_hybrid_metrics.json"))
    summary = read_json(resolve_artifact("context_hybrid_experiment_summary.json"))
    if not hybrid.get("frozen_configuration_id"):
        hybrid["frozen_configuration_id"] = summary["winner"]
    if "fusion" not in hybrid:
        hybrid["fusion"] = next(
            run["fusion"]
            for run in hybrid["seed_runs"]
            if run["seed"] == 42
        )
    return hybrid, summary


def main() -> None:
    audit = read_json(resolve_artifact("data_audit.json"))
    baseline = read_json(resolve_artifact("baseline_metrics.json"))
    lstm = read_json(resolve_artifact("lstm_metrics.json"))
    hybrid, summary = load_hybrid_bundle()
    config.REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Learning curves use the LSTM backbone history; overlay hybrid freeze note.
    backbone_for_curves = dict(lstm)
    create_data_pipeline(audit, config.REPORT_FIGURES_DIR / "data_pipeline.png")
    create_learning_curves(
        backbone_for_curves,
        config.REPORT_FIGURES_DIR / "learning_curves.png",
        title=(
            "LSTM backbone training (weight_7 seed 42); "
            "hybrid freezes late fusion after this checkpoint"
        ),
    )
    create_confusion_matrices(
        baseline,
        hybrid,
        config.REPORT_FIGURES_DIR / "confusion_matrices.png",
        primary_title=f"{hybrid['frozen_configuration_id']} (seed 42)",
    )
    create_architecture_comparison(
        baseline,
        hybrid,
        config.REPORT_FIGURES_DIR / "architecture_comparison.png",
    )
    hybrid_predictions = resolve_artifact(
        "experiments",
        f"{hybrid['frozen_configuration_id']}_seed42_test_predictions.csv",
    )
    baseline_predictions = resolve_artifact("baseline_predictions.csv")
    build_qualitative_examples(
        hybrid_predictions,
        baseline_predictions,
        config.RESULTS_DIR / "qualitative_examples.csv",
    )
    write_generated_metrics(audit, baseline, lstm, hybrid, summary)
    print(f"Wrote report assets to {config.REPORT_FIGURES_DIR}")
    print(
        f"Primary model: {hybrid['frozen_configuration_id']} "
        f"test F1 {hybrid['test_aggregate']['f1']['mean']:.3f} "
        f"vs baseline {baseline['test']['f1']:.3f}"
    )


if __name__ == "__main__":
    main()
