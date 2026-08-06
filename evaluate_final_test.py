"""Evaluate the locked fresh-test prediction CSV without retuning.

This script does not run inference or search thresholds. It reads the one-shot
hybrid output, applies the already-frozen branch thresholds, and writes:

- ``artifacts/final_test/final_test_metrics.json``
- ``artifacts/final_test/final_test_error_analysis.csv``
- ``reports/final_report/final_test_confusion_matrices.png``
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import config

DEFAULT_PREDICTIONS = (
    config.FINAL_TEST_ARTIFACTS_DIR / "final_hybrid_predictions.csv"
)
DEFAULT_METRICS = config.FINAL_TEST_ARTIFACTS_DIR / "final_test_metrics.json"
DEFAULT_ERRORS = (
    config.FINAL_TEST_ARTIFACTS_DIR / "final_test_error_analysis.csv"
)
DEFAULT_FIGURE = (
    config.ROOT / "reports" / "final_report" / "fresh_confusion.png"
)
LSTM_RUN = config.EXPERIMENTS_DIR / "weight_7_seed42.json"
HYBRID_RUN = config.EXPERIMENTS_DIR / "weight7_hybrid_late_seed42.json"
SVM_THRESHOLD = 0.5  # sigmoid(LinearSVC margin) >= 0.5 iff margin >= 0
REQUIRED_COLUMNS = {
    "match_id",
    "message_order",
    "text",
    "human_label",
    "prediction",
    "probability",
    "p_lstm",
    "p_svm",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} contains no predictions")
    invalid = [
        index
        for index, row in enumerate(rows, start=2)
        if row["human_label"].strip() not in {"0", "1"}
    ]
    if invalid:
        raise ValueError(f"human_label must be locked to 0/1; bad rows: {invalid}")
    return rows


def _metrics(labels: list[int], predictions: list[int], threshold: float) -> dict:
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "predicted_toxic": tp + fp,
    }


def _match_dates(match_ids: set[str]) -> list[str]:
    dates = []
    for match_id in match_ids:
        match = re.match(r"^M-(\d{4})(\d{2})(\d{2})-", match_id)
        if match:
            dates.append("-".join(match.groups()))
    return sorted(set(dates))


def _write_errors(
    path: Path,
    rows: list[dict[str, str]],
    labels: list[int],
    predictions: list[int],
) -> None:
    errors = []
    for row, target, prediction in zip(rows, labels, predictions):
        if target == prediction:
            continue
        errors.append(
            {
                "error_type": "false_positive" if prediction else "false_negative",
                "match_id": row["match_id"],
                "message_order": row["message_order"],
                "text": row["text"],
                "human_label": target,
                "hybrid_prediction": prediction,
                "hybrid_probability": float(row["probability"]),
                "lstm_probability": float(row["p_lstm"]),
                "svm_probability": float(row["p_svm"]),
                "notes": row.get("notes", ""),
            }
        )
    errors.sort(
        key=lambda item: (
            item["error_type"],
            -item["hybrid_probability"],
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(errors[0]))
        writer.writeheader()
        writer.writerows(errors)


def _write_figure(path: Path, model_metrics: dict[str, dict]) -> None:
    """Write an RGB PNG that pdfLaTeX/Overleaf can include reliably."""
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    titles = {
        "svm": "TF-IDF + LinearSVC",
        "lstm": "Single-message LSTM",
        "hybrid": "Late-fusion hybrid",
    }
    for axis, model in zip(axes, ("svm", "lstm", "hybrid")):
        matrix = np.asarray(model_metrics[model]["confusion_matrix"])
        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=["Non-toxic", "Toxic"],
        )
        display.plot(
            ax=axis,
            cmap="Blues",
            colorbar=False,
            values_format="d",
            text_kw={"fontsize": 11, "color": "black"},
        )
        # Force readable counts on both light and dark cells.
        for text_obj, value in zip(
            display.text_.ravel(),
            matrix.ravel(),
        ):
            text_obj.set_color("white" if value >= matrix.max() / 2 else "black")
            text_obj.set_fontsize(11)
        axis.set_title(titles[model], fontsize=10)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label" if model == "svm" else "")
    fig.suptitle(
        "Locked fresh-test confusion matrices (332 messages)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    # RGB (no alpha): RGBA PNGs often fail or look wrong under pdfLaTeX.
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    try:
        from PIL import Image

        image = Image.open(path).convert("RGB")
        image.save(path, format="PNG", optimize=True)
    except ImportError:
        pass


def evaluate(
    predictions_path: Path,
    metrics_path: Path,
    errors_path: Path,
    figure_path: Path,
) -> dict:
    rows = _load_rows(predictions_path)
    lstm_run = _load_json(LSTM_RUN)
    hybrid_run = _load_json(HYBRID_RUN)
    lstm_threshold = float(lstm_run["selected_threshold"])
    hybrid_threshold = float(hybrid_run["fusion"]["threshold"])
    hybrid_alpha = float(hybrid_run["fusion"]["alpha"])

    labels = [int(row["human_label"]) for row in rows]
    model_predictions = {
        "svm": [int(float(row["p_svm"]) >= SVM_THRESHOLD) for row in rows],
        "lstm": [
            int(float(row["p_lstm"]) >= lstm_threshold) for row in rows
        ],
        "hybrid": [int(row["prediction"]) for row in rows],
    }
    recomputed_hybrid = [
        int(float(row["probability"]) >= hybrid_threshold) for row in rows
    ]
    if recomputed_hybrid != model_predictions["hybrid"]:
        raise ValueError(
            "Stored hybrid predictions do not match the frozen threshold"
        )

    thresholds = {
        "svm": SVM_THRESHOLD,
        "lstm": lstm_threshold,
        "hybrid": hybrid_threshold,
    }
    model_metrics = {
        model: _metrics(labels, values, thresholds[model])
        for model, values in model_predictions.items()
    }
    match_ids = {row["match_id"] for row in rows}
    toxic_count = sum(labels)
    all_safe_accuracy = (len(labels) - toxic_count) / len(labels)
    development_hybrid = hybrid_run["test"]

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_predictions": str(predictions_path.relative_to(config.ROOT)),
        "protocol": {
            "labels_locked": True,
            "inference_run": "weight7_hybrid_late_seed42",
            "checkpoint": hybrid_run["checkpoint_path"],
            "hybrid_alpha": hybrid_alpha,
            "thresholds_selected_on": "L2DTnH validation split",
            "test_threshold_search": False,
        },
        "dataset": {
            "messages": len(rows),
            "matches": len(match_ids),
            "match_dates": _match_dates(match_ids),
            "non_toxic": len(labels) - toxic_count,
            "toxic": toxic_count,
            "toxic_prevalence": toxic_count / len(labels),
        },
        "models": model_metrics,
        "comparisons": {
            "all_non_toxic_accuracy": all_safe_accuracy,
            "hybrid_accuracy_minus_all_non_toxic": (
                model_metrics["hybrid"]["accuracy"] - all_safe_accuracy
            ),
            "svm_f1_minus_hybrid_f1": (
                model_metrics["svm"]["f1"] - model_metrics["hybrid"]["f1"]
            ),
            "hybrid_fresh_minus_development": {
                metric: model_metrics["hybrid"][metric]
                - float(development_hybrid[metric])
                for metric in (
                    "accuracy",
                    "balanced_accuracy",
                    "precision",
                    "recall",
                    "f1",
                )
            },
        },
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_errors(
        errors_path,
        rows,
        labels,
        model_predictions["hybrid"],
    )
    _write_figure(figure_path, model_metrics)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate locked final-test prediction probabilities"
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--errors", type=Path, default=DEFAULT_ERRORS)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = evaluate(
        args.predictions,
        args.metrics,
        args.errors,
        args.figure,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
