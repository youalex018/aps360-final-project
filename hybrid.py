"""Late fusion of TF-IDF LinearSVC scores with LSTM probabilities.

The lexical scorer is always fit on train text only. Blend weight ``alpha`` and
the decision threshold are chosen on validation toxic-class F1; test is scored
only with already-frozen fusion parameters.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

import config
from dataset import add_context_column, clean_text, load_splits
from train import (
    collect_split_probabilities,
    experiment_config_from_dict,
    metric_summary,
    save_predictions,
)


def fuse(p_lstm, p_svm, alpha: float) -> np.ndarray:
    """Linear blend: ``alpha * p_lstm + (1 - alpha) * p_svm``."""
    p_lstm = np.asarray(p_lstm, dtype=np.float64)
    p_svm = np.asarray(p_svm, dtype=np.float64)
    return alpha * p_lstm + (1.0 - alpha) * p_svm


def fit_lexical_scorer(train_texts, train_labels):
    """Fit the same TF-IDF + LinearSVC recipe as ``baseline.py`` on train only."""
    vectorizer = TfidfVectorizer(tokenizer=clean_text, token_pattern=None)
    x_train = vectorizer.fit_transform(train_texts)
    clf = LinearSVC(class_weight="balanced", random_state=config.SEED)
    clf.fit(x_train, train_labels)
    return clf, vectorizer


def svm_probabilities(clf, vectorizer, texts) -> np.ndarray:
    """Map SVM decision margins to (0, 1) via a logistic squashing."""
    margins = clf.decision_function(vectorizer.transform(texts))
    return 1.0 / (1.0 + np.exp(-np.asarray(margins, dtype=np.float64)))


def select_fusion_on_validation(
    y_val,
    p_lstm,
    p_svm,
    *,
    alpha_grid=config.ALPHA_GRID,
    threshold_grid=config.THRESHOLD_GRID,
) -> dict:
    """Jointly choose alpha and threshold by validation toxic-class F1."""
    y_val = np.asarray(y_val, dtype=np.int64)
    best: dict | None = None
    for alpha in alpha_grid:
        blended = fuse(p_lstm, p_svm, float(alpha))
        for threshold in threshold_grid:
            metrics = metric_summary(y_val, blended, loss=0.0, threshold=threshold)
            metrics.pop("loss", None)
            candidate = {
                "alpha": float(alpha),
                "threshold": float(threshold),
                "metrics": metrics,
            }
            if best is None:
                best = candidate
                continue
            key = (
                metrics["f1"],
                metrics["precision"],
                metrics["balanced_accuracy"],
                -abs(threshold - 0.5),
                -threshold,
                -abs(alpha - 0.5),
            )
            incumbent = (
                best["metrics"]["f1"],
                best["metrics"]["precision"],
                best["metrics"]["balanced_accuracy"],
                -abs(best["threshold"] - 0.5),
                -best["threshold"],
                -abs(best["alpha"] - 0.5),
            )
            if key > incumbent:
                best = candidate
    assert best is not None
    return best


def late_fuse_lstm_run(
    lstm_result: dict,
    *,
    hybrid_config_id: str,
    device_name: str | None = None,
    evaluate_test: bool = False,
    metrics_path: Path | None = None,
) -> dict:
    """Attach validation-tuned SVM late fusion onto an existing LSTM run JSON."""
    train_df, val_df, test_df = load_splits()
    checkpoint_path = config.ROOT / lstm_result["checkpoint_path"]
    p_lstm_val, y_val, spec = collect_split_probabilities(
        checkpoint_path,
        split="val",
        device_name=device_name,
    )
    clf, vectorizer = fit_lexical_scorer(train_df["text"], train_df["label"])

    if spec.context_k is not None:
        ordered_val = add_context_column(val_df, spec.context_k)
        p_svm_val = svm_probabilities(clf, vectorizer, ordered_val["text"])
        y_for_fusion = ordered_val["label"].to_numpy(dtype=np.int64)
    else:
        p_svm_val = svm_probabilities(clf, vectorizer, val_df["text"])
        y_for_fusion = val_df["label"].to_numpy(dtype=np.int64)

    if len(p_lstm_val) != len(p_svm_val):
        raise ValueError(
            f"LSTM/SVM probability length mismatch: "
            f"{len(p_lstm_val)} vs {len(p_svm_val)}"
        )
    if list(y_for_fusion) != list(y_val):
        raise ValueError(
            "Validation label order mismatch between LSTM and SVM paths."
        )

    selected = select_fusion_on_validation(y_for_fusion, p_lstm_val, p_svm_val)
    hybrid_spec = replace(
        experiment_config_from_dict(lstm_result["experiment_config"]),
        config_id=hybrid_config_id,
        hybrid=True,
        seed=int(lstm_result["seed"]),
    )
    fused_val = fuse(p_lstm_val, p_svm_val, selected["alpha"])
    validation = metric_summary(
        y_for_fusion,
        fused_val,
        loss=0.0,
        threshold=selected["threshold"],
    )
    validation.pop("loss", None)

    result = {
        "schema_version": 2,
        "dataset": lstm_result["dataset"],
        "run_name": f"{hybrid_config_id}_seed{lstm_result['seed']}",
        "configuration_id": hybrid_config_id,
        "seed": int(lstm_result["seed"]),
        "checkpoint_path": lstm_result["checkpoint_path"],
        "base_configuration_id": lstm_result["configuration_id"],
        "base_run_name": lstm_result["run_name"],
        "selection_metric": "validation toxic-class F1 (late fusion)",
        "threshold_grid": list(config.THRESHOLD_GRID),
        "alpha_grid": list(config.ALPHA_GRID),
        "best_epoch": lstm_result.get("best_epoch"),
        "selected_threshold": selected["threshold"],
        "positive_weight": lstm_result.get("positive_weight"),
        "vocab_size": lstm_result.get("vocab_size"),
        "parameter_count": lstm_result.get("parameter_count"),
        "experiment_config": asdict(hybrid_spec),
        "environment": lstm_result.get("environment"),
        "history": lstm_result.get("history"),
        "validation": validation,
        "lstm_only_validation": lstm_result["validation"],
        "fusion": {
            "method": "late_linear_blend",
            "alpha": selected["alpha"],
            "threshold": selected["threshold"],
            "svm_fit_split": "train",
            "selection_split": "val",
            "test_used_for_selection": False,
        },
        "test_accessed": bool(evaluate_test),
    }

    if evaluate_test:
        p_lstm_test, y_test, _ = collect_split_probabilities(
            checkpoint_path,
            split="test",
            device_name=device_name,
        )
        if spec.context_k is not None:
            ordered_test = add_context_column(test_df, spec.context_k)
            p_svm_test = svm_probabilities(clf, vectorizer, ordered_test["text"])
            texts = (
                ordered_test["raw_text"]
                if "raw_text" in ordered_test
                else ordered_test["text"]
            )
            y_test_aligned = ordered_test["label"].to_numpy(dtype=np.int64)
        else:
            p_svm_test = svm_probabilities(clf, vectorizer, test_df["text"])
            texts = (
                test_df["raw_text"] if "raw_text" in test_df else test_df["text"]
            )
            y_test_aligned = test_df["label"].to_numpy(dtype=np.int64)
        if list(y_test_aligned) != list(y_test):
            raise ValueError(
                "Test label order mismatch between LSTM and SVM paths."
            )
        fused_test = fuse(p_lstm_test, p_svm_test, selected["alpha"])
        test_metrics = metric_summary(
            y_test_aligned,
            fused_test,
            loss=0.0,
            threshold=selected["threshold"],
        )
        test_metrics.pop("loss", None)
        result["test"] = test_metrics
        prediction_path = (
            config.EXPERIMENTS_DIR
            / f"{hybrid_config_id}_seed{lstm_result['seed']}_test_predictions.csv"
        )
        save_predictions(
            prediction_path,
            texts,
            y_test_aligned,
            fused_test,
            selected["threshold"],
        )

    if metrics_path is None:
        metrics_path = (
            config.EXPERIMENTS_DIR
            / f"{hybrid_config_id}_seed{lstm_result['seed']}.json"
        )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
