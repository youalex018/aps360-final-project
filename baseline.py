"""TF-IDF + Linear SVM baseline with saved evaluation evidence.

Uses the same splits and the same `clean_text` tokenizer as the LSTM so the
reported accuracy is an apples-to-apples reference point, isolating the value
the LSTM adds over a classic bag-of-words model.
"""
import argparse
import csv
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.svm import LinearSVC

import config
from dataset import clean_text, load_splits


def _metric_summary(labels, predictions) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
    }


def run_baseline(*, evaluate_test: bool = False) -> dict:
    train_df, val_df, test_df = load_splits()

    # Pass our tokenizer directly so slang expansion and ASCII stripping are
    # identical to the deep model's input.
    vectorizer = TfidfVectorizer(tokenizer=clean_text, token_pattern=None)
    x_train = vectorizer.fit_transform(train_df["text"])
    x_val = vectorizer.transform(val_df["text"])

    clf = LinearSVC(class_weight="balanced", random_state=config.SEED)
    clf.fit(x_train, train_df["label"])

    val_predictions = clf.predict(x_val)
    metrics = {
        "schema_version": 2,
        "dataset": str(config.DATA_PATH.relative_to(config.ROOT)),
        "model": "TF-IDF + LinearSVC",
        "class_weight": "balanced",
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "validation": _metric_summary(val_df["label"], val_predictions),
        "test_accessed": bool(evaluate_test),
    }
    test_predictions = None
    test_margins = None
    if evaluate_test:
        x_test = vectorizer.transform(test_df["text"])
        test_predictions = clf.predict(x_test)
        test_margins = clf.decision_function(x_test)
        metrics["test"] = _metric_summary(test_df["label"], test_predictions)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "baseline_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    if evaluate_test:
        with (config.RESULTS_DIR / "baseline_predictions.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["text", "label", "prediction", "margin"],
            )
            writer.writeheader()
            texts = (
                test_df["raw_text"] if "raw_text" in test_df else test_df["text"]
            )
            for text, label, prediction, margin in zip(
                texts,
                test_df["label"],
                test_predictions,
                test_margins,
            ):
                writer.writerow(
                    {
                        "text": text,
                        "label": int(label),
                        "prediction": int(prediction),
                        "margin": f"{margin:.6f}",
                    }
                )

    val = metrics["validation"]
    print(
        "Baseline validation: "
        f"balanced accuracy {val['balanced_accuracy']:.3f}; "
        f"toxic-class F1 {val['f1']:.3f}"
    )
    if evaluate_test:
        test = metrics["test"]
        print(
            "Frozen baseline test: "
            f"balanced accuracy {test['balanced_accuracy']:.3f}; "
            f"toxic-class F1 {test['f1']:.3f}\n"
        )
        print(
            classification_report(
                test_df["label"],
                test_predictions,
                target_names=["safe", "toxic"],
            )
        )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Run only after all LSTM choices are frozen.",
    )
    args = parser.parse_args()
    run_baseline(evaluate_test=args.evaluate_test)


if __name__ == "__main__":
    main()
