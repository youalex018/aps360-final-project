"""TF-IDF + Linear SVM baseline with saved evaluation evidence.

Uses the same splits and the same `clean_text` tokenizer as the LSTM so the
reported accuracy is an apples-to-apples reference point, isolating the value
the LSTM adds over a classic bag-of-words model.
"""
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


def main():
    train_df, val_df, test_df = load_splits()

    # Pass our tokenizer directly so slang expansion and ASCII stripping are
    # identical to the deep model's input.
    vectorizer = TfidfVectorizer(tokenizer=clean_text, token_pattern=None)
    x_train = vectorizer.fit_transform(train_df["text"])
    x_test = vectorizer.transform(test_df["text"])

    clf = LinearSVC(class_weight="balanced", random_state=config.SEED)
    clf.fit(x_train, train_df["label"])

    preds = clf.predict(x_test)
    margins = clf.decision_function(x_test)
    acc = accuracy_score(test_df["label"], preds)
    balanced_acc = balanced_accuracy_score(test_df["label"], preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_df["label"],
        preds,
        average="binary",
        zero_division=0,
    )
    metrics = {
        "dataset": str(config.DATA_PATH.relative_to(config.ROOT)),
        "model": "TF-IDF + LinearSVC",
        "class_weight": "balanced",
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "test": {
            "accuracy": float(acc),
            "balanced_accuracy": float(balanced_acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix": confusion_matrix(
                test_df["label"], preds, labels=[0, 1]
            ).tolist(),
        },
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "baseline_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    with (config.RESULTS_DIR / "baseline_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["text", "label", "prediction", "margin"],
        )
        writer.writeheader()
        texts = test_df["raw_text"] if "raw_text" in test_df else test_df["text"]
        for text, label, prediction, margin in zip(
            texts, test_df["label"], preds, margins
        ):
            writer.writerow(
                {
                    "text": text,
                    "label": int(label),
                    "prediction": int(prediction),
                    "margin": f"{margin:.6f}",
                }
            )

    print(f"Baseline (TF-IDF + LinearSVC) test accuracy: {acc:.3f}\n")
    print(f"Balanced accuracy: {balanced_acc:.3f}; toxic-class F1: {f1:.3f}\n")
    print(classification_report(test_df["label"], preds, target_names=["safe", "toxic"]))


if __name__ == "__main__":
    main()
