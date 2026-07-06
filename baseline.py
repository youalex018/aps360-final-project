"""TF-IDF + Linear SVM baseline.

Uses the same splits and the same `clean_text` tokenizer as the LSTM so the
reported accuracy is an apples-to-apples reference point, isolating the value
the LSTM adds over a classic bag-of-words model.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import LinearSVC

from dataset import clean_text, load_splits


def main():
    train_df, val_df, test_df = load_splits()

    # Pass our tokenizer directly so slang expansion and ASCII stripping are
    # identical to the deep model's input.
    vectorizer = TfidfVectorizer(tokenizer=clean_text, token_pattern=None)
    x_train = vectorizer.fit_transform(train_df["text"])
    x_test = vectorizer.transform(test_df["text"])

    clf = LinearSVC()
    clf.fit(x_train, train_df["label"])

    preds = clf.predict(x_test)
    acc = accuracy_score(test_df["label"], preds)

    print(f"Baseline (TF-IDF + LinearSVC) test accuracy: {acc:.3f}\n")
    print(classification_report(test_df["label"], preds, target_names=["safe", "toxic"]))


if __name__ == "__main__":
    main()
