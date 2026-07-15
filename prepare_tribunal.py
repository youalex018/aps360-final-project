"""Prepare a balanced Tribunal chat dataset for LSTM/TF-IDF training."""
import pandas as pd

import config


SOURCE_TEXT_COL = "message"
SOURCE_LABEL_COL = "association_to_offender"
OUTPUT_COLUMNS = ["text", "label"]


def load_relevant_columns() -> pd.DataFrame:
    """Read only columns used for supervision to avoid loading case metadata."""
    return pd.read_csv(
        config.TRIBUNAL_RAW_PATH,
        usecols=[SOURCE_TEXT_COL, SOURCE_LABEL_COL],
    )


def make_text_label_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.rename(columns={SOURCE_TEXT_COL: "text"}).copy()
    df = df[df["text"].notna() & df[SOURCE_LABEL_COL].notna()]
    df["text"] = df["text"].astype(str).str.strip()
    # Keep only rows that can produce at least one token under clean_text().
    df = df[df["text"].str.contains(r"[A-Za-z0-9]", regex=True)]
    df["label"] = (df[SOURCE_LABEL_COL] == "offender").astype(int)
    return df[OUTPUT_COLUMNS]


def balanced_subsample(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    per_class = sample_size // 2
    counts = df["label"].value_counts()
    if counts.min() < per_class:
        raise ValueError(
            f"Need at least {per_class} rows per class, got {counts.to_dict()}."
        )

    parts = [
        df[df["label"] == label].sample(n=per_class, random_state=config.SEED)
        for label in (0, 1)
    ]
    return (
        pd.concat(parts)
        .sample(frac=1, random_state=config.SEED)
        .reset_index(drop=True)
    )


def main() -> None:
    raw_df = load_relevant_columns()
    labeled_df = make_text_label_frame(raw_df)
    sample_df = balanced_subsample(labeled_df, config.SUBSAMPLE_SIZE)

    config.TRIBUNAL_PREPARED_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(config.TRIBUNAL_PREPARED_PATH, index=False)

    print(f"Wrote: {config.TRIBUNAL_PREPARED_PATH}")
    print(f"Rows: {len(sample_df):,}")
    print("Class counts:")
    print(sample_df["label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
