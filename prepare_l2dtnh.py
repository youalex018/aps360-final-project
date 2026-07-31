"""Prepare the expert-annotated L2DTnH corpus with leakage-safe splits.

Source:
https://github.com/irdin-pekaric/esorics26_toxicity

The original Tribunal export does not contain message-level toxicity labels or
case verdicts. L2DTnH adds expert message-level labels to Tribunal chat and
retains ``chatlog_id``, which lets us keep each match in exactly one split.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import config
from dataset import clean_text

SOURCE_COLUMNS = ["message", "label", "chatlog_id"]
ORDER_COLUMNS = ["time", "id"]
OUTPUT_COLUMNS = ["raw_text", "text", "label", "group_id", "msg_index", "split"]


def _assign_msg_index(df: pd.DataFrame) -> pd.DataFrame:
    """Order messages within each match and assign a stable ``msg_index``.

    Prefer raw ``time`` then ``id`` when present; otherwise keep encounter order.
    """
    ordered = df.copy()
    sort_cols = ["chatlog_id"]
    if "time" in ordered.columns:
        sort_cols.append("time")
    if "id" in ordered.columns:
        sort_cols.append("id")
    ordered = ordered.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    ordered["msg_index"] = ordered.groupby("chatlog_id", sort=False).cumcount()
    return ordered


def _label_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = df["label"].value_counts().sort_index()
    return {str(int(label)): int(count) for label, count in counts.items()}


def _best_group_holdout(
    df: pd.DataFrame,
    holdout_fraction: float,
    seed: int,
    candidates: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose a group-disjoint split close to target size and class balance."""
    splitter = GroupShuffleSplit(
        n_splits=candidates,
        test_size=holdout_fraction,
        random_state=seed,
    )
    target_rate = float(df["label"].mean())
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    dummy = np.zeros(len(df), dtype=np.uint8)

    for keep_idx, holdout_idx in splitter.split(dummy, df["label"], df["group_id"]):
        holdout = df.iloc[holdout_idx]
        size_error = abs(len(holdout) / len(df) - holdout_fraction)
        rate_error = abs(float(holdout["label"].mean()) - target_rate)
        positive_share = holdout["label"].sum() / max(1, df["label"].sum())
        positive_error = abs(float(positive_share) - holdout_fraction)
        score = size_error + 2.0 * rate_error + positive_error
        if best is None or score < best[0]:
            best = (score, keep_idx, holdout_idx)

    if best is None:
        raise RuntimeError("Could not construct a grouped holdout split.")
    return best[1], best[2]


def assign_grouped_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Assign approximately 70/15/15 train/val/test splits by chat log."""
    train_idx, temp_idx = _best_group_holdout(
        df, holdout_fraction=1.0 - config.TRAIN_FRAC, seed=config.SEED
    )
    train = df.iloc[train_idx].copy()
    temp = df.iloc[temp_idx].copy()
    val_idx, test_idx = _best_group_holdout(
        temp,
        holdout_fraction=0.5,
        seed=config.SEED + 1,
    )
    val = temp.iloc[val_idx].copy()
    test = temp.iloc[test_idx].copy()
    train["split"], val["split"], test["split"] = "train", "val", "test"
    prepared = pd.concat([train, val, test], ignore_index=True)

    groups = {
        name: set(part["group_id"])
        for name, part in prepared.groupby("split", sort=False)
    }
    if groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"]:
        raise AssertionError("Grouped split leaked a chatlog_id between partitions.")
    return prepared


def prepare_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    missing_columns = set(SOURCE_COLUMNS) - set(raw.columns)
    if missing_columns:
        raise ValueError(f"L2DTnH source is missing columns: {sorted(missing_columns)}")

    audit: dict[str, object] = {
        "source_rows": int(len(raw)),
        "source_label_counts": _label_counts(raw),
        "source_groups": int(raw["chatlog_id"].nunique()),
        "missing_message_rows": int(raw["message"].isna().sum()),
        "missing_group_rows": int(raw["chatlog_id"].isna().sum()),
        "missing_required_rows": int(
            raw[["message", "label", "chatlog_id"]].isna().any(axis=1).sum()
        ),
    }

    df = raw.dropna(subset=["message", "label", "chatlog_id"]).copy()
    df["raw_text"] = df["message"].astype(str)
    df["text"] = df["raw_text"].map(lambda value: " ".join(clean_text(value)))
    empty_mask = df["text"].eq("")
    audit["empty_after_normalization_rows"] = int(empty_mask.sum())
    df = df[~empty_mask].copy()

    label_cardinality = df.groupby("text")["label"].nunique()
    conflicting_texts = set(label_cardinality[label_cardinality > 1].index)
    conflict_mask = df["text"].isin(conflicting_texts)
    audit["conflicting_normalized_texts"] = int(len(conflicting_texts))
    audit["conflicting_rows_removed"] = int(conflict_mask.sum())
    audit["same_label_duplicate_rows_retained"] = int(
        df[~conflict_mask].duplicated(["text", "label"]).sum()
    )
    df = df[~conflict_mask].copy()

    df["label"] = df["label"].astype(int)
    df = _assign_msg_index(df)
    df["group_id"] = df["chatlog_id"].astype(int)
    audit["ordering_columns_used"] = [
        column for column in ORDER_COLUMNS if column in df.columns
    ]
    prepared = assign_grouped_splits(df)
    prepared = prepared[OUTPUT_COLUMNS].sort_values(
        ["split", "group_id", "msg_index"], kind="stable"
    )

    token_lengths = prepared["text"].str.split().str.len()
    audit.update(
        {
            "prepared_rows": int(len(prepared)),
            "prepared_label_counts": _label_counts(prepared),
            "prepared_positive_rate": float(prepared["label"].mean()),
            "prepared_groups": int(prepared["group_id"].nunique()),
            "token_length": {
                "mean": float(token_lengths.mean()),
                "median": float(token_lengths.median()),
                "p95": float(token_lengths.quantile(0.95)),
                "max": int(token_lengths.max()),
            },
            "splits": {
                name: {
                    "rows": int(len(part)),
                    "groups": int(part["group_id"].nunique()),
                    "label_counts": _label_counts(part),
                    "positive_rate": float(part["label"].mean()),
                }
                for name, part in prepared.groupby("split", sort=False)
            },
        }
    )

    changed = prepared[prepared["raw_text"].str.lower().str.strip() != prepared["text"]]
    audit["cleaning_examples"] = [
        {
            "raw": row.raw_text,
            "cleaned": row.text,
            "label": int(row.label),
        }
        for row in changed.head(5).itertuples()
    ]
    return prepared, audit


def audit_legacy_proxy(path: Path) -> dict[str, object] | None:
    """Summarize the first-iteration role-proxy file when it is available."""
    if not path.exists():
        return None
    legacy = pd.read_csv(path)
    legacy["normalized"] = legacy["text"].map(lambda value: " ".join(clean_text(value)))
    conflicts = legacy.groupby("normalized")["label"].nunique()
    conflict_keys = set(conflicts[conflicts > 1].index)
    lengths = legacy["normalized"].str.split().str.len()
    return {
        "rows": int(len(legacy)),
        "label_counts": _label_counts(legacy),
        "conflicting_normalized_texts": int(len(conflict_keys)),
        "rows_with_conflicting_labels": int(legacy["normalized"].isin(conflict_keys).sum()),
        "median_tokens": float(lengths.median()),
    }


def main() -> None:
    raw = pd.read_csv(config.L2DTNH_RAW_PATH, sep=";")
    prepared, audit = prepare_frame(raw)
    audit["legacy_role_proxy"] = audit_legacy_proxy(config.TRIBUNAL_PREPARED_PATH)

    config.L2DTNH_PREPARED_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(config.L2DTNH_PREPARED_PATH, index=False)
    audit_path = config.RESULTS_DIR / "data_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(f"Wrote: {config.L2DTNH_PREPARED_PATH}")
    print(f"Audit: {audit_path}")
    print(f"Rows: {len(prepared):,}")
    print(f"Groups: {prepared['group_id'].nunique():,}")
    print("Class counts:")
    print(prepared["label"].value_counts().sort_index().to_string())
    print("Split summary:")
    print(
        prepared.groupby("split")
        .agg(rows=("label", "size"), positives=("label", "sum"), groups=("group_id", "nunique"))
        .to_string()
    )


if __name__ == "__main__":
    main()
