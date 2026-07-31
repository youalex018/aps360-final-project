"""Focused regression tests for leakage-safe model selection."""
from __future__ import annotations

import csv

import pandas as pd
import pytest

import dataset
from dataset import get_dataloaders, load_splits
from train import (
    checkpoint_is_better,
    metric_summary,
    save_predictions,
    select_validation_threshold,
)


def _prepared_csv(tmp_path, *, leak: bool = False):
    rows = [
        {"text": "safe one", "label": 0, "group_id": 1, "split": "train"},
        {"text": "toxic one", "label": 1, "group_id": 2, "split": "train"},
        {"text": "safe two", "label": 0, "group_id": 3, "split": "train"},
        {"text": "toxic two", "label": 1, "group_id": 4, "split": "train"},
        {"text": "validation safe", "label": 0, "group_id": 5, "split": "val"},
        {"text": "validation toxic", "label": 1, "group_id": 6, "split": "val"},
        {
            "text": "test only",
            "label": 0,
            "group_id": 5 if leak else 7,
            "split": "test",
        },
    ]
    path = tmp_path / "prepared.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_f1_selects_checkpoint_and_loss_only_breaks_ties():
    incumbent = {"f1": 0.70, "loss": 0.40}
    assert checkpoint_is_better({"f1": 0.71, "loss": 0.90}, incumbent)
    assert checkpoint_is_better({"f1": 0.70, "loss": 0.39}, incumbent)
    assert not checkpoint_is_better({"f1": 0.69, "loss": 0.10}, incumbent)
    assert not checkpoint_is_better({"f1": 0.70, "loss": 0.41}, incumbent)


def test_selected_threshold_is_persisted_and_applied(tmp_path):
    selected = select_validation_threshold(
        labels=[0, 0, 1, 1],
        probabilities=[0.10, 0.30, 0.35, 0.90],
        loss=0.5,
        threshold_grid=(0.25, 0.50),
    )
    assert selected["threshold"] == 0.25
    assert selected["f1"] > metric_summary(
        [0, 0, 1, 1], [0.10, 0.30, 0.35, 0.90], 0.5, 0.50
    )["f1"]

    output = tmp_path / "predictions.csv"
    save_predictions(
        output,
        ["a", "b"],
        [0, 1],
        [0.24, 0.25],
        selected["threshold"],
    )
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["prediction"] for row in rows] == ["0", "1"]
    assert {row["threshold"] for row in rows} == {"0.25"}


def test_seeded_training_order_is_reproducible(tmp_path):
    path = _prepared_csv(tmp_path)
    first, _, _, _ = get_dataloaders(path, batch_size=2, seed=43)
    second, _, _, _ = get_dataloaders(path, batch_size=2, seed=43)
    first_order = [labels.tolist() for _, _, labels in first]
    second_order = [labels.tolist() for _, _, labels in second]
    assert first_order == second_order


def test_group_isolation_is_enforced(tmp_path):
    with pytest.raises(ValueError, match="group_id"):
        load_splits(_prepared_csv(tmp_path, leak=True))


def test_tuning_does_not_construct_test_dataset(tmp_path, monkeypatch):
    path = _prepared_csv(tmp_path)
    constructed_splits = []
    original = dataset.ToxicChatDataset

    def recording_dataset(frame, vocab, max_len=50, **kwargs):
        constructed_splits.append(set(frame["split"]))
        return original(frame, vocab, max_len, **kwargs)

    monkeypatch.setattr(dataset, "ToxicChatDataset", recording_dataset)
    _, _, test_loader, _ = get_dataloaders(
        path,
        batch_size=2,
        seed=42,
        include_test=False,
    )
    assert test_loader is None
    assert constructed_splits == [{"train"}, {"val"}]
