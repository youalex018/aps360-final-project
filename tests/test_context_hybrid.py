"""Tests for same-match context windows and validation-only late fusion."""
from __future__ import annotations

import csv

import numpy as np
import pandas as pd
import pytest

import dataset
from dataset import (
    CTX_TOKEN,
    MSG_TOKEN,
    TGT_TOKEN,
    add_context_column,
    build_context_tokens,
    get_dataloaders,
)
from hybrid import (
    fit_lexical_scorer,
    fuse,
    select_fusion_on_validation,
    svm_probabilities,
)
from train import save_predictions


def _context_csv(tmp_path):
    rows = [
        # group 1 train: three ordered messages
        {
            "text": "hello team",
            "raw_text": "hello team",
            "label": 0,
            "group_id": 1,
            "msg_index": 0,
            "split": "train",
        },
        {
            "text": "you suck",
            "raw_text": "you suck",
            "label": 1,
            "group_id": 1,
            "msg_index": 1,
            "split": "train",
        },
        {
            "text": "report mid",
            "raw_text": "report mid",
            "label": 1,
            "group_id": 1,
            "msg_index": 2,
            "split": "train",
        },
        {
            "text": "other match",
            "raw_text": "other match",
            "label": 0,
            "group_id": 2,
            "msg_index": 0,
            "split": "train",
        },
        {
            "text": "val safe",
            "raw_text": "val safe",
            "label": 0,
            "group_id": 3,
            "msg_index": 0,
            "split": "val",
        },
        {
            "text": "val toxic insult",
            "raw_text": "val toxic insult",
            "label": 1,
            "group_id": 4,
            "msg_index": 0,
            "split": "val",
        },
        {
            "text": "test safe",
            "raw_text": "test safe",
            "label": 0,
            "group_id": 5,
            "msg_index": 0,
            "split": "test",
        },
        {
            "text": "test toxic",
            "raw_text": "test toxic",
            "label": 1,
            "group_id": 6,
            "msg_index": 0,
            "split": "test",
        },
    ]
    path = tmp_path / "prepared.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_context_builder_stays_inside_group():
    tokens = build_context_tokens(
        ["alpha", "beta toxic", "gamma"],
        target_pos=1,
        k=1,
    )
    assert tokens[0] == CTX_TOKEN
    assert "alpha" in tokens
    assert TGT_TOKEN in tokens
    assert "beta" in tokens
    assert MSG_TOKEN in tokens
    assert "gamma" in tokens
    assert "foreign" not in tokens


def test_add_context_column_never_mixes_group_ids():
    frame = pd.DataFrame(
        [
            {"text": "a", "label": 0, "group_id": 10, "msg_index": 0},
            {"text": "b", "label": 1, "group_id": 10, "msg_index": 1},
            {"text": "SECRET", "label": 0, "group_id": 99, "msg_index": 0},
        ]
    )
    with_context = add_context_column(frame, context_k=2)
    group_ten = with_context[with_context["group_id"] == 10]
    for text in group_ten["context_text"]:
        assert "SECRET" not in text
        assert TGT_TOKEN in text.split()


def test_context_tuning_skips_test_loader(tmp_path, monkeypatch):
    path = _context_csv(tmp_path)
    constructed = []
    original = dataset.ToxicChatDataset

    def recording_dataset(frame, vocab, max_len=50, **kwargs):
        constructed.append(set(frame["split"].unique()))
        return original(frame, vocab, max_len, **kwargs)

    monkeypatch.setattr(dataset, "ToxicChatDataset", recording_dataset)
    _, _, test_loader, vocab = get_dataloaders(
        path,
        batch_size=2,
        seed=42,
        include_test=False,
        context_k=1,
        max_len=40,
    )
    assert test_loader is None
    assert constructed == [{"train"}, {"val"}]
    for token in (CTX_TOKEN, MSG_TOKEN, TGT_TOKEN):
        assert token in vocab.stoi


def test_fusion_fit_uses_train_only_and_persists_threshold(tmp_path):
    train_texts = ["nice play", "kill yourself noob", "gl hf", "trash jungler"]
    train_labels = [0, 1, 0, 1]
    val_texts = ["well played", "you are trash"]
    val_labels = [0, 1]
    test_texts = ["unused test toxic"]
    # Fit must not see test texts.
    clf, vectorizer = fit_lexical_scorer(train_texts, train_labels)
    assert "unused" not in vectorizer.vocabulary_

    p_svm = svm_probabilities(clf, vectorizer, val_texts)
    p_lstm = np.asarray([0.20, 0.80], dtype=np.float64)
    selected = select_fusion_on_validation(
        val_labels,
        p_lstm,
        p_svm,
        alpha_grid=(0.0, 0.5, 1.0),
        threshold_grid=(0.25, 0.50, 0.75),
    )
    blended = fuse(p_lstm, p_svm, selected["alpha"])
    output = tmp_path / "hybrid_preds.csv"
    save_predictions(
        output,
        val_texts,
        val_labels,
        blended,
        selected["threshold"],
    )
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["threshold"] for row in rows} == {f"{selected['threshold']:.2f}"}
    assert all(row["prediction"] in {"0", "1"} for row in rows)
    # Sanity: test texts were never required for selection.
    assert test_texts[0] not in {row["text"] for row in rows}


def test_prepare_msg_index_orders_within_group():
    from prepare_l2dtnh import _assign_msg_index

    raw = pd.DataFrame(
        [
            {
                "message": "second",
                "label": 0,
                "chatlog_id": 7,
                "time": "00:00:20",
                "id": 2,
            },
            {
                "message": "first",
                "label": 0,
                "chatlog_id": 7,
                "time": "00:00:10",
                "id": 1,
            },
            {
                "message": "other",
                "label": 1,
                "chatlog_id": 8,
                "time": "00:00:01",
                "id": 3,
            },
        ]
    )
    ordered = _assign_msg_index(raw)
    group7 = ordered[ordered["chatlog_id"] == 7].sort_values("msg_index")
    assert group7["message"].tolist() == ["first", "second"]
    assert group7["msg_index"].tolist() == [0, 1]
