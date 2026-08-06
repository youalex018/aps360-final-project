"""Tests for locked final-test metric calculations."""
from __future__ import annotations

import pytest

from evaluate_final_test import _match_dates, _metrics


def test_metrics_uses_toxic_as_positive_class():
    result = _metrics(
        labels=[0, 0, 0, 1, 1],
        predictions=[0, 1, 0, 1, 0],
        threshold=0.6,
    )

    assert result["confusion_matrix"] == [[2, 1], [1, 1]]
    assert result["true_negatives"] == 2
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["true_positives"] == 1
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)


def test_match_dates_extracts_unique_iso_dates():
    assert _match_dates(
        {
            "M-20260804-aaaa",
            "M-20260801-bbbb",
            "M-20260804-cccc",
            "invalid",
        }
    ) == ["2026-08-01", "2026-08-04"]
