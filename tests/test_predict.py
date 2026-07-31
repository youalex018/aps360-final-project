"""Tests for the interactive hybrid predictor."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import config
from dataset import Vocab, PAD_TOKEN, UNK_TOKEN
from hybrid import fit_lexical_scorer
from predict import (
    DEFAULT_CHECKPOINT,
    DEFAULT_HYBRID_JSON,
    Predictor,
    format_score,
    load_fusion_params,
    resolve_checkpoint,
    score_message,
)


class _ConstantLogitModel(nn.Module):
    """Tiny stand-in that ignores inputs and returns a fixed logit."""

    def __init__(self, logit: float):
        super().__init__()
        self.logit = float(logit)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return torch.full((x.size(0),), self.logit, dtype=torch.float32)


def _tiny_predictor(*, logit: float = 0.0, alpha: float = 0.3, threshold: float = 0.6) -> Predictor:
    texts = ["gg wp", "nice play", "kys trash", "you suck idiot"]
    labels = [0, 0, 1, 1]
    clf, vectorizer = fit_lexical_scorer(texts, labels)
    vocab = Vocab({PAD_TOKEN: 0, UNK_TOKEN: 1, "gg": 2, "wp": 3, "trash": 4})
    return Predictor(
        model=_ConstantLogitModel(logit),
        vocab=vocab,
        max_len=10,
        device=torch.device("cpu"),
        clf=clf,
        vectorizer=vectorizer,
        alpha=alpha,
        threshold=threshold,
        configuration_id="weight7_hybrid_late",
        checkpoint_path=Path("fake.pt"),
    )


def test_load_fusion_params(tmp_path):
    path = tmp_path / "hybrid.json"
    path.write_text(
        json.dumps(
            {
                "configuration_id": "weight7_hybrid_late",
                "fusion": {"alpha": 0.3, "threshold": 0.6},
            }
        ),
        encoding="utf-8",
    )
    alpha, threshold, configuration_id = load_fusion_params(path)
    assert alpha == 0.3
    assert threshold == 0.6
    assert configuration_id == "weight7_hybrid_late"


def test_score_message_marks_toxic_when_fused_prob_above_threshold():
    # logit 10 → p_lstm ≈ 1.0; toxic-ish text keeps p_svm high → fused toxic
    predictor = _tiny_predictor(logit=10.0, alpha=0.3, threshold=0.6)
    result = score_message(predictor, "kys you are trash")
    assert result["prediction"] == 1
    assert result["label"] == "toxic"
    assert result["probability"] >= predictor.threshold
    assert 0.0 <= result["p_lstm"] <= 1.0
    assert 0.0 <= result["p_svm"] <= 1.0


def test_score_message_marks_non_toxic_when_fused_prob_below_threshold():
    # logit -10 → p_lstm ≈ 0.0; safe text keeps p_svm low → fused non-toxic
    predictor = _tiny_predictor(logit=-10.0, alpha=0.3, threshold=0.6)
    result = score_message(predictor, "gg wp")
    assert result["prediction"] == 0
    assert result["label"] == "non-toxic"
    assert result["probability"] < predictor.threshold


def test_format_score_includes_components():
    text = format_score(
        {
            "label": "toxic",
            "probability": 0.8123,
            "p_lstm": 0.9,
            "p_svm": 0.7,
            "alpha": 0.3,
            "threshold": 0.6,
        }
    )
    assert "toxic" in text
    assert "lstm=" in text
    assert "svm=" in text


def test_resolve_checkpoint_uses_explicit_path(tmp_path):
    path = tmp_path / "custom.pt"
    path.write_bytes(b"x")
    assert resolve_checkpoint(path) == path


@pytest.mark.skipif(
    not DEFAULT_CHECKPOINT.is_file() and not config.CKPT_PATH.is_file(),
    reason="LSTM checkpoint not present locally",
)
@pytest.mark.skipif(
    not DEFAULT_HYBRID_JSON.is_file(),
    reason="Hybrid run JSON not present",
)
@pytest.mark.skipif(
    not config.DATA_PATH.is_file(),
    reason="Prepared L2DTnH CSV not present",
)
def test_integration_scores_sample_chat_lines():
    from predict import load_predictor

    predictor = load_predictor(device_name="cpu")
    safe = score_message(predictor, "gg wp everyone good game")
    toxic = score_message(predictor, "uninstall the game you are trash")
    assert safe["label"] in {"toxic", "non-toxic"}
    assert toxic["label"] in {"toxic", "non-toxic"}
    assert 0.0 <= safe["probability"] <= 1.0
    assert 0.0 <= toxic["probability"] <= 1.0
    # Soft sanity: the clearly abusive line should not score lower than the polite one.
    assert toxic["probability"] >= safe["probability"]
