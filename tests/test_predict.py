"""Tests for the interactive hybrid predictor."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from toxic_chat import config
from toxic_chat.dataset import Vocab, PAD_TOKEN, UNK_TOKEN
from toxic_chat.hybrid import fit_lexical_scorer
from predict import (
    DEFAULT_HYBRID_JSON,
    FROZEN_HYBRID_CONFIG,
    Predictor,
    format_score,
    load_final_test_messages,
    load_fusion_params,
    resolve_checkpoint,
    resolve_frozen_assets,
    score_csv,
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


def test_resolve_frozen_assets_reads_winner(tmp_path):
    ckpt = tmp_path / "winner.pt"
    ckpt.write_bytes(b"x")
    frozen = tmp_path / "frozen.json"
    frozen.write_text(
        json.dumps(
            {
                "top_two": [
                    {
                        "configuration_id": "weight7_hybrid_late",
                        "runs": [
                            {
                                "seed": 42,
                                "checkpoint_path": str(ckpt),
                                "fusion": {"alpha": 0.3, "threshold": 0.6},
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assets = resolve_frozen_assets(frozen_config=frozen)
    assert assets.checkpoint_path == ckpt
    assert assets.configuration_id == "weight7_hybrid_late"
    assert assets.alpha == 0.3
    assert assets.threshold == 0.6


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
    assert "alpha=" in text


def test_resolve_checkpoint_uses_explicit_path(tmp_path):
    path = tmp_path / "custom.pt"
    path.write_bytes(b"x")
    assert resolve_checkpoint(path) == path


def test_load_final_test_messages_skips_blank_text(tmp_path):
    path = tmp_path / "final_chat.csv"
    path.write_text(
        "match_id,message_order,text,label,notes\n"
        "M-1,1,gg wp,,\n"
        "M-1,2,,,skip\n"
        "M-1,3, trash talk ,1,edge\n",
        encoding="utf-8",
    )
    rows = load_final_test_messages(path)
    assert len(rows) == 2
    assert rows[0]["text"] == "gg wp"
    assert rows[1]["text"] == "trash talk"
    assert rows[1]["label"] == "1"


def test_score_csv_writes_predictions(tmp_path):
    predictor = _tiny_predictor(logit=-10.0, alpha=0.3, threshold=0.6)
    csv_path = tmp_path / "final_chat.csv"
    csv_path.write_text(
        "match_id,message_order,text,label,notes\nM-1,1,gg wp,,\n",
        encoding="utf-8",
    )
    out = score_csv(predictor, csv_path)
    assert out.is_file()
    content = out.read_text(encoding="utf-8")
    assert "prediction" in content
    assert "gg wp" in content


@pytest.mark.skipif(
    not config.CKPT_PATH.is_file()
    and not (config.EXPERIMENTS_DIR / "weight_7_seed42.pt").is_file(),
    reason="LSTM checkpoint not present locally",
)
@pytest.mark.skipif(
    not FROZEN_HYBRID_CONFIG.is_file() and not DEFAULT_HYBRID_JSON.is_file(),
    reason="Frozen hybrid config / hybrid run JSON not present",
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
