"""Validation-only improved hybrid screen (char n-grams + F2 + optional MIN_FREQ=2).

Re-fuses existing ``weight_7`` LSTM checkpoints with the updated lexical branch and
F2 fusion rule under a new configuration id so frozen Batch-A artifacts are not
overwritten. Optionally trains ``weight7_minfreq2`` LSTM seeds when ``--train-minfreq2``
is set. Opens neither the grouped L2DTnH test nor fresh Batch A/B labels.

Abort rule: mean validation toxic F1 across seeds 42-44 must beat 0.671 to
freeze the improved recipe for Batch B.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from toxic_chat import config
from toxic_chat.hybrid import late_fuse_lstm_run
from toxic_chat.train import ExperimentConfig, run_experiment

BASELINE_VAL_F1 = 0.671
HYBRID_CONFIG_ID = "weight7_char_f2_hybrid_late"
HYBRID_F1_CONFIG_ID = "weight7_char_f1_hybrid_late"
MINFREQ2_LSTM_ID = "weight7_minfreq2"
MINFREQ2_HYBRID_ID = "weight7_minfreq2_char_f2_hybrid_late"
FROZEN_IMPROVED = config.ARTIFACTS_DIR / "frozen_improved_hybrid_config.json"
SUMMARY_PATH = config.ARTIFACTS_DIR / "improved_hybrid_screen_summary.json"


def _weight7_base(**overrides) -> ExperimentConfig:
    params = dict(
        positive_weight=7.0,
        embed_dim=100,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        bidirectional=False,
        pooling="last",
        optimizer="adam",
        weight_decay=0.0,
        learning_rate=1e-3,
        batch_size=64,
        max_epochs=15,
        patience=3,
    )
    params.update(overrides)
    return ExperimentConfig(**params)


def _aggregate(results: list[dict]) -> dict:
    summary = {}
    for field in ("f1", "precision", "recall", "balanced_accuracy"):
        values = [run["validation"][field] for run in results]
        summary[field] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "values": values,
        }
    summary["thresholds"] = [run["selected_threshold"] for run in results]
    summary["alphas"] = [run["fusion"]["alpha"] for run in results]
    summary["seeds"] = [run["seed"] for run in results]
    return summary


def _load_lstm(seed: int) -> dict:
    path = config.EXPERIMENTS_DIR / f"weight_7_seed{seed}.json"
    ckpt = config.EXPERIMENTS_DIR / f"weight_7_seed{seed}.pt"
    if not path.is_file() or not ckpt.is_file():
        raise FileNotFoundError(
            f"Missing weight_7 seed {seed} under {config.EXPERIMENTS_DIR}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _fuse_seed(lstm_result: dict, hybrid_config_id: str, device: str | None) -> dict:
    metrics_path = (
        config.EXPERIMENTS_DIR / f"{hybrid_config_id}_seed{lstm_result['seed']}.json"
    )
    # Always recompute: lexical features and F2 selection changed.
    return late_fuse_lstm_run(
        lstm_result,
        hybrid_config_id=hybrid_config_id,
        device_name=device,
        evaluate_test=False,
        metrics_path=metrics_path,
    )


def _train_minfreq2(seed: int, device: str | None) -> dict:
    spec = replace(_weight7_base(config_id=MINFREQ2_LSTM_ID), seed=seed)
    metrics_path = config.EXPERIMENTS_DIR / f"{spec.config_id}_seed{seed}.json"
    ckpt_path = config.EXPERIMENTS_DIR / f"{spec.config_id}_seed{seed}.pt"
    if metrics_path.exists() and ckpt_path.exists():
        result = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not result.get("test_accessed", True):
            print(f"Reusing {result['run_name']}")
            return result
    print(f"Training {spec.config_id}_seed{seed} (MIN_FREQ={config.MIN_FREQ})...")
    return run_experiment(spec, evaluate_test=False, device_name=device)


def run_screen(
    *,
    device: str | None,
    train_minfreq2: bool,
    seeds: tuple[int, ...] = (42, 43, 44),
) -> dict:
    refusion_runs = [
        _fuse_seed(_load_lstm(seed), HYBRID_CONFIG_ID, device) for seed in seeds
    ]
    char_f1_runs = [
        _fuse_seed(_load_lstm(seed), HYBRID_F1_CONFIG_ID, device) for seed in seeds
    ]
    candidates = [
        {
            "configuration_id": HYBRID_CONFIG_ID,
            "base_lstm": "weight_7",
            "runs": refusion_runs,
            "validation_aggregate": _aggregate(refusion_runs),
        },
        {
            "configuration_id": HYBRID_F1_CONFIG_ID,
            "base_lstm": "weight_7",
            "runs": char_f1_runs,
            "validation_aggregate": _aggregate(char_f1_runs),
        },
    ]

    if train_minfreq2:
        minfreq_lstm = [_train_minfreq2(seed, device) for seed in seeds]
        minfreq_hybrid = [
            _fuse_seed(run, MINFREQ2_HYBRID_ID, device) for run in minfreq_lstm
        ]
        candidates.append(
            {
                "configuration_id": MINFREQ2_HYBRID_ID,
                "base_lstm": MINFREQ2_LSTM_ID,
                "runs": minfreq_hybrid,
                "validation_aggregate": _aggregate(minfreq_hybrid),
            }
        )

    winner = max(
        candidates,
        key=lambda item: (
            item["validation_aggregate"]["f1"]["mean"],
            item["validation_aggregate"]["recall"]["mean"],
            item["validation_aggregate"]["precision"]["mean"],
        ),
    )
    mean_f1 = winner["validation_aggregate"]["f1"]["mean"]
    success = mean_f1 > BASELINE_VAL_F1
    payload = {
        "schema_version": 1,
        "selection_rule": (
            f"mean validation toxic F1 across seeds {list(seeds)}; "
            f"must beat {BASELINE_VAL_F1} to freeze for Batch B"
        ),
        "baseline_val_f1_gate": BASELINE_VAL_F1,
        "success_against_gate": success,
        "winner": {
            "configuration_id": winner["configuration_id"],
            "base_lstm": winner["base_lstm"],
            "validation_aggregate": winner["validation_aggregate"],
        },
        "candidates": [
            {
                "configuration_id": item["configuration_id"],
                "base_lstm": item["base_lstm"],
                "validation_aggregate": item["validation_aggregate"],
            }
            for item in candidates
        ],
        "fusion_selection_beta": config.FUSION_SELECTION_BETA,
        "min_freq": config.MIN_FREQ,
        "lexical_features": "word_tfidf+char_wb_3_5",
        "test_accessed": False,
        "batch_a_retuned": False,
    }
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if success:
        frozen = {
            "selection_rule": payload["selection_rule"],
            "success_against_gate": True,
            "top_two": [
                {
                    "configuration_id": winner["configuration_id"],
                    "runs": winner["runs"],
                    "validation_aggregate": winner["validation_aggregate"],
                }
            ],
            "frozen_configuration_id": winner["configuration_id"],
            "min_freq": config.MIN_FREQ,
            "fusion_selection_beta": config.FUSION_SELECTION_BETA,
            "lexical_features": "word_tfidf+char_wb_3_5",
        }
        FROZEN_IMPROVED.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
        print(
            f"FROZE {winner['configuration_id']} "
            f"(mean val F1={mean_f1:.3f} > {BASELINE_VAL_F1})"
        )
    else:
        if FROZEN_IMPROVED.exists():
            FROZEN_IMPROVED.unlink()
        print(
            f"ABORT improved branch: mean val F1={mean_f1:.3f} "
            f"did not beat {BASELINE_VAL_F1}. Keep original frozen hybrid."
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument(
        "--train-minfreq2",
        action="store_true",
        help="Also train weight7_minfreq2 LSTM seeds before fusion",
    )
    args = parser.parse_args()
    payload = run_screen(device=args.device, train_minfreq2=args.train_minfreq2)
    print(json.dumps(payload, indent=2))
    return 0 if payload["success_against_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
