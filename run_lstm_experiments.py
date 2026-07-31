"""Run the predeclared validation-only LSTM screen and frozen evaluation."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil

import numpy as np

import config
from baseline import run_baseline
from prepare_embeddings import build_embedding_artifact
from train import (
    ExperimentConfig,
    evaluate_checkpoint,
    experiment_config_from_dict,
    metric_summary,
    run_experiment,
)


SCREEN_CONFIGS = [
    ExperimentConfig(config_id="corrected_current"),
    ExperimentConfig(config_id="weight_3", positive_weight=3.0),
    ExperimentConfig(config_id="weight_5", positive_weight=5.0),
    ExperimentConfig(config_id="weight_7", positive_weight=7.0),
    ExperimentConfig(config_id="weight_auto_adamw", optimizer="adamw"),
    ExperimentConfig(
        config_id="regularized_wd",
        optimizer="adamw",
        weight_decay=1e-4,
    ),
    ExperimentConfig(config_id="dropout_05", dropout=0.5),
    ExperimentConfig(
        config_id="small_1x96",
        hidden_dim=96,
        num_layers=1,
        dropout=0.5,
        positive_weight=5.0,
        optimizer="adamw",
        weight_decay=1e-4,
    ),
    ExperimentConfig(
        config_id="bilstm_pool_w5",
        hidden_dim=64,
        num_layers=1,
        dropout=0.5,
        bidirectional=True,
        pooling="mean_max",
        positive_weight=5.0,
        optimizer="adamw",
        weight_decay=1e-4,
    ),
    ExperimentConfig(
        config_id="bilstm_pool_auto",
        hidden_dim=64,
        num_layers=1,
        dropout=0.5,
        bidirectional=True,
        pooling="mean_max",
        optimizer="adamw",
        weight_decay=1e-4,
    ),
]


def _paths(spec: ExperimentConfig) -> tuple[Path, Path]:
    stem = f"{spec.config_id}_seed{spec.seed}"
    return (
        config.EXPERIMENTS_DIR / f"{stem}.json",
        config.EXPERIMENTS_DIR / f"{stem}.pt",
    )


def run_or_load(spec: ExperimentConfig, device: str | None) -> dict:
    metrics_path, checkpoint_path = _paths(spec)
    if metrics_path.exists() and checkpoint_path.exists():
        result = json.loads(metrics_path.read_text(encoding="utf-8"))
        saved = result.get("experiment_config", {})
        if (
            not result.get("test_accessed", True)
            and asdict(experiment_config_from_dict(saved)) == asdict(spec)
        ):
            print(f"Reusing completed validation run: {result['run_name']}")
            return result
    return run_experiment(spec, evaluate_test=False, device_name=device)


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
    summary["seeds"] = [run["seed"] for run in results]
    return summary


def _pretrained_specs(embedding_path: Path) -> list[ExperimentConfig]:
    common = dict(
        embed_dim=100,
        hidden_dim=64,
        num_layers=1,
        dropout=0.5,
        bidirectional=True,
        pooling="mean_max",
        positive_weight=5.0,
        optimizer="adamw",
        weight_decay=1e-4,
        embedding_path=str(embedding_path),
    )
    return [
        ExperimentConfig(
            config_id="bilstm_glove_frozen",
            freeze_embeddings=True,
            **common,
        ),
        ExperimentConfig(
            config_id="bilstm_glove_finetuned",
            freeze_embeddings=False,
            **common,
        ),
    ]


def run_screen(device: str | None) -> tuple[dict, list[dict]]:
    baseline = run_baseline(evaluate_test=False)
    baseline_f1 = baseline["validation"]["f1"]
    results = [run_or_load(spec, device) for spec in SCREEN_CONFIGS]
    best_moderate = max(results, key=lambda run: run["validation"]["f1"])
    if best_moderate["validation"]["f1"] < baseline_f1:
        embedding_path = config.ARTIFACTS_DIR / "glove_100d_l2dtnh.pt"
        if not embedding_path.exists():
            build_embedding_artifact(embedding_path)
        results.extend(
            run_or_load(spec, device)
            for spec in _pretrained_specs(embedding_path)
        )
    else:
        print(
            "Pretrained-embedding escalation skipped: the moderate LSTM "
            "already meets/exceeds baseline validation F1."
        )
    return baseline, results


def run_three_seeds(
    screen_results: list[dict],
    device: str | None,
) -> tuple[list[dict], dict]:
    top_two = sorted(
        screen_results,
        key=lambda run: (
            run["validation"]["f1"],
            run["validation"]["precision"],
        ),
        reverse=True,
    )[:2]
    candidate_runs = []
    for candidate in top_two:
        base = experiment_config_from_dict(candidate["experiment_config"])
        runs = [
            run_or_load(replace(base, seed=seed), device)
            for seed in (42, 43, 44)
        ]
        candidate_runs.append(
            {
                "configuration_id": base.config_id,
                "experiment_config": asdict(base),
                "runs": runs,
                "validation_aggregate": _aggregate(runs),
            }
        )
    winner = max(
        candidate_runs,
        key=lambda candidate: (
            candidate["validation_aggregate"]["f1"]["mean"],
            candidate["validation_aggregate"]["precision"]["mean"],
        ),
    )
    frozen = {
        "selection_rule": (
            "Highest mean validation toxic-class F1 across seeds 42, 43, 44; "
            "mean precision breaks ties."
        ),
        "top_two": candidate_runs,
        "winner_configuration_id": winner["configuration_id"],
        "winner_experiment_config": winner["experiment_config"],
        "winner_validation_aggregate": winner["validation_aggregate"],
        "test_observed_during_selection": False,
    }
    (config.ARTIFACTS_DIR / "frozen_lstm_config.json").write_text(
        json.dumps(frozen, indent=2),
        encoding="utf-8",
    )
    return candidate_runs, winner


def final_evaluation(
    baseline_validation: dict,
    all_screen_results: list[dict],
    candidates: list[dict],
    winner: dict,
    device: str | None,
) -> dict:
    baseline = run_baseline(evaluate_test=True)
    seed_tests = []
    for run in winner["runs"]:
        checkpoint_path = config.ROOT / run["checkpoint_path"]
        prediction_path = config.EXPERIMENTS_DIR / (
            f"{run['run_name']}_test_predictions.csv"
        )
        seed_tests.append(
            evaluate_checkpoint(
                checkpoint_path,
                device_name=device,
                prediction_path=prediction_path,
            )
        )

    representative = next(run for run in seed_tests if run["seed"] == 42)
    representative_training = next(
        run for run in winner["runs"] if run["seed"] == 42
    )
    representative_checkpoint = config.ROOT / representative["checkpoint_path"]
    shutil.copy2(representative_checkpoint, config.CKPT_PATH)
    shutil.copy2(
        config.EXPERIMENTS_DIR
        / f"{winner['configuration_id']}_seed42_test_predictions.csv",
        config.ARTIFACTS_DIR / "lstm_predictions.csv",
    )

    test_aggregate = {}
    for field in ("f1", "precision", "recall", "balanced_accuracy", "accuracy"):
        values = [run["test"][field] for run in seed_tests]
        test_aggregate[field] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "values": values,
        }

    labels = representative["labels"]
    probability_matrix = np.asarray(
        [run["probabilities"] for run in seed_tests],
        dtype=np.float64,
    )
    ensemble_probabilities = probability_matrix.mean(axis=0)
    ensemble_threshold = float(
        np.mean([run["selected_threshold"] for run in seed_tests])
    )
    ensemble = metric_summary(
        labels,
        ensemble_probabilities,
        loss=0.0,
        threshold=ensemble_threshold,
    )
    ensemble.pop("loss")

    canonical = dict(representative_training)
    canonical.update(
        {
            "test_accessed": True,
            "test": representative["test"],
            "seed_runs": [
                {
                    key: value
                    for key, value in run.items()
                    if key not in {"probabilities", "labels"}
                }
                for run in seed_tests
            ],
            "validation_aggregate": winner["validation_aggregate"],
            "test_aggregate": test_aggregate,
            "ensemble_secondary": ensemble,
            "frozen_configuration_id": winner["configuration_id"],
        }
    )
    (config.ARTIFACTS_DIR / "lstm_metrics.json").write_text(
        json.dumps(canonical, indent=2),
        encoding="utf-8",
    )

    error_analysis = {
        "baseline_confusion_matrix": baseline["test"]["confusion_matrix"],
        "lstm_seed42_confusion_matrix": representative["test"][
            "confusion_matrix"
        ],
        "baseline_test_f1": baseline["test"]["f1"],
        "lstm_seed42_test_f1": representative["test"]["f1"],
        "lstm_three_seed_test_f1_mean": test_aggregate["f1"]["mean"],
        "lstm_three_seed_test_f1_std": test_aggregate["f1"]["std"],
        "success_against_baseline": (
            test_aggregate["f1"]["mean"] > baseline["test"]["f1"]
        ),
        "note": (
            "The grouped test split was opened only after configuration and "
            "per-seed thresholds were frozen from validation results."
        ),
    }
    (config.ARTIFACTS_DIR / "error_analysis.json").write_text(
        json.dumps(error_analysis, indent=2),
        encoding="utf-8",
    )

    summary = {
        "protocol": {
            "primary_metric": "toxic-class F1",
            "threshold_grid": list(config.THRESHOLD_GRID),
            "screen_seed": 42,
            "final_seeds": [42, 43, 44],
            "test_opened_after_freeze": True,
        },
        "baseline_validation": baseline_validation["validation"],
        "screen": [
            {
                "configuration_id": run["configuration_id"],
                "validation": run["validation"],
                "best_epoch": run["best_epoch"],
            }
            for run in all_screen_results
        ],
        "three_seed_candidates": candidates,
        "winner": winner["configuration_id"],
        "baseline_test": baseline["test"],
        "lstm_test_aggregate": test_aggregate,
        "lstm_seed_tests": canonical["seed_runs"],
        "ensemble_secondary": ensemble,
        "error_analysis": error_analysis,
    }
    (config.ARTIFACTS_DIR / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument(
        "--screen-only",
        action="store_true",
        help="Stop after validation screening and do not open the test split.",
    )
    args = parser.parse_args()
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    baseline, screen_results = run_screen(args.device)
    candidates, winner = run_three_seeds(screen_results, args.device)
    if args.screen_only:
        print(f"Frozen winner: {winner['configuration_id']}; test not accessed.")
        return
    summary = final_evaluation(
        baseline,
        screen_results,
        candidates,
        winner,
        args.device,
    )
    print(
        f"Winner: {summary['winner']}; mean test F1 "
        f"{summary['lstm_test_aggregate']['f1']['mean']:.3f} vs baseline "
        f"{summary['baseline_test']['f1']:.3f}"
    )


if __name__ == "__main__":
    main()
