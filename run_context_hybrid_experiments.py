"""Validation-only context-window + late-fusion screen, then frozen test once."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

import config
from baseline import run_baseline
from hybrid import late_fuse_lstm_run
from train import (
    ExperimentConfig,
    evaluate_checkpoint,
    experiment_config_from_dict,
    run_experiment,
)


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


def context_screen_configs() -> list[ExperimentConfig]:
    return [
        _weight7_base(
            config_id=f"ctx_k{k}_weight7",
            context_k=k,
            max_len=config.CONTEXT_MAX_LEN,
            hybrid=False,
        )
        for k in config.CONTEXT_K_GRID
    ]


def _paths(spec: ExperimentConfig) -> tuple[Path, Path]:
    stem = f"{spec.config_id}_seed{spec.seed}"
    return (
        config.EXPERIMENTS_DIR / f"{stem}.json",
        config.EXPERIMENTS_DIR / f"{stem}.pt",
    )


def _configs_match(saved: dict, spec: ExperimentConfig) -> bool:
    return asdict(experiment_config_from_dict(saved)) == asdict(spec)


def run_or_load(spec: ExperimentConfig, device: str | None) -> dict:
    metrics_path, checkpoint_path = _paths(spec)
    if metrics_path.exists() and checkpoint_path.exists():
        result = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            not result.get("test_accessed", True)
            and _configs_match(result.get("experiment_config", {}), spec)
        ):
            print(f"Reusing completed validation run: {result['run_name']}")
            return result
    return run_experiment(spec, evaluate_test=False, device_name=device)


def _load_hybrid_or_fuse(
    lstm_result: dict,
    hybrid_config_id: str,
    device: str | None,
) -> dict:
    metrics_path = config.EXPERIMENTS_DIR / f"{hybrid_config_id}_seed{lstm_result['seed']}.json"
    if metrics_path.exists():
        result = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            not result.get("test_accessed", True)
            and result.get("base_run_name") == lstm_result["run_name"]
            and result.get("fusion", {}).get("test_used_for_selection") is False
        ):
            print(f"Reusing completed hybrid run: {result['run_name']}")
            return result
    return late_fuse_lstm_run(
        lstm_result,
        hybrid_config_id=hybrid_config_id,
        device_name=device,
        evaluate_test=False,
        metrics_path=metrics_path,
    )


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
    if results and "fusion" in results[0]:
        summary["alphas"] = [run["fusion"]["alpha"] for run in results]
    return summary


def _rank_key(run: dict) -> tuple:
    return (
        run["validation"]["f1"],
        run["validation"]["precision"],
        run["validation"]["balanced_accuracy"],
    )


def run_context_screen(device: str | None) -> list[dict]:
    results = []
    for spec in context_screen_configs():
        results.append(run_or_load(replace(spec, seed=42), device))
    return results


def run_hybrid_stage(
    context_results: list[dict],
    device: str | None,
) -> list[dict]:
    best_context = max(context_results, key=_rank_key)
    best_k = experiment_config_from_dict(best_context["experiment_config"]).context_k
    hybrid_results = [
        _load_hybrid_or_fuse(
            best_context,
            f"ctx_k{best_k}_hybrid_late",
            device,
        )
    ]

    weight7_path = config.EXPERIMENTS_DIR / "weight_7_seed42.json"
    weight7_ckpt = config.EXPERIMENTS_DIR / "weight_7_seed42.pt"
    if weight7_path.exists() and weight7_ckpt.exists():
        weight7 = json.loads(weight7_path.read_text(encoding="utf-8"))
        hybrid_results.append(
            _load_hybrid_or_fuse(weight7, "weight7_hybrid_late", device)
        )
    else:
        print(
            "Skipping weight7_hybrid_late control: "
            "artifacts/experiments/weight_7_seed42.{json,pt} not found."
        )
    return hybrid_results


def _materialize_candidate_run(
    template: dict,
    seed: int,
    device: str | None,
) -> dict:
    """Train or fuse one seed for a surviving configuration template."""
    base_cfg = experiment_config_from_dict(template["experiment_config"])
    if base_cfg.hybrid:
        # Hybrid configs reuse an underlying LSTM config_id without "_hybrid_late".
        base_lstm_id = template.get("base_configuration_id")
        if base_lstm_id is None:
            raise ValueError(f"Hybrid template missing base_configuration_id: {template}")
        lstm_cfg = replace(
            experiment_config_from_dict(
                {
                    **template["experiment_config"],
                    "config_id": base_lstm_id,
                    "hybrid": False,
                }
            ),
            seed=seed,
        )
        # Context hybrid keeps context_k; weight7 hybrid has context_k None.
        lstm_run = run_or_load(lstm_cfg, device)
        return _load_hybrid_or_fuse(lstm_run, base_cfg.config_id, device)

    return run_or_load(replace(base_cfg, seed=seed), device)


def run_three_seeds(
    survivors: list[dict],
    device: str | None,
) -> tuple[list[dict], dict]:
    top_two = sorted(survivors, key=_rank_key, reverse=True)[:2]
    candidate_runs = []
    for candidate in top_two:
        runs = [
            _materialize_candidate_run(candidate, seed, device)
            for seed in (42, 43, 44)
        ]
        base = experiment_config_from_dict(candidate["experiment_config"])
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
        key=lambda item: (
            item["validation_aggregate"]["f1"]["mean"],
            item["validation_aggregate"]["precision"]["mean"],
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
    (config.ARTIFACTS_DIR / "frozen_context_hybrid_config.json").write_text(
        json.dumps(frozen, indent=2),
        encoding="utf-8",
    )
    return candidate_runs, winner


def _evaluate_hybrid_test(run: dict, device: str | None) -> dict:
    """Open grouped test once for a frozen hybrid run using saved fusion params."""
    base = json.loads(
        (config.EXPERIMENTS_DIR / f"{run['base_run_name']}.json").read_text(
            encoding="utf-8"
        )
    ) if "base_run_name" in run else None
    # Re-run late fusion with evaluate_test using the same base LSTM checkpoint.
    if base is None:
        # Reconstruct a minimal base result from the hybrid artifact.
        base = {
            "dataset": run["dataset"],
            "run_name": run.get("base_run_name", run["run_name"]),
            "configuration_id": run.get("base_configuration_id", run["configuration_id"]),
            "seed": run["seed"],
            "checkpoint_path": run["checkpoint_path"],
            "best_epoch": run.get("best_epoch"),
            "positive_weight": run.get("positive_weight"),
            "vocab_size": run.get("vocab_size"),
            "parameter_count": run.get("parameter_count"),
            "experiment_config": {
                **run["experiment_config"],
                "config_id": run.get(
                    "base_configuration_id", run["configuration_id"]
                ),
                "hybrid": False,
            },
            "environment": run.get("environment"),
            "history": run.get("history"),
            "validation": run.get("lstm_only_validation", run["validation"]),
        }
    fused = late_fuse_lstm_run(
        base,
        hybrid_config_id=run["configuration_id"],
        device_name=device,
        evaluate_test=True,
    )
    return {
        "configuration_id": fused["configuration_id"],
        "seed": fused["seed"],
        "checkpoint_path": fused["checkpoint_path"],
        "selected_threshold": fused["selected_threshold"],
        "validation": fused["validation"],
        "test": fused["test"],
        "fusion": fused["fusion"],
        "probabilities": None,
        "labels": None,
    }


def final_evaluation(
    baseline_validation: dict,
    screen_results: list[dict],
    hybrid_results: list[dict],
    candidates: list[dict],
    winner: dict,
    device: str | None,
) -> dict:
    baseline = run_baseline(evaluate_test=True)
    seed_tests = []
    winner_is_hybrid = bool(
        winner["experiment_config"].get("hybrid")
        or "fusion" in winner["runs"][0]
    )
    for run in winner["runs"]:
        if winner_is_hybrid:
            seed_tests.append(_evaluate_hybrid_test(run, device))
        else:
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

    test_aggregate = {}
    for field in ("f1", "precision", "recall", "balanced_accuracy", "accuracy"):
        values = [run["test"][field] for run in seed_tests]
        test_aggregate[field] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "values": values,
        }

    representative = next(run for run in seed_tests if run["seed"] == 42)
    representative_training = next(
        run for run in winner["runs"] if run["seed"] == 42
    )
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
            "frozen_configuration_id": winner["configuration_id"],
        }
    )
    (config.ARTIFACTS_DIR / "context_hybrid_metrics.json").write_text(
        json.dumps(canonical, indent=2),
        encoding="utf-8",
    )

    success = test_aggregate["f1"]["mean"] > baseline["test"]["f1"]
    error_analysis = {
        "baseline_test_f1": baseline["test"]["f1"],
        "context_hybrid_seed42_test_f1": representative["test"]["f1"],
        "context_hybrid_three_seed_test_f1_mean": test_aggregate["f1"]["mean"],
        "context_hybrid_three_seed_test_f1_std": test_aggregate["f1"]["std"],
        "success_against_baseline": success,
        "note": (
            "Grouped test opened only after context/hybrid configuration and "
            "fusion parameters were frozen from validation. Progress report "
            "assets should be regenerated only when success_against_baseline "
            "is true."
        ),
    }
    (config.ARTIFACTS_DIR / "context_hybrid_error_analysis.json").write_text(
        json.dumps(error_analysis, indent=2),
        encoding="utf-8",
    )

    summary = {
        "protocol": {
            "primary_metric": "toxic-class F1",
            "context_k_grid": list(config.CONTEXT_K_GRID),
            "alpha_grid": list(config.ALPHA_GRID),
            "threshold_grid": list(config.THRESHOLD_GRID),
            "screen_seed": 42,
            "final_seeds": [42, 43, 44],
            "test_opened_after_freeze": True,
            "progress_report_update_allowed": success,
        },
        "baseline_validation": baseline_validation["validation"],
        "context_screen": [
            {
                "configuration_id": run["configuration_id"],
                "validation": run["validation"],
                "best_epoch": run.get("best_epoch"),
            }
            for run in screen_results
        ],
        "hybrid_stage": [
            {
                "configuration_id": run["configuration_id"],
                "validation": run["validation"],
                "fusion": run.get("fusion"),
            }
            for run in hybrid_results
        ],
        "three_seed_candidates": candidates,
        "winner": winner["configuration_id"],
        "baseline_test": baseline["test"],
        "test_aggregate": test_aggregate,
        "error_analysis": error_analysis,
    }
    (config.ARTIFACTS_DIR / "context_hybrid_experiment_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    if success:
        print(
            "SUCCESS: mean test F1 beats baseline. Safe to regenerate progress "
            "report assets with generate_report_assets.py."
        )
    else:
        print(
            "No progress-report update: mean test F1 did not beat baseline "
            f"({test_aggregate['f1']['mean']:.3f} vs {baseline['test']['f1']:.3f})."
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument(
        "--screen-only",
        action="store_true",
        help="Stop after validation screening/fusion; do not open the test split.",
    )
    args = parser.parse_args()
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    config.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    baseline = run_baseline(evaluate_test=False)
    context_results = run_context_screen(args.device)
    hybrid_results = run_hybrid_stage(context_results, args.device)
    survivors = list(context_results) + list(hybrid_results)
    candidates, winner = run_three_seeds(survivors, args.device)
    if args.screen_only:
        print(
            f"Frozen winner (validation only): {winner['configuration_id']}; "
            "test not accessed."
        )
        return
    summary = final_evaluation(
        baseline,
        context_results,
        hybrid_results,
        candidates,
        winner,
        args.device,
    )
    print(
        f"Winner: {summary['winner']}; mean test F1 "
        f"{summary['test_aggregate']['f1']['mean']:.3f} vs baseline "
        f"{summary['baseline_test']['f1']:.3f}"
    )


if __name__ == "__main__":
    main()
