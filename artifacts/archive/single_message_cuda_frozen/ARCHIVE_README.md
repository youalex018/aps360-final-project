# Single-message CUDA frozen run (archived)

**Archived:** 2026-07-16  
**Purpose:** Safe copy of the frozen `weight_7` single-message LSTM screen before replacing `artifacts/` with a full upload from the context/hybrid Colab run.

## What this preserves

- **Freeze record:** `frozen_lstm_config.json` (winner `weight_7`, three-seed val/test aggregates)
- **Canonical metrics:** `lstm_metrics.json`, `experiment_summary.json`, `error_analysis.json`
- **Predictions:** `lstm_predictions.csv`, `baseline_predictions.csv`, per-seed `experiments/*_test_predictions.csv`
- **Baseline reference:** `baseline_metrics.json`
- **Data audit:** `data_audit.json` (pre-`msg_index` prepare if not yet re-run locally)
- **Report inputs:** `qualitative_examples.csv`, `report_metric_manifest.json`
- **Checkpoint:** `best_model.pt` (seed-42 representative `weight_7`)
- **Full screen:** all files under `experiments/` from the single-message validation screen

## Not duplicated here

- `artifacts/archive/tribunal_first_iteration/` — Tribunal role-proxy first iteration
  (formerly `artifacts/old_results/` + `first_iteration_metrics.json`)
- Later hybrid freeze records and context screens live in the live `artifacts/`
  tree / `artifacts/archive/nonselected_screens/` after the 2026-08-04 cleanup.

## After uploading new `artifacts/` from Colab

1. Restore progress-report evidence from this folder if needed:
   - `lstm_metrics.json`, `frozen_lstm_config.json`, `experiment_summary.json`
2. New context/hybrid outputs should appear at:
   - `frozen_context_hybrid_config.json`
   - `context_hybrid_metrics.json`
   - `context_hybrid_experiment_summary.json`
   - `experiments/ctx_k*` and `experiments/*hybrid_late*`
3. Only regenerate `reports/progress/` if `context_hybrid_experiment_summary.json` reports `success_against_baseline: true`.

See `archive_manifest.json` for the exact file list and byte sizes at archive time.
