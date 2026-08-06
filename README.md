# APS360 Final Project - Toxic Gaming Chat Classifier

Classifies League of Legends chat lines as `toxic` (1) or `safe` (0).

Two models share the same preprocessing and data splits so their accuracies are
directly comparable:

- **LSTM** (`model.py` + `train.py`): learned embeddings → packed LSTM → logit.
- **Baseline** (`baseline.py`): TF-IDF features → Linear SVM.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (PowerShell)
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

The optional Windows chat collector also requires the
[Tesseract OCR executable](https://github.com/UB-Mannheim/tesseract/wiki).
Install it and add its directory to `PATH`, or pass
`--tesseract-cmd "C:\path\to\tesseract.exe"` to the collector commands.

## Data

Model development uses the expert-annotated English L2DTnH corpus from
[Ave et al. (2026)](https://github.com/irdin-pekaric/esorics26_toxicity).
Download
`model_creation/1_dataset/2_16000_chatlogs_english_only.csv` to
`data/l2dtnh/l2dtnh_english.csv`, then run:

```bash
python prepare_l2dtnh.py
```

The script writes `data/l2dtnh/l2dtnh_prepared.csv` with:

- `raw_text` and normalized `text`
- expert `label`: `0` = non-toxic, `1` = toxic
- `group_id`: source chat-log identifier
- `msg_index`: within-match order (by raw `time`, then `id`)
- fixed `split`: approximately 70/15/15 by complete chat log

It removes token-empty rows and normalized texts carrying contradictory labels,
then saves a processing audit to `artifacts/data_audit.json`. Complete chat logs
never cross splits.

`prepare_tribunal.py` preserves the original 100k offender-role proxy experiment
for comparison, but those labels identify who spoke rather than whether each
message is toxic and are no longer used for model development.

## Run

```bash
python prepare_l2dtnh.py  # clean, audit, grouped splits, msg_index order
python baseline.py        # validation target; does not access test
python train.py           # one validation-only corrected-current LSTM run
python run_lstm_experiments.py --screen-only  # legacy single-message screen
python run_context_hybrid_experiments.py --screen-only  # context K + hybrid on val
# Only after the winner is frozen:
python run_context_hybrid_experiments.py
# Regenerate progress-report assets only if mean test F1 beats the SVM baseline:
python generate_report_assets.py
```

### Interactive demo

Score chat lines with the frozen primary hybrid. By default `predict.py` reads
`artifacts/frozen_context_hybrid_config.json` (currently `weight7_hybrid_late`,
seed 42 → `artifacts/experiments/weight_7_seed42.pt` / `artifacts/best_model.pt`)
and the prepared L2DTnH train split for the lexical arm.

```bash
python predict.py
python predict.py --once "gg ez mid diff" --show-tokens
python predict.py --csv                       # data/final_test/final_chat.csv
python predict.py --csv data/final_test/final_chat.csv --output artifacts/final_test/final_hybrid_predictions.csv
```

`run_context_hybrid_experiments.py` screens same-match context windows (`K∈{1,2,3}`),
late-fuses the best context LSTM (and a `weight_7` control) with train-only TF-IDF
LinearSVC scores, freezes by mean validation toxic-class F1 across seeds 42–44,
then opens the grouped test once. Artifacts land under `artifacts/` as
`frozen_context_hybrid_config.json`, `context_hybrid_metrics.json`, and
`context_hybrid_experiment_summary.json`.

After Batch~A, `run_improved_hybrid_screen.py` screens character n-gram lexical
branches (`weight7_char_f1_hybrid_late` / F2 variants) without opening Batch~A
labels. If mean val F1 beats 0.671 it writes
`artifacts/frozen_improved_hybrid_config.json` for Batch~B:

```bash
python run_improved_hybrid_screen.py --device cpu
# optional: also train MIN_FREQ=2 LSTMs
python run_improved_hybrid_screen.py --device cpu --train-minfreq2
python predict.py --batch-b
python evaluate_final_test.py --batch-b
```

Place Batch~B labels at `data/final_test/batch_b/final_chat.csv` only after the
improved freeze exists.

`run_lstm_experiments.py` retains the earlier single-message screen (frozen
`weight_7`, which did not beat the SVM). Live evidence stays under
`artifacts/` (freeze records, development metrics, `best_model.pt`) with only
the frozen primary checkpoints in `artifacts/experiments/`
(`weight_7_seed{42,43,44}` and `weight7_hybrid_late_seed{42,43,44}`). Write
fresh-test one-shot outputs to `artifacts/final_test/`. Historical screens and
Tribunal-era files live under `artifacts/archive/` (see `artifacts/LAYOUT.txt`).
Generated figures go to `reports/progress/`. The fresh-data procedure is in
`FINAL_TEST_PROTOCOL.md`.

For GPU training, open `colab_train.ipynb` in Google Colab, enable a GPU runtime,
mount Drive, and run the notebook cells.

## Fresh chat collection

Riot's public and local APIs do not expose in-game team/all chat text. I wanted some way of collecting data that could be automated. `collect_chat.py` therefore passively captures only the calibrated part of the League window and runs local Optical Character Recognition (OCR). It does not inject input, inspect game memory, or label messages. Use it only for your own matches or voluntarily supplied logs under `FINAL_TEST_PROTOCOL.md`.

Use borderless or windowed mode. Finish loading into a Practice Tool or custom
game (lobby/champion-select chat is rejected), open the in-game chat, then
select only the chat rectangle:

```bash
python collect_chat.py list-windows  # optional troubleshooting
python collect_chat.py calibrate
python collect_chat.py collect
python collect_chat.py diagnose --session M-YYYYMMDD-xxxxxxxx  # offline replay gate
python collect_chat.py discard-session --session M-YYYYMMDD-xxxxxxxx  # technical failures only
python collect_chat.py review
python collect_chat.py status
```

Run `collect` once per match. The command waits for the read-only Live Client
Data API to become available and stops after the match ends. Default polling is
**0.75s**. The collector filters system/ping noise, emits high-confidence player
lines on first trailing sighting (uncertain/low-confidence still need a second
frame), keeps pending across a few missed frames, and aborts if the queued rate
exceeds 30 candidates/minute. Only messages actually rendered on the observed
player's screen can be captured; muted, disabled, faded, filtered, or enemy
team-only messages are unavailable. Keep in-game chat open (Enter) during
collection so lines remain visible across polls.

Before reviewing a suspicious session, run `diagnose` twice and confirm the
replay fingerprint matches with ≤150 emitted candidates. Mark irrecoverable
collector failures with `discard-session` so they never enter
`final_chat.csv`.

OCR candidates and screenshots stay in the gitignored
`data/final_test/raw/` directory. During `review`, compare every candidate with
its screenshot, correct transcription errors, and exclude only empty, system,
non-player, or duplicate OCR records. Drop shortcuts: `s` = system, `o` = OCR
duplicate, `n` = non-player; or `d` then `s`/`o`/`e`/`n` (Enter repeats the
last drop reason). The command removes roster names, Riot
IDs, and links, then rebuilds:

- `data/final_test/final_chat.csv` — anonymous text with blank human-label fields
- `data/final_test/collection_manifest.json` — dates and protocol counters
- `data/final_test/exclusion_audit.json` — objective exclusions without text

Delete raw screenshots when prompted after confirming the sanitized export.
Do not use apparent toxicity when deciding which messages to retain.

After the anonymized CSV is ready, label messages interactively:

```bash
python label_final_test.py
python label_final_test.py --start 50      # resume near a CSV row
python label_final_test.py --relabel       # revisit already-labeled rows
```

Type `0` (non-toxic) or `1` (toxic) for each line; `s` skips, `b` undoes the
previous label, `n` adds a note, `q` quits. Each change is saved immediately.

## Layout

| File | Purpose |
| --- | --- |
| `config.py` | Hyperparameters and paths shared across scripts. |
| `dataset.py` | Cleaning, slang expansion, vocab, context windows, `DataLoader`s. |
| `model.py` | `ToxicChatLSTM` architecture. |
| `train.py` | Validation-F1 checkpointing, threshold selection, and frozen test evaluation. |
| `hybrid.py` | Train-only TF-IDF LinearSVC late fusion with validation-tuned blend. |
| `predict.py` | Interactive / one-shot terminal scorer for the frozen hybrid model. |
| `label_final_test.py` | Terminal `0`/`1` labeler for `data/final_test/final_chat.csv`. |
| `baseline.py` | scikit-learn TF-IDF + SVM baseline. |
| `run_lstm_experiments.py` | Legacy single-message validation screen and frozen evaluation. |
| `run_context_hybrid_experiments.py` | Context-window + hybrid screen, three-seed freeze, test once. |
| `prepare_embeddings.py` | Optional vocabulary-aligned 100d GloVe initialization. |
| `prepare_tribunal.py` | Converts raw Tribunal chat logs into a balanced `text,label` CSV. |
| `prepare_l2dtnh.py` | Cleans expert labels, audits data quality, assigns grouped splits + `msg_index`. |
| `generate_report_assets.py` | Generates report plots, diagrams, and qualitative examples. |
| `colab_train.ipynb` | Colab workflow for GPU context/hybrid training. |
| `collect_chat.py` | Calibrates, captures, reviews, and anonymizes fresh LoL chat using screen OCR. |
