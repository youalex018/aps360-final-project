# APS360 Final Project - Toxic Gaming Chat Classifier

Classifies League of Legends chat lines as `toxic` (1) or `safe` (0).

Two models share the same preprocessing and data splits so their accuracies are
directly comparable:

- **LSTM** (`model.py` + `train.py`): learned embeddings → LSTM → sigmoid.
- **Baseline** (`baseline.py`): TF-IDF features → Linear SVM.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (PowerShell)
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

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
- fixed `split`: approximately 70/15/15 by complete chat log

It removes token-empty rows and normalized texts carrying contradictory labels,
then saves a processing audit to `results/data_audit.json`. Complete chat logs
never cross splits.

`prepare_tribunal.py` preserves the original 100k offender-role proxy experiment
for comparison, but those labels identify who spoke rather than whether each
message is toxic and are no longer used for model development.

## Run

```bash
python prepare_l2dtnh.py  # clean, audit, and create grouped splits
python baseline.py        # class-balanced TF-IDF + SVM metrics/predictions
python train.py           # weighted LSTM training, checkpoint, full metrics
python generate_report_assets.py
```

Small result files are written to `results/`; generated report figures are
written to `report/progress/figures/`. The untouched final-data procedure is in
`FINAL_TEST_PROTOCOL.md`.

For GPU training, open `colab_train.ipynb` in Google Colab, enable a GPU runtime,
mount Drive, and run the notebook cells.

## Layout

| File | Purpose |
| --- | --- |
| `config.py` | Hyperparameters and paths shared across scripts. |
| `dataset.py` | Cleaning, slang expansion, vocab, `Dataset`, `DataLoader`s. |
| `model.py` | `ToxicChatLSTM` architecture. |
| `train.py` | Training/validation loops with best-model checkpointing. |
| `baseline.py` | scikit-learn TF-IDF + SVM baseline. |
| `prepare_tribunal.py` | Converts raw Tribunal chat logs into a balanced `text,label` CSV. |
| `prepare_l2dtnh.py` | Cleans expert labels, audits data quality, and assigns grouped splits. |
| `generate_report_assets.py` | Generates report plots, diagrams, and qualitative examples. |
| `colab_train.ipynb` | Colab workflow for GPU training and baseline comparison. |
