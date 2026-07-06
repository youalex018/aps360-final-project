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

Training uses a prepared Tribunal CSV at
`data/tribunal/tribunal_chat_100k_balanced.csv` with two columns:

- `text`: raw chat message
- `label`: `0` = safe, `1` = toxic

Create it from `data/tribunal/chatlogs.csv`:

```bash
python prepare_tribunal.py
```

The prep script keeps only `message` and `association_to_offender`, maps
`offender` messages to `1`, maps all other messages to `0`, then writes a
balanced 100k-row sample. `data/tribunal/` is gitignored, so upload the prepared
CSV to Colab/Drive or regenerate it there from the raw CSV.

## Run

```bash
python prepare_tribunal.py  # one-time local/Colab data preparation
python train.py        # train the LSTM, save best weights, print test accuracy
python baseline.py     # train TF-IDF + SVM baseline, print test accuracy
```

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
| `colab_train.ipynb` | Colab workflow for GPU training and baseline comparison. |
