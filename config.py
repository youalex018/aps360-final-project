"""Single source of truth for hyperparameters and paths.

Centralizing these keeps the LSTM and the baseline trained/evaluated on the
same splits, so their accuracies are directly comparable.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRIBUNAL_RAW_PATH = ROOT / "data" / "tribunal" / "chatlogs.csv"
TRIBUNAL_PREPARED_PATH = ROOT / "data" / "tribunal" / "tribunal_chat_100k_balanced.csv"
L2DTNH_RAW_PATH = ROOT / "data" / "l2dtnh" / "l2dtnh_english.csv"
L2DTNH_PREPARED_PATH = ROOT / "data" / "l2dtnh" / "l2dtnh_prepared.csv"
DATA_PATH = L2DTNH_PREPARED_PATH
ARTIFACTS_DIR = ROOT / "artifacts"
# Backward-compatible alias. All generated evidence now has one authority.
RESULTS_DIR = ARTIFACTS_DIR
EXPERIMENTS_DIR = ARTIFACTS_DIR / "experiments"
FIRST_ITERATION_METRICS_PATH = ARTIFACTS_DIR / "first_iteration_metrics.json"
OLD_RESULTS_DIR = ARTIFACTS_DIR / "old_results"
CKPT_PATH = ARTIFACTS_DIR / "best_model.pt"
REPORT_FIGURES_DIR = ROOT / "reports" / "progress"
SUBSAMPLE_SIZE = 100_000

# Sequence shaping. Chat lines are short, so a small MAX_LEN avoids padding the
# batch with mostly <pad> tokens that dilute the signal.
MAX_LEN = 50
# Context-window runs concatenate neighboring messages; allow a longer sequence.
CONTEXT_MAX_LEN = 100
CONTEXT_K_GRID = (1, 2, 3)
ALPHA_GRID = tuple(round(value / 10, 1) for value in range(0, 11))

# Splits are fractions of the full dataset; test is the remainder.
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
SEED = 42

# Model.
EMBED_DIM = 100
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3

# Optimization.
BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-3
WEIGHT_DECAY = 0.0
EARLY_STOP_PATIENCE = 3
CHECKPOINT_METRIC = "f1"
THRESHOLD_GRID = tuple(round(value / 20, 2) for value in range(1, 20))

# Vocab is capped so rare tokens collapse to <unk>, which keeps the embedding
# table small and improves generalization on unseen chat.
MAX_VOCAB_SIZE = 20000
MIN_FREQ = 1
