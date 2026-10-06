from pathlib import Path


# --------------------------------------------------
# PROJECT DIRECTORIES
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_ROOT / "models"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"


# --------------------------------------------------
# DATA SETTINGS
# --------------------------------------------------

TARGET_COLUMN = "label"

RANDOM_STATE = 42

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.10


# --------------------------------------------------
# MODEL SETTINGS
# --------------------------------------------------

BATCH_SIZE = 128

LEARNING_RATE = 0.001

EPOCHS = 50

DROPOUT = 0.20

NUM_ATTENTION_HEADS = 4


# --------------------------------------------------
# CREATE DIRECTORIES
# --------------------------------------------------

for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)