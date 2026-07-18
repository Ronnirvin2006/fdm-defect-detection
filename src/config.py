import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = Path(os.getenv("FDM_DATA_RAW", PROJECT_ROOT / "data" / "raw"))
DATA_RAW_IS_EXTERNAL = "FDM_DATA_RAW" in os.environ
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
LOGS_DIR = PROJECT_ROOT / "logs"
KAGGLE_CONFIG_DIR = PROJECT_ROOT / ".kaggle"

KAGGLE_DATASET = "wengmhu/fdm-3d-printing-defect-dataset"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

WRITABLE_DIRS = [
    DATA_PROCESSED,
    MODELS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    PREDICTIONS_DIR,
    LOGS_DIR,
    KAGGLE_CONFIG_DIR,
]

if not DATA_RAW_IS_EXTERNAL:
    WRITABLE_DIRS.insert(0, DATA_RAW)

for path in WRITABLE_DIRS:
    path.mkdir(parents=True, exist_ok=True)
