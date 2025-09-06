from pathlib import Path

# --- DIRECTORY PATHS ---
BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"

# --- DATA FILE PATHS (CORRECTED) ---
USER_MATRIX_PATH = RAW_DATA_DIR / "users.npy"  # Changed from .csv to .npy
BANNER_CAPTIONS_PATH = RAW_DATA_DIR / "banners.txt" # Changed from .csv to .txt
SEGMENTS_ARRAY_PATH = RAW_DATA_DIR / "segments.npy" # Changed from .csv to .npy
TRAIN_PREFERENCES_PATH = RAW_DATA_DIR / "p_train.csv"
TEST_PREFERENCES_PATH = RAW_DATA_DIR / "p_test.csv"

# --- PROJECT PARAMETERS ---
TARGET_COLUMN = 'p'
N_SEGMENTS = 500
USER_VECTOR_DIM = 8
RANDOM_STATE = 42

# ... (rest of the file remains the same)
LGBM_PARAMS = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
}