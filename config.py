# config.py
import os

# Project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Folders
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# Paths
MODEL_PATH = os.path.join(MODEL_DIR, "logistic_model.joblib")
DATA_FILE = os.path.join(DATA_DIR, "dummy_mortality_data.csv")

# Hyperparameters
TEST_SIZE = 0.2
MAX_ITER = 1000

# Regularization
LOGISTIC_CONFIGS = {
    "baseline": {
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs"
    },
    "l1": {
        "penalty": "l1",
        "C": 1.0,
        "solver": "liblinear"
    },
    "l2": {
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs"
    }
}
