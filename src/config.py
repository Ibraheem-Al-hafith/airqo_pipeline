import os
from pathlib import Path
from typing import List

class Config:
    # Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_PATH = DATA_DIR / "raw" / "Train.csv"
    TEST_DATA_PATH = DATA_DIR / "raw" / "Test.csv"
    MODEL_DIR = BASE_DIR / "models" / "artifacts"
    MLFLOW_URI = f"file://{BASE_DIR / 'models' / 'mlruns'}"

    #Experiment configs:
    EXPERIMENT_NAME: str = "decision_tree_trial"

    # Data Config
    TARGET = "pm2_5"
    ID_COL = "id"
    DROP_COLS = ["site_id", "site_latitude", "site_longitude", "city", "country"]
    DATE_COL = "date"
    
    # Outlier Handling
    OUTLIER_QUANTILE = 0.98
    MISSING_THRESH = 60.0  # Percent
    
    # Model Config
    MODEL: str = 'decision_tree' # supported models: lgbm, xgboost, random_forest, decision_tree
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    N_JOBS = -1
    
    # Features derived from EDA
    TIME_FEATURES: List[str] = ["month", "week", "day_of_week", "hour"]

os.makedirs(Config.MODEL_DIR, exist_ok=True)
os.makedirs(Config.DATA_DIR / "outputs", exist_ok=True)