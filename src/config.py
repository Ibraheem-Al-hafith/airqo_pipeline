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
    FIGURES_DIR = BASE_DIR / "reports" / "figures"

    # --- Experiment Configs ---
    EXPERIMENT_NAME = "airqo_pm25_prediction"
    
    # --- Data Config ---
    TARGET = "Yield"
    ID_COL = "ID"
    # Columns to drop (high cardinality or irrelevant for training)
    DROP_COLS = ["id","site_id", "site_latitude", "site_longitude", "city", "country", "ID"]
    DATE_COL = ['CropTillageDate', 'RcNursEstDate', 'SeedingSowingTransplanting', 'Harv_date', 'Threshing_date']
    VECTORIZE = ['LandPreparationMethod', 'CropEstMethod', 'NursDetFactor','TransDetFactor','CropbasalFerts',]
    
    # --- Hyperparameters ---
    OUTLIER_QUANTILE = 0.02
    RANDOM_STATE = 42
    MODEL_TYPE = 'lgbm'
    TIME_FEATURES = ["month", "week", "day_of_week",'day']
    CORRELATION_THRESH = 0.85
    MISSING_THRESH = 0.5

# Create directories
os.makedirs(Config.MODEL_DIR, exist_ok=True)
os.makedirs(Config.FIGURES_DIR, exist_ok=True)
os.makedirs(Config.DATA_DIR / "outputs", exist_ok=True)
print("✅ Configuration loaded and directories created.")
