import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple
from src.config import Config

def load_data(path: str) -> pd.DataFrame:
    """Loads CSV data."""
    try:
        df = pd.read_csv(path)
        print(f"✅ Loaded data from {path}: {df.shape}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ File not found at {path}")

def clean_target_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Removes rows where target is above 98th percentile (Specific to AirQo pipeline)."""
    thresh = df[Config.TARGET].quantile(Config.OUTLIER_QUANTILE)
    initial_len = len(df)
    df_clean = df[df[Config.TARGET] <= thresh].reset_index(drop=True)
    print(f"🧹 Removed {initial_len - len(df_clean)} target outliers.")
    return df_clean

def get_train_val_split() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Loads, cleans, and splits data into X_train, X_val, y_train, y_val."""
    df = load_data(Config.RAW_DATA_PATH)
    
    # Basic cleaning
    df = clean_target_outliers(df)
    
    X = df.drop(columns=[Config.TARGET])
    y = df[Config.TARGET]
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
    )
    
    return X_train, X_val, y_train, y_val