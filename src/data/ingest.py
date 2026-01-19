import itertools
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple
from src.config import Config
from pathlib import Path

def load_data(path: str|Path) -> pd.DataFrame:
    """Loads CSV data."""
    try:
        df = pd.read_csv(path)
        print(f"✅ Loaded data from {path}: {df.shape}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ File not found at {path}")

def clean_target_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Removes rows where target is above threshould percentile (Specific to AirQo pipeline)."""
    upper, lower = df[Config.TARGET].quantile(1 - Config.OUTLIER_QUANTILE), df[Config.TARGET].quantile(Config.OUTLIER_QUANTILE)
    initial_len = len(df)
    df_clean = df.copy()
    # df_clean[Config.TARGET] = np.clip(df[Config.TARGET].values, lower, upper)
    df_clean = df[(df[Config.TARGET] > lower ) & (df[Config.TARGET]< upper)].reset_index(drop=True)
    print(f"🧹 Removed {initial_len - len(df_clean)} target outliers.")
    return df_clean

def get_X_y_folds() -> Tuple[pd.DataFrame, pd.Series, dict]:
    """Loads, cleans, and splits data into X_train, X_val, y_train, y_val."""
    df = load_data(Config.RAW_DATA_PATH)
    
    # Basic cleaning
    df = clean_target_outliers(df)
    
    folds = get_train_val_folds(df)

    X = df.drop(columns=[Config.TARGET])
    y = df[Config.TARGET]
    
    return X, y, folds

def get_train_val_folds(X):
    """Custom cross validation method (split data by cities)"""
    unique_cities = X['city'].unique()
    folds = {}
    for i, cities in enumerate(get_all_combinations(unique_cities.tolist())):
        if len(X[X['city'].isin(cities)].index.values) < len(X[~ X['city'].isin(cities)].index.values):
            continue
        folds[i] = (X[X['city'].isin(cities)].index.values, X[~ X['city'].isin(cities)].index.values)
        assert len(folds[i][0])+len(folds[i][1]) == len(X), f"Shape mismatch, got {len(folds[i][0])} and {folds[i][1]} \n X length :{len(X)}"
        # print(cities)
    return folds


def get_all_combinations(input_list):
    """
    Generates all possible combinations (subsets) of a given list,
    including an empty set.
    """
    all_combinations = []
    for r in range(1, len(input_list)):
        # Generate combinations of length 'r'
        combinations_r = list(itertools.combinations(input_list, r))
        all_combinations.extend(combinations_r)
    return all_combinations

