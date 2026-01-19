import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional, Union

class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts temporal features from a date column.
    """
    def __init__(self, date_col: str, features_to_extract: List[str] = None):
        self.date_col = date_col
        self.features_to_extract = features_to_extract or ["month", "hour"]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        if self.date_col not in X_out.columns:
            return X_out # Handle cases where date is already processed
            
        X_out[self.date_col] = pd.to_datetime(X_out[self.date_col])
        
        if "month" in self.features_to_extract:
            X_out[f"{self.date_col}_month"] = X_out[self.date_col].dt.month
        if "hour" in self.features_to_extract:
            X_out[f"{self.date_col}_hour"] = X_out[self.date_col].dt.hour
        if "week" in self.features_to_extract:
            X_out[f"{self.date_col}_week"] = X_out[self.date_col].dt.isocalendar().week.astype(int)
        if "day_of_week" in self.features_to_extract:
            X_out[f"{self.date_col}_dow"] = X_out[self.date_col].dt.dayofweek

        # Drop original date col as it's not ML-ready
        return X_out.drop(columns=[self.date_col])

class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Clips values based on IQR to handle extreme outliers.
    """
    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.lower_bounds_[col] = Q1 - (self.factor * IQR)
            self.upper_bounds_[col] = Q3 + (self.factor * IQR)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col, lower in self.lower_bounds_.items():
            if col in X_out.columns:
                upper = self.upper_bounds_[col]
                X_out[col] = np.clip(X_out[col], lower, upper)
        return X_out