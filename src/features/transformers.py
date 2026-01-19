import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from typing import List

# ==========================================
# 1. FIXED CUSTOM TRANSFORMERS
# ==========================================

class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts temporal features and handles feature name propagation."""
    def __init__(self, date_col: str, features_to_extract: List[str] = None):
        self.date_col = date_col
        self.features_to_extract = features_to_extract or ["month", "hour"]

    def fit(self, X: pd.DataFrame, y=None):
        # Store names to satisfy sklearn validation
        self.feature_names_in_ = np.array(X.columns.tolist())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        # Convert to datetime
        X_out[self.date_col] = pd.to_datetime(X_out[self.date_col])
        
        # Add new features
        if "month" in self.features_to_extract:
            X_out[f"{self.date_col}_month"] = X_out[self.date_col].dt.month
        if "hour" in self.features_to_extract:
            X_out[f"{self.date_col}_hour"] = X_out[self.date_col].dt.hour
        if "week" in self.features_to_extract:
            X_out[f"{self.date_col}_week"] = X_out[self.date_col].dt.isocalendar().week.astype(int)
        if "day_of_week" in self.features_to_extract:
            X_out[f"{self.date_col}_dow"] = X_out[self.date_col].dt.dayofweek
        
        # Drop original date col
        return X_out.drop(columns=[self.date_col])

    def get_feature_names_out(self, input_features=None):
        """Mandatory for pipelines to propagate names through ColumnTransformer."""
        if input_features is None:
            input_features = self.feature_names_in_
        
        # Current columns minus the dropped date column
        features = [f for f in input_features if f != self.date_col]
        
        # Add the generated feature names
        new_feats = []
        if "month" in self.features_to_extract: new_feats.append(f"{self.date_col}_month")
        if "hour" in self.features_to_extract: new_feats.append(f"{self.date_col}_hour")
        if "week" in self.features_to_extract: new_feats.append(f"{self.date_col}_week")
        if "day_of_week" in self.features_to_extract: new_feats.append(f"{self.date_col}_dow")
        
        return np.array(features + new_feats)


class OutlierHandler(BaseEstimator, TransformerMixin):
    """Clips outliers and propagates feature names."""
    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.array(X.columns.tolist())
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

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return self.feature_names_in_
        return np.array(input_features)

# ==========================================
# 2. UPDATED PIPELINE CONSTRUCTION
# ==========================================

def build_pipeline(cat_cols: list, num_cols: list, date_col: str, model_obj) -> Pipeline:
    # 1. Feature Engineering (Handles the raw input)
    feature_engineering = Pipeline([
        ('time_extractor', TimeFeatureExtractor(date_col=date_col)),
        ('outlier_clipper', OutlierHandler(factor=1.5))
    ])
    
    # 2. Preprocessing (Note: Names here will be prefixed by 'num__' or 'cat__')
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])
    
    # 3. Final Pipeline
    pipeline = Pipeline([
        ('feature_eng', feature_engineering),
        ('preprocessor', preprocessor),
        ('regressor', model_obj)
    ])
    
    return pipeline

# ==========================================
# 3. VALIDATION TEST
# ==========================================

if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor
    
    # Mock Data
    data = pd.DataFrame({
        'timestamp': ['2023-01-01 10:00:00', '2023-01-01 11:00:00'],
        'temp': [25.5, 30.0],
        'city': ['Nairobi', 'Kampala']
    })
    y = np.array([100, 110])

    # Config
    DATE_COL = 'timestamp'
    NUM_COLS = ['temp', 'timestamp_month', 'timestamp_hour'] # Features after extraction
    CAT_COLS = ['city']

    # Build and Fit
    pipe = build_pipeline(CAT_COLS, NUM_COLS, DATE_COL, RandomForestRegressor())
    pipe.fit(data, y)

    # TEST: Extracting feature names
    # Slice the pipeline to get all steps EXCEPT the regressor
    features_out = pipe[:-1].get_feature_names_out()
    
    print("✅ Successfully extracted feature names:")
    print(features_out)