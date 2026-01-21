import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from typing import List
import warnings
from typing import Literal
from ..config import Config
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.validation import check_is_fitted

# ==========================================
# 1. FIXED CUSTOM TRANSFORMERS
# ==========================================
# --- 4.1 Time Feature Extractor ---
class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts temporal features and handles feature name propagation."""
    def __init__(self, date_col: str, features_to_extract: List[str] = None):
        self.date_col = date_col
        self.features_to_extract = features_to_extract or ["month", "hour"]

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.array(X.columns.tolist())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        X_out[self.date_col] = pd.to_datetime(X_out[self.date_col])
        
        if "month" in self.features_to_extract:
            X_out[f"{self.date_col}_month"] = X_out[self.date_col].dt.month
        if "hour" in self.features_to_extract:
            X_out[f"{self.date_col}_hour"] = X_out[self.date_col].dt.hour
        if "week" in self.features_to_extract:
            X_out[f"{self.date_col}_week"] = X_out[self.date_col].dt.isocalendar().week.astype(int)
        if "day_of_week" in self.features_to_extract:
            X_out[f"{self.date_col}_dow"] = X_out[self.date_col].dt.dayofweek
        
        return X_out.drop(columns=[self.date_col])

    def get_feature_names_out(self, input_features=None):
        if input_features is None: input_features = self.feature_names_in_
        features = [f for f in input_features if f != self.date_col]
        new_feats = [f"{self.date_col}_{t}" for t in self.features_to_extract if t in ["month", "hour", "week", "day_of_week"]]
        return np.array(features + new_feats)


# --- 4.2 High Missing Dropper ---
class HighMissingDropper(BaseEstimator, TransformerMixin):
    """Drops columns with missing percentage above a threshold."""
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.drop_cols_ = []

    def fit(self, X: pd.DataFrame, y=None):
        # Calculate missing percentage per column
        missing_frac = X.isnull().mean()
        self.drop_cols_ = missing_frac[missing_frac > self.threshold].index.tolist()
        print(f"\n🗑️ HighMissingDropper: Will drop {len(self.drop_cols_)} columns > {self.threshold*100}% missing.")
        self.feature_names_in_ = np.array(X.columns.tolist())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Drop identified columns
        return X.drop(columns=self.drop_cols_, errors='ignore')

    def get_feature_names_out(self, input_features=None):
        if input_features is None: input_features = self.feature_names_in_
        return np.array([f for f in input_features if f not in self.drop_cols_])


# --- 4.3 Correlated Feature Aggregator ---
class CorrelatedFeatureAggregator(BaseEstimator, TransformerMixin):
    """
    Automatically groups correlated features (> threshold) using a graph-based approach,
    replaces them with their mean, and drops the originals.
    """
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.groups_ = {} # Maps new_name -> [list_of_cols]
        self.drop_cols_ = []

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.array(X.columns.tolist())
        
        # Only consider numeric columns for correlation
        num_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(num_cols) < 2:
            return self

        # 1. Compute Correlation Matrix
        corr_matrix = X[num_cols].corr().abs()
        
        # 2. Find Connected Components (Groups)
        # We treat features as nodes and high correlation as edges.
        processed = set()
        group_id = 1
        
        for col in num_cols:
            if col in processed:
                continue
            
            # Find all features connected to 'col' (including itself)
            group = [col]
            stack = [col]
            processed.add(col)
            
            while stack:
                current = stack.pop()
                # Get neighbors with corr > threshold
                neighbors = corr_matrix[current][corr_matrix[current] > self.threshold].index.tolist()
                for neighbor in neighbors:
                    if neighbor not in processed:
                        processed.add(neighbor)
                        stack.append(neighbor)
                        group.append(neighbor)
            
            # If group has more than 1 feature, save it
            if len(group) > 1:
                new_name = f"agg_corr_group_{group_id}"
                self.groups_[new_name] = group
                self.drop_cols_.extend(group)
                group_id += 1
                
        print(f"🔗 CorrelatedFeatureAggregator: Found {len(self.groups_)} groups to aggregate (Threshold: {self.threshold}).")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        
        # Create aggregated mean columns
        for new_col, components in self.groups_.items():
            # Compute mean row-wise
            X_out[new_col] = X_out[components].mean(axis=1)
            
        # Drop original columns
        return X_out.drop(columns=self.drop_cols_, errors='ignore')

    def get_feature_names_out(self, input_features=None):
        if input_features is None: input_features = self.feature_names_in_
        # Remove dropped, add new
        kept = [f for f in input_features if f not in self.drop_cols_]
        new = list(self.groups_.keys())
        return np.array(kept + new)


# --- 4.4 Outlier Handler ---
class OutlierHandler(BaseEstimator, TransformerMixin):
    """Clips outliers using IQR method."""
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
        return self.feature_names_in_

class SmartColumnDropper(BaseEstimator, TransformerMixin):
    """
    A transformer that safely drops or keeps columns based on a strategy.
    
    Parameters:
    -----------
    columns : List[str]
        The list of columns to act upon.
    strategy : 'drop' or 'keep', default='drop'
        'drop': Removes the specified columns from the dataframe.
        'keep': Keeps ONLY the specified columns, dropping everything else.
    """
    def __init__(self, columns: List[str], strategy: Literal['drop', 'keep'] = 'drop'):
        self.columns = columns
        self.strategy = strategy
        self.feature_names_in_ = None
        self.final_columns_ = None

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.array(X.columns.tolist())
        
        if self.strategy == 'drop':
            # Check for columns requested to drop that aren't there
            missing_cols = [c for c in self.columns if c not in X.columns]
            if missing_cols:
                warnings.warn(
                    f"⚠️ SmartColumnDropper (drop): Columns not found to drop: {missing_cols}"
                )
            self.final_columns_ = [c for c in X.columns if c not in self.columns]
            
        elif self.strategy == 'keep':
            # Check for columns requested to keep that aren't there
            missing_cols = [c for c in self.columns if c not in X.columns]
            if missing_cols:
                warnings.warn(
                    f"⚠️ SmartColumnDropper (keep): Columns requested to keep but missing from input: {missing_cols}"
                )
            # We can only keep what actually exists
            self.final_columns_ = [c for c in self.columns if c in X.columns]
            
        else:
            raise ValueError("Strategy must be either 'drop' or 'keep'.")
            
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Use the columns identified during fit
        return X[self.final_columns_].copy()

    def get_feature_names_out(self, input_features=None):
        """
        Returns the names of the features that remain after the transformation.
        """
        return np.array(self.final_columns_)
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.validation import check_is_fitted
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.validation import check_is_fitted

class MultiColumnCountVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, drop_original=True, **vectorizer_params):
        self.columns = columns
        self.drop_original = drop_original
        self.vectorizer_params = vectorizer_params
        # State variables (should be reset in fit)
        self.vectorizers_ = {}
        self.feature_names_out_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        self.n_features_in_ = X.shape[1]
        self.columns_to_process = self.columns if self.columns is not None else X.columns
        
        # FIX: Reset these every time fit is called to avoid appending to old runs
        self.vectorizers_ = {} 
        new_feature_names = []
        
        # Track which columns remain if we don't drop
        current_cols = [c for c in X.columns if c not in self.columns_to_process] if self.drop_original else list(X.columns)

        for col in self.columns_to_process:
            vec = CountVectorizer(**self.vectorizer_params)
            vec.fit(X[col].astype(str).fillna(''))
            self.vectorizers_[col] = vec
            
            names = [f"{col}_{name}" for name in vec.get_feature_names_out()]
            new_feature_names.extend(names)
            
        self.feature_names_out_ = np.array(current_cols + new_feature_names)
        return self

    def transform(self, X):
        check_is_fitted(self)
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        
        all_results = []
        
        # Keep columns that weren't vectorized
        if self.drop_original:
            all_results.append(X_df.drop(columns=self.columns_to_process))
        else:
            all_results.append(X_df)

        for col, vec in self.vectorizers_.items():
            transformed = vec.transform(X_df[col].astype(str).fillna(''))
            names = [f"{col}_{name}" for name in vec.get_feature_names_out()]
            
            # Use sparse matrices where possible, or converted to dense for DF
            feature_df = pd.DataFrame(
                transformed.toarray(), 
                index=X_df.index, 
                columns=names
            )
            all_results.append(feature_df)
            
        # Optimization: One single concat is MUCH faster than concat in a loop
        return pd.concat(all_results, axis=1)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return self.feature_names_out_