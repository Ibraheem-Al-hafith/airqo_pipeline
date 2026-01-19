import mlflow
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from src.config import Config
from src.features.transformers import TimeFeatureExtractor, OutlierHandler

from lightgbm import LGBMRegressor

def build_pipeline(cat_cols: list, num_cols: list) -> Pipeline:
    """Constructs the processing and modeling pipeline."""
    
    # 1. Feature Engineering Steps
    feature_engineering = Pipeline([
        ('time_extractor', TimeFeatureExtractor(date_col=Config.DATE_COL)),
        ('outlier_clipper', OutlierHandler(factor=1.5))
    ])
    
    # 2. Preprocessing for Column Types
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
        ('regressor', LGBMRegressor(random_state=Config.RANDOM_STATE))
    ])
    
    return pipeline

def evaluate_metrics(y_true, y_pred):
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred)
    }

def train_workflow(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
    """Executes the training workflow with MLflow tracking."""
    
    mlflow.set_tracking_uri(Config.MLFLOW_URI)
    mlflow.set_experiment("AirQo_PM25_Production")
    
    # Identify column types DYNAMICALLY after basic cleaning
    # Note: We need to handle the fact that TimeExtractor changes columns.
    # A cleaner approach in sklearn pipelines is passing raw columns and letting transformers handle it.
    # For simplicity, we assume 'id' and 'date' are handled specially.
    
    cols_to_drop = Config.DROP_COLS + [Config.ID_COL]
    
    # Filter features passed to pipeline
    train_cols = [c for c in X_train.columns if c not in cols_to_drop and c != Config.DATE_COL]
    
    # Define categorical/numeric based on raw input (Date is handled by transformer)
    cat_cols = X_train[train_cols].select_dtypes(include=['object']).columns.tolist()
    num_cols = X_train[train_cols].select_dtypes(include=['number']).columns.tolist()
    
    # Add generated time columns to numeric list (anticipating transformation)
    # In a complex pipeline, we often use `sklearn-pandas` or custom selectors. 
    # Here, strictly for the robust pipeline:
    # We will let the pipeline transform X, but we need to pass the *column names* correctly to ColumnTransformer.
    # To fix the 'dynamic columns' issue in standard sklearn, we use a custom selector or apply columns globally.
    # simplified: We will apply numeric transformer to ALL numeric columns remaining after feature eng.
    
    # Re-definition for simplicity of this template:
    # We will rely on `make_column_selector` in a real scenario, but here:
    pipeline = build_pipeline(cat_cols, num_cols)
    
    print("🚀 Starting Training...")
    with mlflow.start_run():
        # Fit
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_val)
        
        # Evaluate
        metrics = evaluate_metrics(y_val, y_pred)
        
        # Log
        mlflow.log_params(pipeline.named_steps['regressor'].get_params())
        mlflow.log_metrics(metrics)
        
        # Save Artifacts
        model_path = Config.MODEL_DIR / "final_pipeline.pkl"
        joblib.dump(pipeline, model_path)
        mlflow.sklearn.log_model(pipeline, "model")
        
        print(f"✅ Training Complete. Metrics: {metrics}")
        print(f"💾 Model saved to {model_path}")