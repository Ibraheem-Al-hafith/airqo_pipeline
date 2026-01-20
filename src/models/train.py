import mlflow
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from src.config import Config
from src.features.transformers import TimeFeatureExtractor, OutlierHandler,SmartColumnDropper,CorrelatedFeatureAggregator,HighMissingDropper
from src.utils.plots import save_regression_plot, save_feature_importance


def evaluate_metrics(y_true, y_pred):
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred)
    }



def build_pipeline(cat_cols: list, num_cols: list) -> Pipeline:
    """Constructs the full processing and modeling pipeline."""
    
    # Model Registry
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    from catboost import CatBoostRegressor
    from sklearn.tree import DecisionTreeRegressor

    MODELS = {
        "lgbm": LGBMRegressor(verbose = -1, random_state=Config.RANDOM_STATE),
        "decision_tree": DecisionTreeRegressor(random_state=Config.RANDOM_STATE),
        "random_forest": RandomForestRegressor(random_state=Config.RANDOM_STATE),
        "xgboost": XGBRegressor(random_state = Config.RANDOM_STATE),
        "catboost": CatBoostRegressor(random_state=Config.RANDOM_STATE, verbose=0)
    }

    # 1. Feature Engineering (Applied sequentially)
    # Note: These transformers handle dataframe input/output
    feature_engineering = Pipeline([
        ("column_filterer", SmartColumnDropper(Config.DROP_COLS)),
        ('corr_aggregator', CorrelatedFeatureAggregator(threshold=Config.CORRELATION_THRESH)),
        ('missing_dropper', HighMissingDropper(threshold=Config.MISSING_THRESH)),
        ('time_extractor', TimeFeatureExtractor(date_col=Config.DATE_COL, features_to_extract=Config.TIME_FEATURES)),
        ('outlier_clipper', OutlierHandler(factor=1.5)),
    ])
    
    # 2. Column Preprocessing (Handling missing values & scaling)
    # NOTE: Since feature_engineering changes columns dynamically, we cannot rely on 
    # fixed lists 'num_cols' passed at the start. 
    # We must use 'make_column_selector' to select columns dynamically after feature engineering.
    from sklearn.compose import make_column_selector

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, make_column_selector(dtype_include=np.number)),
        ('cat', categorical_transformer, make_column_selector(dtype_include=object))
    ])
    
    # 3. Final Assembly
    pipeline = Pipeline([
        ('feature_eng', feature_engineering),
        ('preprocessor', preprocessor),
        ('regressor', MODELS[Config.MODEL_TYPE])
    ])
    
    return pipeline


def train_workflow(X: pd.DataFrame, y: pd.Series, folds: dict):
    """Executes the training workflow with MLflow tracking."""
    
    mlflow.set_tracking_uri(Config.MLFLOW_URI)
    mlflow.set_experiment(Config.EXPERIMENT_NAME)
    
    # Identify column types DYNAMICALLY after basic cleaning
    # Note: We need to handle the fact that TimeExtractor changes columns.
    # A cleaner approach in sklearn pipelines is passing raw columns and letting transformers handle it.
    # For simplicity, we assume 'id' and 'date' are handled specially.
    
    cols_to_drop = Config.DROP_COLS + [Config.ID_COL]
    
    # Filter features passed to pipeline
    train_cols = [c for c in X.columns if c not in cols_to_drop and c != Config.DATE_COL]
    
    # Define categorical/numeric based on raw input (Date is handled by transformer)
    cat_cols = X[train_cols].select_dtypes(include=['object']).columns.tolist()
    num_cols = X[train_cols].select_dtypes(include=['number']).columns.tolist()
    
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
    with mlflow.start_run(run_name=Config.EXPERIMENT_NAME):
        
        # Evaluate
        metrics, preds = run_cross_validation(pipeline, X, y, folds)

        # Log
        mlflow.log_params(pipeline.named_steps['regressor'].get_params())
        mlflow.log_metrics(metrics)
        
        #train the pipeline on the full dataset:
        pipeline.fit(X, y)

        # saving plots:
        save_feature_importance(pipeline, Config.MODEL_TYPE,Config.FIGURES_DIR)
        save_regression_plot(y, preds, Config.FIGURES_DIR)

        # Save Artifacts
        model_path = Config.MODEL_DIR / "final_pipeline.pkl"
        joblib.dump(pipeline, model_path)
        mlflow.sklearn.log_model(pipeline, Config.MODEL_TYPE)
        
        print(f"✅ Training Complete. Metrics: {metrics}")
        print(f"💾 Model saved to {model_path}")


def run_cross_validation(pipeline, X, y, folds):
    all_metrics = []
    val_weights = []
    
    # FIX: Initialize with zeros
    preds = np.zeros(len(y))
    preds_counts = np.zeros(len(y)) 

    for fold_id, (train_idx, val_idx) in folds.items():
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Fit and predict
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        
        # Calculate metrics for this specific fold
        fold_metrics = evaluate_metrics(y_true=y_val, y_pred=y_pred)
        all_metrics.append(fold_metrics)
        val_weights.append(len(val_idx))
        
        # Accumulate predictions and increment counts
        preds[val_idx] += y_pred
        preds_counts[val_idx] += 1

    # Weighted average of metrics across folds
    avg_metrics = np.average(pd.DataFrame(all_metrics), axis=0, weights=val_weights)
    metrics_dict = {key: avg_metrics[i] for i, key in enumerate(all_metrics[0].keys())}

    # FIX: Avoid division by zero for samples never seen in validation
    # If a sample was never in validation, counts will be 0. 
    # We use np.where to keep them as 0 instead of NaN.
    final_preds = np.divide(preds, preds_counts, out=np.zeros_like(preds), where=preds_counts != 0)
    
    return metrics_dict, final_preds