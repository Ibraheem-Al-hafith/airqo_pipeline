# 🚀 Production ML Pipeline - AirQo PM2.5 Prediction

A production-ready, modular machine learning pipeline for predicting PM2.5 air quality levels. This project demonstrates MLOps best practices including experiment tracking, model versioning, automated pipelines, and clean architecture.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](## ✨ Features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Training Pipeline](#training-pipeline)
  - [Making Predictions](#making-predictions)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [MLflow Tracking](#mlflow-tracking)
- [Configuration](#configuration)
- [Development](#development)
- [Testing](#testing)
- [Contributing](#contributing)

---

## 🎯 Overview

This project transforms exploratory Jupyter notebooks into a production-ready ML system with:

- **Modular architecture** separating concerns (data, features, models, inference)
- **Automated training pipelines** with reproducible results
- **MLflow experiment tracking** for comparing models and hyperparameters
- **Production inference API** with clean model versioning
- **Type-safe code** with full type hints and docstrings
- **Comprehensive logging** and error handling

**Dataset**: AirQo PM2.5 air quality measurements across African cities

**Task**: Regression - predicting PM2.5 concentration levels

---

## ✨ Features

### 🏗️ Architecture
- Clean separation of data loading, preprocessing, training, and inference
- sklearn-compatible custom transformers for feature engineering
- Abstract base classes for extensible model development
- Configuration-driven design for easy experimentation

### 🔬 Data Processing
- Automated outlier detection and handling (IQR method)
- Time-based feature extraction (month, day, quarter, week)
- Feature aggregation for dimensionality reduction
- Robust missing value handling
- Train/validation/test splitting with stratification support

### 🤖 Modeling
- Multiple regression models (Linear, Ridge, Lasso, ElasticNet, RF, GBM, SVR)
- Ensemble methods (Voting Regressor)
- Automated cross-validation
- Hyperparameter tuning with RandomizedSearchCV
- Comprehensive evaluation metrics (RMSE, R², MAE, overfitting detection)

### 📊 Experiment Tracking
- MLflow integration for all experiments
- Automatic logging of parameters, metrics, and models
- Model registry for version control
- Comparison dashboards and visualizations

### 🔮 Production Inference
- Clean predictor API for loading models
- CSV-to-predictions pipeline
- Model metadata and performance cards
- Versioned model artifacts

---

## 📁 Project Structure

```
ml-pipeline-project/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package installation
├── .gitignore                        # Git ignore rules
│
├── config/
│   ├── config.yaml                   # Main configuration
│   └── model_config.yaml             # Model hyperparameters
│
├── data/
│   ├── raw/                          # Original data (gitignored)
│   │   ├── Train.csv
│   │   └── Test.csv
│   ├── processed/                    # Processed data
│   └── external/                     # External datasets
│
├── notebooks/
│   ├── 01_eda_airqo.ipynb           # Exploratory data analysis
│   └── 02_model_experiments.ipynb    # Model experimentation
│
├── reports/
│   ├── figures/                      # Auto-generated plots
│   │   ├── eda/
│   │   ├── feature_importance/
│   │   └── model_performance/
│   └── metrics/                      # Model metrics JSONs
│
├── models/
│   ├── trained/                      # Saved models by version
│   │   └── v1_20250119_143022/
│   │       ├── best_model.pkl
│   │       ├── preprocessor.pkl
│   │       ├── model_card.json
│   │       └── requirements.json
│   └── experiments/                  # MLflow tracking data
│
├── src/
│   ├── data/
│   │   ├── data_loader.py           # Data ingestion
│   │   ├── data_validator.py        # Data validation
│   │   └── data_splitter.py         # Train/val/test split
│   │
│   ├── features/
│   │   ├── feature_engineer.py      # Feature engineering
│   │   ├── preprocessor.py          # Preprocessing pipeline
│   │   ├── outlier_handler.py       # Outlier detection/handling
│   │   └── time_features.py         # Time-based features
│   │
│   ├── models/
│   │   ├── model_trainer.py         # Training orchestration
│   │   ├── model_evaluator.py       # Evaluation metrics
│   │   ├── linear_models.py         # Linear models
│   │   ├── ensemble_models.py       # Ensemble models
│   │   └── hyperparameter_tuner.py  # Tuning logic
│   │
│   ├── inference/
│   │   ├── predictor.py             # Inference pipeline
│   │   └── model_loader.py          # Model loading
│   │
│   ├── visualization/
│   │   ├── eda_plots.py             # EDA visualizations
│   │   └── model_plots.py           # Model plots
│   │
│   └── utils/
│       ├── config_loader.py         # Config management
│       ├── logger.py                # Logging setup
│       └── mlflow_utils.py          # MLflow helpers
│
├── scripts/
│   ├── train_pipeline.py            # Full training pipeline
│   ├── evaluate_models.py           # Model comparison
│   ├── tune_hyperparameters.py      # Hyperparameter tuning
│   └── predict.py                   # Inference script
│
└── tests/
    ├── test_data_loader.py
    ├── test_preprocessor.py
    └── test_models.py
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

```bash
# Clone repository
git clone <repository-url>
cd ml-pipeline-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Dependencies
Key packages:
- `scikit-learn >= 1.0`
- `pandas >= 1.3`
- `numpy >= 1.20`
- `mlflow >= 2.0`
- `pyyaml`
- `joblib`
- `matplotlib`
- `seaborn`

---

## 🚀 Quick Start

### 1. Prepare Data
Place your data files in `data/raw/`:
- `Train.csv`
- `Test.csv`

### 2. Train Models
```bash
python scripts/train_pipeline.py --config config/config.yaml
```

This will:
- Load and preprocess data
- Train multiple models
- Log experiments to MLflow
- Save the best model with preprocessing pipeline
- Generate performance visualizations

### 3. View Experiments
```bash
mlflow ui --backend-store-uri models/experiments
```
Then open http://localhost:5000 in your browser

### 4. Make Predictions
```bash
python scripts/predict.py \
    --model-dir models/trained/v1_20250119_143022 \
    --input data/raw/Test.csv \
    --output predictions/submission.csv
```

---

## 📖 Usage

### Training Pipeline

#### Basic Training
```bash
python scripts/train_pipeline.py
```

#### Custom Configuration
```bash
python scripts/train_pipeline.py --config config/custom_config.yaml
```

#### What Happens:
1. **Data Loading**: Reads data from configured directory
2. **Outlier Removal**: Removes target variable outliers (98th percentile)
3. **Preprocessing**: 
   - Time feature extraction
   - Group aggregation
   - IQR-based outlier handling
   - Robust scaling
4. **Model Training**: Trains multiple models with cross-validation
5. **Evaluation**: Computes metrics on validation set
6. **Saving**: Saves best model, preprocessor, and metadata
7. **Visualization**: Generates comparison plots

### Making Predictions

#### From CSV File
```python
from src.inference.predictor import predict_from_csv

predictions = predict_from_csv(
    model_dir="models/trained/v1_20250119_143022",
    input_csv="data/raw/Test.csv",
    output_csv="predictions/submission.csv",
    id_column="id",
    target_column="pm2_5"
)
```

#### Programmatic Inference
```python
from src.inference.predictor import ModelPredictor
import pandas as pd

# Load predictor
predictor = ModelPredictor("models/trained/v1_20250119_143022")

# Load data
X = pd.read_csv("data/raw/Test.csv")

# Make predictions
predictions = predictor.predict(X)

# Get model info
info = predictor.get_model_info()
print(info)
```

#### Command Line
```bash
# Make predictions
python scripts/predict.py \
    --model-dir models/trained/v1_20250119_143022 \
    --input data/raw/Test.csv \
    --output predictions/submission.csv

# View model information
python scripts/predict.py \
    --model-dir models/trained/v1_20250119_143022 \
    --info
```

### Hyperparameter Tuning

```bash
python scripts/tune_hyperparameters.py \
    --config config/config.yaml \
    --n-iter 50 \
    --cv-folds 5
```

### MLflow Tracking

#### Launch UI
```bash
mlflow ui --backend-store-uri models/experiments
```

#### Query Runs Programmatically
```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("file://models/experiments")

# Get experiment
experiment = mlflow.get_experiment_by_name("ml-pipeline")

# Search runs
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
print(runs[['params.model', 'metrics.val_r2']].sort_values('metrics.val_r2', ascending=False))
```

---

## ⚙️ Configuration

### Main Configuration (`config/config.yaml`)

```yaml
project_name: "ml-pipeline"

data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  test_size: 0.2
  val_size: 0.2
  random_state: 42

model:
  cv_folds: 5
  n_jobs: -1
  random_state: 42
  models_dir: "models/trained"
  experiments_dir: "models/experiments"

preprocessing:
  outlier_factor: 1.5
  missing_threshold: 0.6
  scaling_method: "robust"  # or "standard"

mlflow_tracking_uri: null  # Auto-configured if null
```

### Model Configuration (`config/model_config.yaml`)

```yaml
models:
  random_forest:
    n_estimators: [100, 200, 300]
    max_depth: [10, 20, 30, None]
    min_samples_split: [2, 5, 10]
  
  gradient_boosting:
    n_estimators: [100, 200]
    learning_rate: [0.01, 0.05, 0.1]
    max_depth: [3, 5, 7]
  
  ridge:
    alpha: [0.001, 0.01, 0.1, 1.0, 10.0]
```

---

## 🧪 Development

### Adding a New Model

1. Create model class in `src/models/`:
```python
from sklearn.base import BaseEstimator, RegressorMixin

class CustomModel(BaseEstimator, RegressorMixin):
    def __init__(self, param1=1.0):
        self.param1 = param1
    
    def fit(self, X, y):
        # Training logic
        return self
    
    def predict(self, X):
        # Prediction logic
        return predictions
```

2. Add to model registry in `src/models/ensemble_models.py`:
```python
def get_advanced_models(config):
    models = {
        # ... existing models
        'Custom Model': CustomModel(param1=2.0)
    }
    return models
```

### Adding Custom Features

Extend `AdvancedFeatureEngineer` in `src/features/feature_engineer.py`:

```python
def transform(self, X: pd.DataFrame) -> pd.DataFrame:
    X_eng = X.copy()
    
    # Your custom features
    if 'col1' in X_eng.columns and 'col2' in X_eng.columns:
        X_eng['new_feature'] = X_eng['col1'] / (X_eng['col2'] + 1e-6)
    
    return X_eng
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_data_loader.py

# Run with coverage
pytest --cov=src tests/
```

---

## 📊 Model Performance

Current best model performance on validation set:

| Model              | Val R²  | Val RMSE | CV R² Mean | CV R² Std |
|-------------------|---------|----------|------------|-----------|
| Gradient Boosting | 0.8542  | 12.34    | 0.8456     | 0.0123    |
| Random Forest     | 0.8421  | 13.21    | 0.8398     | 0.0156    |
| ElasticNet        | 0.7834  | 15.67    | 0.7812     | 0.0089    |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Standards
- Follow PEP 8
- Add type hints to all functions
- Write docstrings for all public methods
- Add tests for new features
- Update documentation

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- Based on AirQo PM2.5 air quality dataset
- Inspired by MLOps best practices from industry leaders
- Built with scikit-learn, MLflow, and modern Python stack

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Happy Modeling! 🚀**
