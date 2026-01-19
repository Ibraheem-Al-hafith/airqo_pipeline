# AirQo PM2.5 Prediction Pipeline 🌍

A production-ready machine learning pipeline for forecasting Particulate Matter (PM2.5) air quality. This project transitions experimental notebooks into a modular, reproducible MLOps architecture using Scikit-Learn pipelines and MLflow.

## 🏗 Architecture

The project follows a component-based architecture:
* **`src/data`**: Ingestion, cleaning, and schema validation.
* **`src/features`**: Custom Scikit-learn transformers (`TimeFeatureExtractor`, `OutlierHandler`).
* **`src/models`**: Training workflows, cross-validation, and hyperparameter tracking.
* **`src/inference`**: Automated prediction generation compatible with submission formats.

## ⚙️ Setup

1.  **Clone the repository:**
    ```bash
    git clone <repo_url>
    cd airqo_ml_pipeline
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Data Placement:**
    Place `Train.csv` and `Test.csv` inside `data/raw/`.

## 🚀 Usage

### 1. Exploratory Data Analysis (EDA)
EDA is decoupled from production logic. Run the notebook found in `notebooks/01_eda_and_prototyping.ipynb`.
* Plots are automatically saved to `reports/figures/`.

### 2. Training the Model
To train the model, evaluate performance, and log artifacts to MLflow:
```bash
python main.py --mode train

```

* **Artifacts:** Saved to `models/artifacts/final_pipeline.pkl`
* **Logs:** Saved to `models/mlruns/`

### 3. Inference (Prediction)

To generate predictions on the test set (or a custom file):

```bash
python main.py --mode predict
# OR
python main.py --mode predict --input data/raw/NewData.csv

```

* **Output:** Generated at `data/outputs/submission.csv` containing columns `(id, pm2_5)`.

## 🔧 Configuration

Modify `src/config.py` to adjust:

* Hyperparameters
* Paths
* Outlier thresholds
* Feature selection

## 📊 MLOps Features

* **Pipelines:** Preprocessing and modeling are encapsulated in a single serializable object.
* **Experiment Tracking:** MLflow logs params, metrics (RMSE, MAE, R2), and models.
* **Type Hinting:** Full Python typing for robustness.

```

```