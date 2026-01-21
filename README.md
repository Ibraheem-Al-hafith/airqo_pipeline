

# 🌱 AgriYield: Production-Grade Crop Yield Estimation

**A modular, robust MLOps pipeline designed to predict crop yields by analyzing agricultural practices, temporal planting data, and environmental factors.**

---

<div align="center">
<img src="assets/logo.png">

</div>

---

## 🎯 Project Overview

Accurate crop yield estimation is vital for food security, supply chain planning, and farmer economic stability. This project transitions from standard analysis to a **production-ready Machine Learning pipeline**. It ingests raw agricultural survey data, applies graph-based feature reduction, processes complex temporal cropping patterns, and deploys high-performance Gradient Boosting models to forecast yield.

### 🚀 Key Capabilities

* **🧠 Intelligent Feature Engineering**:
* **Graph-Based Correlation Removal**: Automatically builds feature graphs to detect and aggregate highly correlated numerical features, reducing multicollinearity.
* **Smart Column Dropping**: Dynamically filters columns based on high cardinality or missing data thresholds (>50%).
* **Multi-Column Vectorization**: Custom transformer to handle complex categorical descriptions (e.g., `LandPreparationMethod`, `CropEstMethod`) using count vectorization.


* **🏗️ Modular Architecture**: strict separation of concerns—Config, Data Ingestion, Feature Extraction, and Modeling are decoupled for scalability.
* **⏱️ Temporal Intelligence**: Specialized `TimeFeatureExtractor` that parses critical agricultural dates (`CropTillageDate`, `Harv_date`, etc.) into seasonal features.
* **🛡️ Robust Validation**: Implements K-Fold Cross-Validation and outlier handling (IQR Clipping) to ensure the model generalizes well to unseen data.
* **📊 Full Observability**: Integrated with **MLflow** to track every experiment, hyperparameter, and metric (RMSE, R², MAE).

---

## 🛠️ Tech Stack

| Domain | Technologies |
| --- | --- |
| **Core** | `Python 3.10+` |
| **Data Processing** | `Pandas`, `NumPy` |
| **Machine Learning** | `LightGBM` (Default), `XGBoost`, `CatBoost`, `Scikit-Learn` |
| **Orchestration** | `Scikit-Learn Pipelines` (Custom Transformers) |
| **Tracking & Ops** | `MLflow`, `Joblib` |
| **Dependency Mgmt** | `uv` (Fast Python package installer) |

---

## 📂 Repository Structure

```bash
AgriYield_Pipeline/
├── data/
│   ├── raw/                # Input CSVs (Train.csv, Test.csv)
│   └── outputs/            # Processed datasets
├── models/
│   ├── artifacts/          # Serialized .pkl pipelines (Production ready)
│   └── mlruns/             # MLflow local tracking database
├── reports/
│   └── figures/            # Automated Feature Importance & Regression Plots
├── src/
│   ├── config.py           # Central Control: Hyperparams, Column definitions
│   ├── data/               # Ingestion & Target Cleaning logic
│   ├── features/           # ⚡ The Engine: Custom Transformers (GraphCorr, TimeExtract)
│   └── models/             # Training loop & Evaluation logic
├── app.py                  # Streamlit Dashboard (Inference Interface)
├── main.py                 # CLI Entry point
└── README.md               # Project Documentation

```

---

## ⚙️ Installation & Setup

This project utilizes **`uv`** for lightning-fast dependency resolution, ensuring a reproducible environment.

1. **Clone the Repository**
```bash
git clone https://github.com/YourUsername/AgriYield_Pipeline.git
cd AgriYield_Pipeline

```


2. **Install Dependencies**
```bash
uv sync

```


3. **Activate Environment**
* *Linux/Mac:* `source .venv/bin/activate`
* *Windows:* `.venv\Scripts\activate`



---

## 🏃 Usage

### 1️⃣ Train the Model

Run the full training pipeline. This will load data, engineer features, train the LightGBM regressor, and log results.

```bash
python main.py --mode train

```

> **Output:** The trained model is saved to `models/artifacts/final_pipeline.pkl`. Performance metrics are logged to MLflow.

### 2️⃣ Predict on New Data

Generate yield predictions for a test dataset.

```bash
python main.py --mode predict --input data/raw/Test.csv

```

### 3️⃣ Interactive Dashboard

Launch the web interface for non-technical stakeholders to upload data and visualize predictions.

```bash
uv run python -m streamlit run app.py

```
---
### 📺 Demo Video 📺 
https://github.com/user-attachments/assets/c72174a2-800f-458c-9602-55dfaaf037df

---

## 🧠 The Pipeline Logic

The raw data undergoes a rigorous transformation process defined in `src/features/transformers.py`:

1. **Smart Filter**: Drops IDs and columns with >50% missing values.
2. **Correlation Aggregation**: Features with >0.85 correlation are grouped; their mean is kept, and originals dropped to reduce noise.
3. **Time Extraction**: Dates like `SeedingSowingTransplanting` are broken down into Month, Week, and Day to capture seasonal trends.
4. **Vectorization**: Text columns (e.g., `NursDetFactor`) are vectorized to capture categorical nuances.
5. **Robust Scaling**: Numeric data is scaled using RobustScaler to minimize the impact of remaining outliers.
6. **Modeling**: The processed data is fed into a **LightGBM Regressor** optimized for speed and accuracy.

---

## 📊 Performance & Visualization

The pipeline automatically generates reports in `reports/figures/`:

* **Feature Importance**: See exactly which agricultural factors (e.g., *Fertilizer usage*, *Tillage Date*) drive yield.
* **Regression Plots**: Visual comparison of Predicted vs. Actual Yield.

---

## 🤝 Contributing

We welcome contributions to improve agricultural forecasting!

1. Fork the repo.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes.
4. Open a Pull Request.

---

## 📄 License

Distributed under the MIT License.

**Built to empower farmers with data.** 🌾🚜
