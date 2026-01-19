import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import mean_squared_error
import streamlit as st

def save_regression_plot(y_true, y_pred, saving_dir):
    """Plots y_true vs y_pred with a regression fit line and RMSE."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, color='blue', label='Predictions')
    
    # Calculate regression line fit (y = mx + c)
    m, c = np.polyfit(y_true, y_pred, 1)
    plt.plot(y_true, m*y_true + c, color='red', label=f'Fit Line (m={m:.2f})')
    
    plt.title(f'Actual vs Predicted (RMSE: {rmse:.4f})')
    plt.xlabel('Actual Values')
    plt.ylabel('Predictions')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save the plot
    save_path = os.path.join(saving_dir, "regression_fit.png")
    plt.savefig(save_path)
    plt.close() # Close to free up memory
    print(f"Regression plot saved to: {save_path}")

def save_feature_importance(pipeline, model_name, saving_dir):
    """Extracts and saves feature importance using automatic name discovery."""
    
    # 1. Get the names from the preprocessing steps
    # We slice [:-1] to get everything EXCEPT the regressor
    preprocessor_part = pipeline[:-1]
    feature_names = preprocessor_part.get_feature_names_out()
    
    # 2. Get the model (the last step)
    model = pipeline.steps[-1][1]
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # 3. Create DataFrame
        feat_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).head(10)

        # 4. Plotting
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feat_df, x='Importance', y='Feature', hue='Feature')
        plt.title(f'Top 10 Features - {model_name}')
        # plt.yticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(saving_dir, f"{model_name}_importance.png"))
        plt.close()
    else:
        st.warning(f"Model {model_name} does not support feature importances.")