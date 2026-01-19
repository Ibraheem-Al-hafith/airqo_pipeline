import joblib
import pandas as pd
from src.config import Config

def make_predictions(input_file: str = None) -> pd.DataFrame:
    """
    Loads model, predicts on input_file, returns formatted DataFrame (id, target).
    """
    file_path = input_file if input_file else Config.TEST_DATA_PATH
    print(f"🔍 Loading test data from: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Validation: Ensure ID column exists
    if Config.ID_COL not in df.columns:
        raise ValueError(f"Input file missing required column: {Config.ID_COL}")
    
    ids = df[Config.ID_COL]
    
    # Load Pipeline
    model_path = Config.MODEL_DIR / "final_pipeline.pkl"
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Run training first.")
        
    pipeline = joblib.load(model_path)
    
    # Predict
    print("⚡ Generating predictions...")
    preds = pipeline.predict(df)
    
    # Format Output
    results = pd.DataFrame({
        Config.ID_COL: ids,
        Config.TARGET: preds
    })
    
    output_path = Config.DATA_DIR / "outputs" / "submission.csv"
    results.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to {output_path}")
    
    return results