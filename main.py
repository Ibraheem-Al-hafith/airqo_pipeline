import argparse
from src.data.ingest import get_X_y_folds
from src.models.train import train_workflow
from src.inference.predict import make_predictions

def main():
    parser = argparse.ArgumentParser(description="AirQo MLOps Pipeline")
    parser.add_argument("--mode", choices=["train", "predict"], required=True, help="Pipeline mode")
    parser.add_argument("--input", type=str, help="Path to input CSV for prediction", default=None)
    
    args = parser.parse_args()
    
    if args.mode == "train":
        X, y, folds = get_X_y_folds()
        
        train_workflow(X, y, folds)
        
    elif args.mode == "predict":
        make_predictions(args.input)

if __name__ == "__main__":
    main()