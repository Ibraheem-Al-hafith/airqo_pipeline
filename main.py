import argparse
from src.data.ingest import get_train_val_split
from src.models.train import train_workflow
from src.inference.predict import make_predictions

def main():
    parser = argparse.ArgumentParser(description="AirQo MLOps Pipeline")
    parser.add_argument("--mode", choices=["train", "predict"], required=True, help="Pipeline mode")
    parser.add_argument("--input", type=str, help="Path to input CSV for prediction", default=None)
    
    args = parser.parse_args()
    
    if args.mode == "train":
        X_train, X_val, y_train, y_val = get_train_val_split()
        train_workflow(X_train, y_train, X_val, y_val)
        
    elif args.mode == "predict":
        make_predictions(args.input)

if __name__ == "__main__":
    main()