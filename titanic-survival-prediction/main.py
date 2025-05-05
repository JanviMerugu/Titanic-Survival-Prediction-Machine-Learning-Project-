# main.py

import os
from src.preprocessing import load_and_preprocess_data
from src.train_model import train_model
from src.evaluation import evaluate_model

def main():
    # Define paths
    input_path = os.path.join("data", "titanic.csv")
    cleaned_path = os.path.join("data", "titanic_clean.csv")
    model_path = os.path.join("src", "logistic_model.pkl")
    scaler_path = os.path.join("src", "scaler.pkl")

    # Step 1: Preprocess data
    load_and_preprocess_data(input_path, cleaned_path)

    # Step 2: Train model
    train_model(cleaned_path, model_path, scaler_path)

    # Step 3: Evaluate model
    evaluate_model(cleaned_path, model_path, scaler_path)

if __name__ == "__main__":
    main()
