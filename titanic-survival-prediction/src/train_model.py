# src/train_model.py

import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def train_model(data_path, model_path, scaler_path):
    # Load preprocessed data
    data = pd.read_csv(data_path)

    # Features and target
    X = data.drop('Survived', axis=1)
    y = data['Survived']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # Save model and scaler
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print("✅ Model and scaler saved successfully.")
    return X_test_scaled, y_test  # Return for evaluation

if __name__ == "__main__":
    data_file = os.path.join("data", "titanic_clean.csv")
    model_file = os.path.join("src", "logistic_model.pkl")
    scaler_file = os.path.join("src", "scaler.pkl")

    train_model(data_file, model_file, scaler_file)
