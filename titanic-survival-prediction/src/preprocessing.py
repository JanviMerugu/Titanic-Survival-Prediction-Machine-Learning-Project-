# src/preprocessing.py

import pandas as pd
import os

def load_and_preprocess_data(input_path, output_path):
    # Load dataset
    data = pd.read_csv(input_path)

    # Drop irrelevant columns
    data.drop(['PassengerId'], axis=1, inplace=True)

    # Fill missing numeric values with median
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())

    # Fill missing categorical values with mode
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

    # Encode categorical variables using one-hot encoding
    data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

    # Drop less useful text columns
    data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    # Final safety check: remove any remaining rows with NaNs
    data.dropna(inplace=True)

    # Confirm no missing values remain
    print("✅ Missing values after cleaning:\n", data.isnull().sum())

    # Save cleaned data
    data.to_csv(output_path, index=False)
    print(f"✅ Preprocessed data saved to: {output_path}")

if __name__ == "__main__":
    input_file = os.path.join("data", "titanic.csv")
    output_file = os.path.join("data", "titanic_clean.csv")
    load_and_preprocess_data(input_file, output_file)
