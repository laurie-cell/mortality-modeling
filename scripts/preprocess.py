import os
import pandas as pd
from sklearn.model_selection import train_test_split

from data.schema import CATEGORICAL_COLS, NUMERICAL_COLS, BINARY_COLS
from config import DATA_DIR, TEST_SIZE

def preprocess_data(csv_path=os.path.join(DATA_DIR, "dummy_data_no_times.csv")):
    df = pd.read_csv(csv_path)

    df = df.drop(columns="ID") # De-identify data

    # Handle binary columns
    for col in BINARY_COLS:
        df[col] = df[col].map({
            "Yes": 1, "Y": 1,
            "No": 0, "N": 0
        })

    # Handle missing values
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna(df[col].mode()[0])

    for col in NUMERICAL_COLS:
        df[col] = df[col].fillna(df[col].median())

    for col in BINARY_COLS:
        df[col] = df[col].fillna(0)

    # One-hot encoding for categorical variables
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    # Split data
    X = df_encoded.drop(columns="death_within_1_day_of_hospital_discharge")
    y = df_encoded["death_within_1_day_of_hospital_discharge"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test
