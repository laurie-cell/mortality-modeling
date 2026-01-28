import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from data.schema import CATEGORICAL_COLS, NUMERICAL_COLS, BINARY_COLS, TIMESTAMP_COLS
from config import DATA_DIR, TEST_SIZE, DATA_FILE

def parse_datetime_column(series, has_time=True):
    """Parse datetime column handling different formats"""
    if has_time:
        # Try format M/D/YY H:M first
        parsed = pd.to_datetime(series, format='%m/%d/%y %H:%M', errors='coerce')
        # If that fails, try other common formats
        if parsed.isna().any():
            parsed = pd.to_datetime(series, errors='coerce')
    else:
        # For death_date which is M/D/YYYY format
        parsed = pd.to_datetime(series, format='%m/%d/%Y', errors='coerce')
    return parsed

def extract_datetime_features(df):
    """Extract meaningful features from datetime columns"""
    # Parse datetime columns
    df['hospital_admission_dt'] = parse_datetime_column(df['hospital_admission'], has_time=True)
    df['hospital_discharge_dt'] = parse_datetime_column(df['hospital_discharge'], has_time=True)

    # Handle unit_admission (note: column name has a space before underscore in CSV)
    unit_admission_col = [col for col in df.columns if 'unit' in col.lower() and 'admission' in col.lower()][0]
    unit_discharge_col = [col for col in df.columns if 'unit' in col.lower() and 'discharge' in col.lower()][0]

    df['unit_admission_dt'] = parse_datetime_column(df[unit_admission_col], has_time=True)
    df['unit_discharge_dt'] = parse_datetime_column(df[unit_discharge_col], has_time=True)
    df['death_date_dt'] = parse_datetime_column(df['death_date'], has_time=False)

    # Extract features from hospital admission
    df['admission_hour'] = df['hospital_admission_dt'].dt.hour
    df['admission_day_of_week'] = df['hospital_admission_dt'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['admission_month'] = df['hospital_admission_dt'].dt.month

    # Extract features from hospital discharge
    df['discharge_hour'] = df['hospital_discharge_dt'].dt.hour
    df['discharge_day_of_week'] = df['hospital_discharge_dt'].dt.dayofweek
    df['discharge_month'] = df['hospital_discharge_dt'].dt.month

    # Calculate time differences (in hours)
    df['hospital_stay_hours'] = (df['hospital_discharge_dt'] - df['hospital_admission_dt']).dt.total_seconds() / 3600
    df['unit_stay_hours'] = (df['unit_discharge_dt'] - df['unit_admission_dt']).dt.total_seconds() / 3600

    # Time to death (if death_date exists)
    df['hours_to_death'] = (df['death_date_dt'] - df['hospital_discharge_dt']).dt.total_seconds() / 3600
    df['has_death_date'] = df['death_date_dt'].notna().astype(int)

    # Drop original datetime columns and intermediate parsed columns
    cols_to_drop = ['hospital_admission', 'hospital_discharge', unit_admission_col, unit_discharge_col,
                   'death_date', 'hospital_admission_dt', 'hospital_discharge_dt',
                   'unit_admission_dt', 'unit_discharge_dt', 'death_date_dt']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    return df

def preprocess_data(csv_path=None):
    if csv_path is None:
        csv_path = DATA_FILE

    df = pd.read_csv(csv_path)

    df = df.drop(columns="ID") # De-identify data

    # Extract datetime features before other preprocessing
    df = extract_datetime_features(df)

    # Handle binary columns
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map({
                "Yes": 1, "Y": 1,
                "No": 0, "N": 0
            })

    # Handle missing values
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else "Unknown")

    for col in NUMERICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Handle missing values for new datetime-derived features
    datetime_features = ['admission_hour', 'admission_day_of_week', 'admission_month',
                        'discharge_hour', 'discharge_day_of_week', 'discharge_month',
                        'hospital_stay_hours', 'unit_stay_hours', 'hours_to_death']
    for col in datetime_features:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # One-hot encoding for categorical variables
    categorical_cols_to_encode = [col for col in CATEGORICAL_COLS if col in df.columns]
    df_encoded = pd.get_dummies(df, columns=categorical_cols_to_encode, drop_first=True)

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
