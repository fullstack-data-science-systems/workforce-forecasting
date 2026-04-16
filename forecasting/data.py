"""Data loading and preprocessing utilities."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_data(filepath: str) -> pd.DataFrame:
    """Load the employment data from CSV."""
    df = pd.read_csv(filepath)
    print("=== Original Dataset Shape ===")
    print(f"Shape: {df.shape}")
    print("First 10 rows:")
    print(df.head(10))
    return df


def explore_dataset(df_original: pd.DataFrame) -> None:
    """Explore the dataset structure."""
    print("=== Dataset Information ===")
    print(df_original.info())
    print("\n=== Statistical Summary ===")
    print(df_original.describe())
    print("\n=== Unique Values ===")
    print(f"Variables: {df_original['variable'].unique()}")
    print(f"Sex Categories: {df_original['sex'].unique()}")
    print(f"Provinces: {df_original.columns[3:].tolist()}")
    print(f"Date Range: {df_original['month'].min()} to {df_original['month'].max()}")
    print(f"Total Records: {len(df_original)}")


def pivot_employment_data(df: pd.DataFrame) -> pd.DataFrame:
    """Transform employment data from long to wide format."""
    df_pivot_list = []
    provinces = df.columns[3:].tolist()

    df_ft = df[df["variable"] == "Full-time employment"].copy()
    df_pt = df[df["variable"].str.strip() == "Part-time employment"].copy()

    for province in provinces:
        ft_f = df_ft[df_ft["sex"] == "Females"][["month", province]].copy()
        ft_f.columns = ["month", f"{province}_FT_F"]

        ft_m = df_ft[df_ft["sex"] == "Males"][["month", province]].copy()
        ft_m.columns = ["month", f"{province}_FT_M"]

        pt_f = df_pt[df_pt["sex"] == "Females"][["month", province]].copy()
        pt_f.columns = ["month", f"{province}_PT_F"]

        pt_m = df_pt[df_pt["sex"] == "Males"][["month", province]].copy()
        pt_m.columns = ["month", f"{province}_PT_M"]

        df_pivot_list.extend([ft_f, ft_m, pt_f, pt_m])

    df_result = df_pivot_list[0]
    for df_temp in df_pivot_list[1:]:
        df_result = df_result.merge(df_temp, on="month", how="outer")

    return df_result.sort_values("month").reset_index(drop=True)


def handle_missing_values(df: pd.DataFrame):
    """Handle missing values using forward and backward fill."""
    print("=== Handling Missing Values ===")
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

    df_clean[numeric_cols] = df_clean[numeric_cols].ffill().bfill()

    remaining_missing = df_clean[numeric_cols].isnull().sum().sum()
    print(f"Remaining missing values: {remaining_missing}")
    return df_clean, numeric_cols


def scale_features(features: np.ndarray):
    """Scale features using MinMaxScaler."""
    print("=== Feature Scaling ===")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)

    print(f"Scaled data shape: {scaled_features.shape}")
    print(f"Scaled data range: [{scaled_features.min():.4f}, {scaled_features.max():.4f}]")
    return scaled_features, scaler


def create_sequences(data: np.ndarray, n_steps: int):
    """Create sequences for time series prediction."""
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i - n_steps : i])
        y.append(data[i])
    return np.array(X), np.array(y)


def split_data(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.70, val_ratio: float = 0.15):
    """Split data into train, validation, and test sets."""
    print("=== Data Split ===")

    train_size = int(len(X) * train_ratio)
    val_size = int(len(X) * val_ratio)
    test_size = len(X) - train_size - val_size

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_val = X[train_size : train_size + val_size]
    y_val = y[train_size : train_size + val_size]

    X_test = X[train_size + val_size :]
    y_test = y[train_size + val_size :]

    print(f"Training set: {X_train.shape[0]} samples ({train_size/len(X)*100:.1f}%)")
    print(f"Validation set: {X_val.shape[0]} samples ({val_size/len(X)*100:.1f}%)")
    print(f"Test set: {X_test.shape[0]} samples ({test_size/len(X)*100:.1f}%)")

    return X_train, y_train, X_val, y_val, X_test, y_test
