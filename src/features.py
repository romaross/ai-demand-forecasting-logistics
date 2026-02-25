import os
from typing import List, Tuple
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    return df

def add_lag_features(df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"orders_lag_{lag}"] = df["orders"].shift(lag)
    return df

def add_rolling_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    df = df.copy()
    for window in windows:
        df[f"orders_roll_mean_{window}"] = (
            df["orders"].shift(1).rolling(window=window, min_periods=1).mean()
        )
    return df

def build_feature_dataset(df: pd.DataFrame, lags: List[int] = None, windows: List[int] = None) -> pd.DataFrame:
    if lags is None:
        lags = [1, 7, 14]
    if windows is None:
        windows = [7, 14, 28]
    df = df.sort_values("date").reset_index(drop=True)
    df = add_calendar_features(df)
    df = add_lag_features(df, lags)
    df = add_rolling_features(df, windows)
    max_lag = max(max(lags), max(windows))
    df = df.iloc[max_lag:].reset_index(drop=True)
    return df

def time_series_train_val_test_split(df: pd.DataFrame, test_days: int = 60, val_days: int = 60, date_col: str = "date") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(date_col).reset_index(drop=True)
    total_days = len(df)
    test_start_idx = total_days - test_days
    val_start_idx = test_start_idx - val_days
    train_df = df.iloc[:val_start_idx]
    val_df = df.iloc[val_start_idx:test_start_idx]
    test_df = df.iloc[test_start_idx:]
    return train_df, val_df, test_df
