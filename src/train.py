import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .data import load_raw_data
from .evaluate import compute_metrics
from .features import build_feature_dataset, time_series_train_val_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def create_baseline_forecasts(df: pd.DataFrame, target_col: str = "orders") -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)
    df["naive_forecast"] = df[target_col].shift(1)
    df["seasonal_naive_forecast"] = df[target_col].shift(7)
    return df

def train_sarimax(train_series: pd.Series, val_series: pd.Series):
    full_series = pd.concat([train_series, val_series])
    model = SARIMAX(full_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7), enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)
    return results

def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    return model

def prepare_ml_data(train_df, val_df, test_df, target_col="orders"):
    feature_cols = [c for c in train_df.columns if c not in ["date", target_col, "country"]]
    X_train, y_train = train_df[feature_cols].values, train_df[target_col].values
    X_val, y_val = val_df[feature_cols].values, val_df[target_col].values
    X_test, y_test = test_df[feature_cols].values, test_df[target_col].values
    return feature_cols, X_train, y_train, X_val, y_val, X_test, y_test

def main() -> None:
    raw_df = load_raw_data()
    raw_df = raw_df.sort_values("date").reset_index(drop=True)
    baseline_df = create_baseline_forecasts(raw_df.copy()).dropna().reset_index(drop=True)
    total_days = len(baseline_df); test_days = 60; val_days = 60
    test_start_idx = total_days - test_days; val_start_idx = test_start_idx - val_days
    baseline_test = baseline_df.iloc[test_start_idx:].copy().dropna(subset=["seasonal_naive_forecast"])
    baseline_metrics = compute_metrics(baseline_test["orders"].values, baseline_test["seasonal_naive_forecast"].values)
    print("Baseline (seasonal naive) metrics on test set:"); [print(f"  {k}: {v:.3f}") for k, v in baseline_metrics.items()]
    feature_df = build_feature_dataset(raw_df)
    train_df, val_df, test_df = time_series_train_val_test_split(feature_df, test_days=test_days, val_days=val_days)
    train_series, val_series = train_df["orders"], val_df["orders"]
    sarimax_results = train_sarimax(train_series, val_series)
    n_test = len(test_df)
    sarimax_forecast = sarimax_results.forecast(steps=n_test)
    sarimax_metrics = compute_metrics(test_df["orders"].values, sarimax_forecast.values)
    print("\nSARIMAX metrics on test set:"); [print(f"  {k}: {v:.3f}") for k, v in sarimax_metrics.items()]
    feature_cols, X_train, y_train, X_val, y_val, X_test, y_test = prepare_ml_data(train_df, val_df, test_df)
    gb_model = train_gradient_boosting(X_train, y_train)
    y_pred_test = gb_model.predict(X_test)
    gb_metrics = compute_metrics(y_test, y_pred_test)
    print("\nGradient Boosting metrics on test set:"); [print(f"  {k}: {v:.3f}") for k, v in gb_metrics.items()]
    joblib.dump(sarimax_results, os.path.join(MODELS_DIR, "sarimax_model.pkl"))
    joblib.dump({"model": gb_model, "feature_cols": feature_cols}, os.path.join(MODELS_DIR, "gb_model.joblib"))

if __name__ == "__main__":
    main()
