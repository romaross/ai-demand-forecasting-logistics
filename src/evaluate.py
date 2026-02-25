import os
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true > 1.0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + eps))) * 100.0

def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denominator = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return np.mean(2.0 * np.abs(y_pred - y_true) / denominator) * 100.0

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {"RMSE": rmse(y_true, y_pred), "MAPE": mape(y_true, y_pred), "sMAPE": smape(y_true, y_pred)}

def plot_actual_vs_forecast(df: pd.DataFrame, date_col: str, actual_col: str, forecast_col: str, title: str, filename: str) -> str:
    plt.figure(figsize=(12, 5))
    plt.plot(df[date_col], df[actual_col], label="Actual", linewidth=2)
    plt.plot(df[date_col], df[forecast_col], label="Forecast", linewidth=2)
    plt.legend(); plt.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path); plt.close(); return path
