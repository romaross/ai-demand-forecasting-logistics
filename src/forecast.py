import os
from datetime import timedelta
import joblib
import numpy as np
import pandas as pd

from .data import load_raw_data
from .features import build_feature_dataset
from .evaluate import plot_actual_vs_forecast

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORECASTS_DIR = os.path.join(PROJECT_ROOT, "forecasts")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(FORECASTS_DIR, exist_ok=True)

FORECAST_HORIZON_DAYS = 28

def load_gb_model():
    path = os.path.join(MODELS_DIR, "gb_model.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}. Run training first.")
    obj = joblib.load(path)
    return obj["model"], obj["feature_cols"]

def generate_forecast_next_28_days() -> pd.DataFrame:
    raw_df = load_raw_data().sort_values("date").reset_index(drop=True)
    feature_df = build_feature_dataset(raw_df)
    last_feature_row = feature_df.iloc[-1].copy()
    model, feature_cols = load_gb_model()
    forecast_rows = []
    current_date = last_feature_row["date"]
    last_orders = last_feature_row["orders"]
    recent_orders = list(feature_df["orders"].values)
    for _ in range(FORECAST_HORIZON_DAYS):
        next_date = current_date + timedelta(days=1)
        tmp = {
            "date": next_date,
            "orders": last_orders,
            "is_holiday": 0,
            "is_promotion": 0,
            "country": "PL",
        }
        tmp["day_of_week"] = next_date.weekday()
        tmp["month"] = next_date.month
        tmp["day_of_year"] = next_date.timetuple().tm_yday
        for lag in [1, 7, 14]:
            tmp[f"orders_lag_{lag}"] = recent_orders[-lag] if len(recent_orders) >= lag else recent_orders[-1]
        for window in [7, 14, 28]:
            tmp[f"orders_roll_mean_{window}"] = np.mean(recent_orders[-window:]) if len(recent_orders) >= window else np.mean(recent_orders)
        row_df = pd.DataFrame([tmp])
        X = row_df[feature_cols].values
        y_pred = model.predict(X)[0]
        forecast_rows.append({"date": next_date, "forecast": y_pred})
        current_date = next_date
        last_orders = y_pred
        recent_orders.append(y_pred)
    return pd.DataFrame(forecast_rows)

def main() -> None:
    forecast_df = generate_forecast_next_28_days()
    path = os.path.join(FORECASTS_DIR, "forecast_next_28_days.csv")
    forecast_df.to_csv(path, index=False)
    print(f"Forecast saved to: {path}")
    raw_df = load_raw_data().sort_values("date").reset_index(drop=True)
    history_tail = raw_df.tail(60).copy()
    history_tail.rename(columns={"orders": "actual"}, inplace=True)
    history_tail = history_tail[["date", "actual"]]
    history_tail["forecast"] = np.nan
    forecast_plot_df = pd.concat([history_tail, forecast_df.rename(columns={"forecast": "forecast"})], ignore_index=True, sort=False)
    plot_actual_vs_forecast(forecast_plot_df, "date", "actual", "forecast", "History + 28-day Forecast", "history_plus_forecast_gb.png")

if __name__ == "__main__":
    main()
