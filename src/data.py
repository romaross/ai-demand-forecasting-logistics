import os
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
os.makedirs(RAW_DATA_DIR, exist_ok=True)

@dataclass
class SyntheticConfig:
    start_date: str = "2022-01-01"
    end_date: str = "2024-12-31"
    base_demand: float = 800.0
    trend_per_day: float = 0.2
    weekly_seasonality_strength: float = 150.0
    holiday_uplift: float = 300.0
    promo_uplift: float = 250.0
    noise_std: float = 80.0
    promo_probability: float = 0.1

def get_polish_holidays(year: int) -> pd.DatetimeIndex:
    holidays_str = [
        f"{year}-01-01", f"{year}-01-06", f"{year}-05-01", f"{year}-05-03",
        f"{year}-08-15", f"{year}-11-01", f"{year}-11-11", f"{year}-12-25", f"{year}-12-26",
    ]
    return pd.to_datetime(holidays_str)

def generate_synthetic_demand(config: SyntheticConfig) -> pd.DataFrame:
    dates = pd.date_range(config.start_date, config.end_date, freq="D")
    n = len(dates)
    df = pd.DataFrame({"date": dates})
    days_since_start = np.arange(n)
    trend = config.trend_per_day * days_since_start
    base = config.base_demand + trend
    weekday = df["date"].dt.weekday
    weekly_seasonality = np.where(weekday < 5, 1.0, 0.7)
    weekly_component = (weekly_seasonality - 0.9) * config.weekly_seasonality_strength
    all_holidays = pd.DatetimeIndex([])
    for year in sorted(set(df["date"].dt.year)):
        all_holidays = all_holidays.append(get_polish_holidays(year))
    df["is_holiday"] = df["date"].isin(all_holidays).astype(int)
    rng = np.random.default_rng(seed=42)
    df["is_promotion"] = (rng.uniform(0, 1, size=n) < config.promo_probability).astype(int)
    demand = (
        base
        + weekly_component
        + df["is_holiday"] * config.holiday_uplift
        + df["is_promotion"] * config.promo_uplift
        + rng.normal(0, config.noise_std, size=n)
    )
    df["orders"] = np.clip(demand, a_min=0, a_max=None).round().astype(int)
    df["country"] = "PL"
    return df

def save_synthetic_data(df: pd.DataFrame, filename: str = "synthetic_demand.csv") -> str:
    path = os.path.join(RAW_DATA_DIR, filename)
    df.to_csv(path, index=False)
    return path

def load_raw_data(path: str = None) -> pd.DataFrame:
    if path is None:
        path = os.path.join(RAW_DATA_DIR, "synthetic_demand.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data file not found at {path}. Run data generation first.")
    df = pd.read_csv(path, parse_dates=["date"])
    return df

def generate_and_save_default() -> Tuple[pd.DataFrame, str]:
    config = SyntheticConfig()
    df = generate_synthetic_demand(config)
    path = save_synthetic_data(df)
    print(f"Synthetic data saved to: {path}")
    return df, path

if __name__ == "__main__":
    generate_and_save_default()
