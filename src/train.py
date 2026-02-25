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
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def create_baseline_forecasts(df: pd.DataFrame, target_col: str = "orders") -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)
    df["naive_forecast"] = df[target_col].shift(1)
    df["seasonal_naive_forecast"] = df[target_col].shift(7)
    return df


def train_sarimax(train_series: pd.Series, val_series: pd.Series):
    full_series = pd.concat([train_series, val_series])
    model = SARIMAX(
        full_series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = model.fit(disp=False)
    return results


def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def prepare_ml_data(train_df, val_df, test_df, target_col="orders"):
    feature_cols = [c for c in train_df.columns if c not in ["date", target_col, "country"]]
    X_train, y_train = train_df[feature_cols].values, train_df[target_col].values
    X_val, y_val = val_df[feature_cols].values, val_df[target_col].values
    X_test, y_test = test_df[feature_cols].values, test_df[target_col].values
    return feature_cols, X_train, y_train, X_val, y_val, X_test, y_test


def _plot_actual_vs_forecast_gb(test_df: pd.DataFrame, y_pred_test: np.ndarray) -> None:
    """Plot actual vs predicted shipments for the test period and save to disk."""
    import matplotlib.pyplot as plt

    df_plot = test_df.copy()
    df_plot["gb_forecast"] = y_pred_test

    plt.figure(figsize=(12, 5))
    plt.plot(df_plot["date"], df_plot["orders"], label="Actual", linewidth=2)
    plt.plot(df_plot["date"], df_plot["gb_forecast"], label="Gradient Boosting forecast", linewidth=2)
    plt.xlabel("Date")
    plt.ylabel("Orders")
    plt.title("Actual vs Forecast (Gradient Boosting, Test Set)")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "actual_vs_forecast_gb.png")
    plt.savefig(path)
    plt.close()


def _plot_residuals_gb(test_df: pd.DataFrame, y_pred_test: np.ndarray) -> None:
    """Plot residuals (actual - forecast) over time for the test period."""
    import matplotlib.pyplot as plt

    df_plot = test_df.copy()
    df_plot["residual_gb"] = df_plot["orders"] - y_pred_test

    plt.figure(figsize=(12, 4))
    plt.plot(df_plot["date"], df_plot["residual_gb"], label="Residuals")
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Date")
    plt.ylabel("Residual (actual - forecast)")
    plt.title("Residuals (Gradient Boosting, Test Set)")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "residuals_gb.png")
    plt.savefig(path)
    plt.close()


def _plot_feature_importance_gb(feature_cols, gb_model: GradientBoostingRegressor) -> None:
    """Plot feature importance for the Gradient Boosting model."""
    import matplotlib.pyplot as plt
    import numpy as np

    importances = gb_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = np.array(feature_cols)[indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features, sorted_importances)
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title("Feature Importance (Gradient Boosting)")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "feature_importance_gb.png")
    plt.savefig(path)
    plt.close()


def main() -> None:
    # 1. Load and sort raw data
    raw_df = load_raw_data()
    raw_df = raw_df.sort_values("date").reset_index(drop=True)

    # 2. Baseline (seasonal naive)
    baseline_df = create_baseline_forecasts(raw_df.copy()).dropna().reset_index(drop=True)
    total_days = len(baseline_df)
    test_days = 60
    val_days = 60
    test_start_idx = total_days - test_days
    val_start_idx = test_start_idx - val_days

    baseline_test = baseline_df.iloc[test_start_idx:].copy().dropna(subset=["seasonal_naive_forecast"])
    baseline_metrics = compute_metrics(
        baseline_test["orders"].values,
        baseline_test["seasonal_naive_forecast"].values,
    )
    print("Baseline (seasonal naive) metrics on test set:")
    for k, v in baseline_metrics.items():
        print(f"  {k}: {v:.3f}")

    # 3. Build features and time-based split
    feature_df = build_feature_dataset(raw_df)
    train_df, val_df, test_df = time_series_train_val_test_split(
        feature_df, test_days=test_days, val_days=val_days
    )

    # 4. SARIMAX
    train_series, val_series = train_df["orders"], val_df["orders"]
    sarimax_results = train_sarimax(train_series, val_series)
    n_test = len(test_df)
    sarimax_forecast = sarimax_results.forecast(steps=n_test)
    sarimax_metrics = compute_metrics(test_df["orders"].values, sarimax_forecast.values)
    print("\nSARIMAX metrics on test set:")
    for k, v in sarimax_metrics.items():
        print(f"  {k}: {v:.3f}")

    # 5. Gradient Boosting
    feature_cols, X_train, y_train, X_val, y_val, X_test, y_test = prepare_ml_data(
        train_df, val_df, test_df
    )
    gb_model = train_gradient_boosting(X_train, y_train)
    y_pred_test = gb_model.predict(X_test)
    gb_metrics = compute_metrics(y_test, y_pred_test)
    print("\nGradient Boosting metrics on test set:")
    for k, v in gb_metrics.items():
        print(f"  {k}: {v:.3f}")

    # 6. Save models
    joblib.dump(sarimax_results, os.path.join(MODELS_DIR, "sarimax_model.pkl"))
    joblib.dump({"model": gb_model, "feature_cols": feature_cols}, os.path.join(MODELS_DIR, "gb_model.joblib"))

    # 7. Visualizations for Gradient Boosting
    _plot_actual_vs_forecast_gb(test_df, y_pred_test)
    _plot_residuals_gb(test_df, y_pred_test)
    _plot_feature_importance_gb(feature_cols, gb_model)


if __name__ == "__main__":
    main()
