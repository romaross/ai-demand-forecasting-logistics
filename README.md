# AI Demand Forecasting for Logistics 🚚📈

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This project implements a realistic, business-oriented workflow for forecasting daily **shipment demand** and connecting model accuracy to **operational impact** in logistics.

It is designed as a portfolio-ready project for:

- Data Science / Analytics roles in logistics, retail, and e‑commerce  
- Supply Chain / Operations Analytics roles  
- Cloud / ML / Analytics Engineering roles (e.g., Amazon, Google, Microsoft)

---

## 1. Problem Overview

Goal: forecast **daily shipment volume** for the next **28 days** for a logistics operation (parcel network, e‑commerce fulfillment, or linehaul planning).

Key characteristics:

- Daily data (approximately 3 years of history)
- Clear weekly seasonality (weekday vs weekend)
- Trend (business growth)
- Promotion and holiday effects
- Realistic noise and variability

We frame this as a **time-series forecasting** and **supervised regression** problem with a **time-aware train/validation/test split**.

The single target variable throughout the project is:

- `shipments` – daily shipment volume

---

## 2. Why Forecasting Matters in Logistics

Better shipment forecasts directly improve network performance and cost.

### 2.1 Staffing

- Inbound and outbound docks: number of people per shift  
- Sorting centers: number of active lines and shifts  
- Customer service: expected ticket volume and coverage  

Under-forecast → overtime, burnout, missed SLAs  
Over-forecast → idle labor and higher unit costs

### 2.2 Warehouse Capacity and Slotting

- Storage planning: pallets, bins, and floor space  
- Slotting: which routes/customers/SKUs get priority locations  
- Putaway and picking waves: timing and sequence  

Poor forecasts create either stockouts and chaos, or excess inventory and tied-up working capital.

### 2.3 Linehaul and Transport Planning

- Linehaul scheduling: number of trucks and departures  
- Route consolidation vs direct shipments  
- Carrier mix: primary vs backup carriers  

Better forecasts allow:

- Earlier booking and better rates  
- Higher utilization of own fleet  
- Fewer last-minute expedites

### 2.4 Cost-to-Serve

Forecast quality flows directly into **cost-to-serve** via:

- Labor cost (overtime, temporary staff)  
- Transport cost (expedites, underutilized capacity)  
- Penalties and SLA breaches  
- Lost sales due to stockouts  

The project includes an example of how a modest improvement in forecast accuracy can translate into meaningful cost savings.

---

## 3. Project Structure

```text
ai-demand-forecasting-logistics/
├── README.md
├── requirements.txt
├── Makefile
├── .gitignore
├── data/
│   ├── raw/
│   │   └── synthetic_shipments.csv
│   └── processed/
├── notebooks/
│   └── 01_eda_and_baseline.ipynb
└── src/
    ├── __init__.py
    ├── data.py
    ├── features.py
    ├── evaluate.py
    ├── train.py
    └── forecast.py
```

- `README.md` – problem description, business context, usage  
- `data/` – raw and processed datasets (contents are gitignored)  
- `notebooks/` – EDA and baseline modelling  
- `src/` – reusable Python package code  
- `requirements.txt` – Python dependencies  
- `Makefile` – shortcuts for main commands  

---

## 4. Data

### 4.1 Synthetic Dataset

To keep the project reproducible on any laptop, the default dataset is **synthetic**, but structured to resemble real logistics data.

Baseline columns:

- `date` – calendar date  
- `shipments` – daily shipment volume (target)  
- `is_promotion` – promotion flag (0/1)  
- `is_holiday` – holiday flag (0/1, based on a simplified holiday calendar)  

You can generate the synthetic data with:

```bash
make data
# or
python -m src.data
```

Later you can replace the synthetic data with a real dataset (e.g., retail transactions, order history) by changing the loader in `src/data.py` to read from your own CSV and keep the rest of the pipeline unchanged.

---

## 5. Features and Models

### 5.1 Feature Engineering

From the raw daily shipment series, the pipeline builds the following features:

- **Calendar features**
  - `day_of_week`
  - `month`
  - `day_of_year`
  - `is_holiday`
  - `is_promotion`

- **Lag features** (target: `shipments`)
  - `shipments_lag_1`
  - `shipments_lag_7`
  - `shipments_lag_14`

- **Rolling-window statistics**
  - `shipments_roll_mean_7`
  - `shipments_roll_mean_14`
  - `shipments_roll_mean_28`

Implementation: `src/features.py`.

### 5.2 Models

All models use a **time-based** train/validation/test split to avoid data leakage.

1. **Baselines**

   - **Naive baseline**  
     Forecast for today = shipments from yesterday (`shipments_lag_1`).

   - **Seasonal naive baseline**  
     Forecast for today = shipments from 7 days ago (`shipments_lag_7`).

   These baselines show what "no model" performance looks like and set a benchmark.

2. **Classical time-series model**

   - **SARIMAX** (`statsmodels`)  
   - Captures trend and weekly seasonality on the shipment series.

3. **Machine learning model**

   - **GradientBoostingRegressor** (`scikit-learn`)  
   - Uses lag, rolling, and calendar features as inputs.

Training, evaluation, and model saving are implemented in `src/train.py`.

---

## 6. Metrics and Evaluation

The evaluation focuses on metrics that are common in forecasting-heavy organizations:

- **RMSE** (Root Mean Squared Error)  
  - Scale: shipments per day  
  - Penalizes larger errors more strongly.

- **MAPE** (Mean Absolute Percentage Error)  
  - Scale: percentage error  
  - Not robust when actual values are zero or very small.  
  - Implementation excludes days with `shipments <= 1` to avoid extreme values.

- **sMAPE** (Symmetric Mean Absolute Percentage Error)  
  - Uses the sum of absolute actual and forecast values in the denominator.  
  - More stable than MAPE when values are small or zero.

Metric implementations and plotting utilities live in `src/evaluate.py`.

Generated plots include:

- Actual vs forecast over time  
- Residuals over time  
- Feature importance (for the Gradient Boosting model)  

Plots are saved in `plots/`.

---

## 7. End-to-End Workflow

### 7.1 Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 7.2 Generate Data

```bash
make data
# or
python -m src.data
```

This step generates synthetic shipment data and saves it under `data/raw/`.

### 7.3 Train Models

```bash
make train
# or
python -m src.train
```

This step:

- Builds features  
- Splits data into train / validation / test by date  
- Trains baseline, SARIMAX, and Gradient Boosting models  
- Prints metrics for each model  
- Saves model artifacts to `models/`  
- Saves plots to `plots/`  

### 7.4 Generate 28‑Day Forecast

```bash
make forecast
# or
python -m src.forecast
```

This script:

- Loads the trained Gradient Boosting model  
- Iteratively generates a 28‑day daily shipment forecast  
- Saves the forecast to `forecasts/forecast_next_28_days.csv`  
- Produces a plot combining the last 60 historical days with the 28‑day forecast  

### 7.5 Notebook

```bash
jupyter notebook notebooks/01_eda_and_baseline.ipynb
```

The notebook demonstrates:

- Basic EDA on the synthetic shipment data  
- Weekly patterns and seasonality  
- Naive and seasonal naive baselines  

---

## 8. Example Results

Indicative (not fixed) results on the test set:

- **Seasonal naive baseline**  
  - RMSE: around 155 shipments/day  
  - MAPE: around 10%  
  - sMAPE: around 10%

- **SARIMAX**  
  - Typically improves RMSE and percentage errors relative to the baseline in the default configuration.

- **Gradient Boosting**  
  - Performance depends on configuration; the code is structured so hyperparameters can be tuned.

Example plot files:

- `plots/actual_vs_forecast_gb.png`  
- `plots/residuals_gb.png`  
- `plots/feature_importance_gb.png`  

These plots can be reused in a portfolio or interview slide deck.

---

## 9. Business Impact Illustration

A simple example highlights the operational value of improved forecast accuracy.

- Average daily shipment volume: 1,000  
- Legacy model MAPE: 15%  
- New model MAPE: 9%  
- Planning horizon: 30 days  

Assume that days with more than 20% under-forecast result in stockouts or expedited shipments, with an estimated cost of €2,000 per day (extra freight, SLA penalties, or lost margin).

If the improved model reduces the number of such days from 8 to 3 per month:

- Monthly savings: (8 − 3) × €2,000 = **€10,000**  
- Annual savings on a single lane or distribution center: **€120,000**  

Scaling similar improvements across multiple warehouses, transportation lanes, and customer segments can have a direct and material impact on cost-to-serve, working capital, and service levels.

---

## 10. Next Steps Toward Production

Typical steps to transition this pipeline into a production-grade system:

### 10.1 API / Service Layer

- Package the trained model as a REST API (e.g., FastAPI).  
- Expose an endpoint such as `POST /forecast` that receives recent shipment history and returns a forward forecast.

### 10.2 Scheduled Retraining

- Implement daily or weekly retraining pipelines (Airflow, Cloud Composer, Prefect).  
- Automate data ingestion, feature generation, model training, and metric logging.

### 10.3 Monitoring and Drift Detection

- Continuously track RMSE, MAPE, and sMAPE over time.  
- Monitor shifts in input distributions (promotion frequency, channel mix, seasonality).  
- Trigger alerts when performance degrades or drift is detected.

### 10.4 Model Governance

- Version models and configurations (e.g., MLflow).  
- Keep calendars, forecast horizons, and aggregation levels configurable.  
- Support rollback and A/B comparison between model versions.

### 10.5 Scaling Across the Network

- Move from a single aggregate forecast to:  
  - per-warehouse forecasts  
  - per-route or per-customer forecasts  
- Consider hierarchical forecasting and reconciliation methods to ensure consistency across levels.

---

## 11. Use as a Portfolio Project

This repository is designed to be easy to present in a technical or business interview:

- Strong business framing and logistics context  
- Clean and modular Python code in `src/`  
- Correct handling of time-series specifics (time-based splits, baselines, leakage prevention)  
- Comparison of baseline, classical, and ML models  
- Clear translation from model accuracy to operational and financial impact  

The project is intentionally lightweight so it can run on a standard laptop in under 10 minutes, while still demonstrating realistic, production-oriented thinking relevant for large logistics organizations and Big Tech environments.
