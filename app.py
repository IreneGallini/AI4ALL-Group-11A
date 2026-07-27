"""
Streamlit app for the AI4ALL Group 11A electricity demand forecasts.

TODO — full app scope:
  - Model selector: Random Forest vs XGBoost
      models/rf_model_{1_day,1_week,1_month}.joblib
      models/xgb_model_{1_day,1_week,1_month}.json
  - Horizon selector: 1 day / 1 week / 1 month
  - Region selector: BPAT, CISO, ERCO, ISNE, MISO, NYIS, PJM, SWPP
  - Actual vs predicted chart for the test period.
      MVP path: reuse the precomputed reports/rf_test_predictions_{horizon}.csv.
      XGBoost doesn't save an equivalent predictions CSV yet (only PNG plots
      in reports/xgb_actual_vs_predicted_*.png) — add one in xgboost_model.py
      (mirror train_demand_forecast.py's test_out.to_csv(...)) if you want
      the XGB side of this chart to be interactive too.
  - Per region metrics table (MAE/RMSE/MAPE/R2) from
      models/rf_demand_metrics.json / models/xgb_demand_metrics.json
      (both already have "per_region" breakdowns, not just "overall").
  - Feature importance view from reports/rf_feature_importance_*.csv and
      reports/xgb_feature_importance_*.png.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(page_title="AI4ALL Group 11A — Demand Forecast", layout="wide")

st.title("Electricity Demand Forecasting")
st.markdown(
    "Forecasts hourly electricity demand for 8 major US grid regions "
    "(EIA data), comparing a Random Forest and an XGBoost model across "
    "1 day, 1 week, and 1 month horizons."
)


@st.cache_data
def load_metrics():
    with open(MODEL_DIR / "rf_demand_metrics.json") as f:
        rf_metrics = json.load(f)
    with open(MODEL_DIR / "xgb_demand_metrics.json") as f:
        xgb_metrics = json.load(f)
    return rf_metrics, xgb_metrics


rf_metrics, xgb_metrics = load_metrics()

st.header("Model comparison — overall test set performance")
rows = []
for horizon in rf_metrics:
    rf_overall = rf_metrics[horizon]["overall"]
    xgb_overall = xgb_metrics[horizon]["overall"]
    rows.append(
        {
            "horizon": horizon,
            "rf_mae": rf_overall["mae"],
            "rf_rmse": rf_overall["rmse"],
            "rf_mape": rf_overall["mape"],
            "rf_r2": rf_overall["r2"],
            "xgb_mae": xgb_overall["mae"],
            "xgb_rmse": xgb_overall["rmse"],
            "xgb_mape": xgb_overall["mape"],
            "xgb_r2": xgb_overall["r2"],
        }
    )

comparison = pd.DataFrame(rows).set_index("horizon")
st.dataframe(comparison, use_container_width=True)
