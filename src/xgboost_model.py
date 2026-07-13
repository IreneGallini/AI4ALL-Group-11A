"""
Trains an XGBoost regression model to forecast hourly electricity demand
from calendar, weather, and lagged-demand features.

Usage:
    python src/xgboost_model.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).parent.parent
INPUT_FILE = ROOT / "data" / "processed" / "eia_with_features.csv"
MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODEL_FILE = MODEL_DIR / "xgb_demand_model.json"
METRICS_FILE = MODEL_DIR / "xgb_demand_metrics.json"

TARGET = "demand_mwh"
FEATURES = [
    "hour", "day_of_week", "month", "is_weekend",
    "temperature_F", "apparent_temp_F", "humidity_pct",
    "precipitation_mm", "wind_speed_kmh",
    "solar_gen_mwh", "wind_gen_mwh",
    "demand_lag_1h", "demand_lag_24h",
    "region",
]

TEST_DAYS = 60  # most recent window held out to simulate forecasting the future
VAL_DAYS = 30   # earlier window carved out of training data, used for early stopping
PLOT_REGION = "ERCO"

# dataviz reference palette: categorical slot 1 (blue) vs slot 8 (orange)
COLOR_ACTUAL = "#2a78d6"
COLOR_PREDICTED = "#eb6834"
COLOR_IMPORTANCE = "#2a78d6"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_FILE, parse_dates=["datetime_utc"])
    df = df.sort_values(["region", "datetime_utc"]).reset_index(drop=True)
    df["region"] = df["region"].astype("category")

    before = len(df)
    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    print(f"Dropped {before - len(df)} rows with missing values ({before} -> {len(df)})")
    return df


def time_split(df: pd.DataFrame):
    """Chronological train/val/test split so the model never trains on the future."""
    max_date = df["datetime_utc"].max()
    test_cutoff = max_date - pd.Timedelta(days=TEST_DAYS)
    val_cutoff = test_cutoff - pd.Timedelta(days=VAL_DAYS)

    train = df[df["datetime_utc"] < val_cutoff]
    val = df[(df["datetime_utc"] >= val_cutoff) & (df["datetime_utc"] < test_cutoff)]
    test = df[df["datetime_utc"] >= test_cutoff]

    print(f"Train: {train['datetime_utc'].min()} -> {train['datetime_utc'].max()} ({len(train)} rows)")
    print(f"Val:   {val['datetime_utc'].min()} -> {val['datetime_utc'].max()} ({len(val)} rows)")
    print(f"Test:  {test['datetime_utc'].min()} -> {test['datetime_utc'].max()} ({len(test)} rows)")
    return train, val, test


def evaluate(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true))) * 100
    r2 = r2_score(y_true, y_pred)
    print(f"{label:>12}  MAE={mae:8.1f} MWh  RMSE={rmse:8.1f} MWh  MAPE={mape:5.2f}%  R2={r2:.4f}")
    return {"mae": float(mae), "rmse": float(rmse), "mape": mape, "r2": float(r2)}


def plot_feature_importance(model, features, out_path):
    importances = model.feature_importances_
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(features)[order], importances[order], color=COLOR_IMPORTANCE)
    ax.set_xlabel("Importance (gain)")
    ax.set_title("XGBoost Feature Importance — Demand Forecast")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_actual_vs_predicted(test, y_pred, region, out_path):
    mask = (test["region"].astype(str) == region).to_numpy()
    subset = test.loc[mask].copy()
    subset["predicted"] = y_pred[mask]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(subset["datetime_utc"], subset[TARGET], label="Actual", color=COLOR_ACTUAL, linewidth=2)
    ax.plot(subset["datetime_utc"], subset["predicted"], label="Predicted", color=COLOR_PREDICTED, linewidth=2)
    ax.set_title(f"Actual vs Predicted Demand — {region} (test period)")
    ax.set_ylabel("Demand (MWh)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    df = load_data()
    train, val, test = time_split(df)

    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        objective="reg:squarederror",
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=50,
        eval_metric="mae",
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining XGBoost model...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"Best iteration: {model.best_iteration} (of {model.n_estimators} max)")

    y_pred = model.predict(X_test)

    print("\nTest set performance:")
    overall = evaluate(y_test.to_numpy(), y_pred, "Overall")

    print("\nPer-region test performance:")
    per_region = {}
    region_str = test["region"].astype(str).to_numpy()
    y_test_arr = y_test.to_numpy()
    for region in sorted(set(region_str)):
        mask = region_str == region
        per_region[region] = evaluate(y_test_arr[mask], y_pred[mask], region)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model.save_model(str(MODEL_FILE))
    print(f"\nSaved model to {MODEL_FILE}")

    with open(METRICS_FILE, "w") as f:
        json.dump({"overall": overall, "per_region": per_region}, f, indent=2)
    print(f"Saved metrics to {METRICS_FILE}")

    plot_feature_importance(model, FEATURES, REPORTS_DIR / "xgb_feature_importance.png")
    plot_actual_vs_predicted(test, y_pred, PLOT_REGION, REPORTS_DIR / "xgb_actual_vs_predicted.png")


if __name__ == "__main__":
    main()
