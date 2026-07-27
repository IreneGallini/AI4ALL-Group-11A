"""
Electricity demand forecasting with XGBoost — multi-horizon (1 day / 1 week / 1 month ahead)

Same "direct" multi-horizon approach as train_demand_forecast.py's Random Forest: for each
horizon h (in hours), a separate model predicts demand at time T using ONLY information that
would actually be available at forecast time (T - h). Feature engineering is ported from that
script so the two models are trained on the same inputs and are directly comparable — they
differ only in algorithm (XGBoost keeps `region` as a native categorical column instead of
one-hot encoding it).

Feature groups per horizon h:
  - Lagged demand / rolling stats, computed as of the forecast origin (T - h), shifted forward
    by h so they line up with the target row.
  - Calendar features of the TARGET time T (hour, day-of-week, month, weekend, US holiday,
    cyclical encodings) — fully known in advance, no forecast needed.
  - Weather at the TARGET time T (temperature, humidity, wind, precipitation). In this dataset
    these are historical actuals; in production these must be replaced with a weather FORECAST
    for T.
  - Region, as a pandas categorical column (XGBoost handles it natively).

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

from features import HORIZONS, add_calendar_features, load_and_clean, build_horizon_dataset

ROOT = Path(__file__).parent.parent
INPUT_FILE = ROOT / "data" / "processed" / "eia_with_features.csv"
MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
METRICS_FILE = MODEL_DIR / "xgb_demand_metrics.json"

TEST_DAYS = 45  # holdout: last 45 days of the dataset, by target timestamp (matches RF script)
VAL_DAYS = 30   # earlier window carved out of training data, used for early stopping

PLOT_REGION = "ERCO"

# dataviz reference palette: categorical slot 1 (blue) vs slot 8 (orange)
COLOR_ACTUAL = "#2a78d6"
COLOR_PREDICTED = "#eb6834"
COLOR_IMPORTANCE = "#2a78d6"


def evaluate(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true))) * 100
    r2 = r2_score(y_true, y_pred)
    print(f"{label:>12}  MAE={mae:8.1f} MWh  RMSE={rmse:8.1f} MWh  MAPE={mape:5.2f}%  R2={r2:.4f}")
    return {"mae": float(mae), "rmse": float(rmse), "mape": mape, "r2": float(r2)}


def plot_feature_importance(model, features, label, out_path):
    importances = model.feature_importances_
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(features)[order], importances[order], color=COLOR_IMPORTANCE)
    ax.set_xlabel("Importance (gain)")
    ax.set_title(f"XGBoost Feature Importance — Demand Forecast ({label})")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_actual_vs_predicted(test_out, region, label, out_path):
    subset = test_out[test_out["region"].astype(str) == region]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(subset["target_datetime"], subset["target"], label="Actual", color=COLOR_ACTUAL, linewidth=2)
    ax.plot(subset["target_datetime"], subset["predicted"], label="Predicted", color=COLOR_PREDICTED, linewidth=2)
    ax.set_title(f"Actual vs Predicted Demand — {region} ({label} horizon, test period)")
    ax.set_ylabel("Demand (MWh)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def train_and_eval(df, h, label):
    print(f"\n=== Horizon: {label} ({h}h) ===")
    data = build_horizon_dataset(df, h)
    data["region"] = data["region"].astype("category")
    feature_cols = [c for c in data.columns if c not in ("target", "target_datetime")]

    test_cutoff = data["target_datetime"].max() - pd.Timedelta(days=TEST_DAYS)
    val_cutoff = test_cutoff - pd.Timedelta(days=VAL_DAYS)

    train_mask = data["target_datetime"] < val_cutoff
    val_mask = (data["target_datetime"] >= val_cutoff) & (data["target_datetime"] < test_cutoff)
    test_mask = data["target_datetime"] >= test_cutoff

    X_train, y_train = data.loc[train_mask, feature_cols], data.loc[train_mask, "target"]
    X_val, y_val = data.loc[val_mask, feature_cols], data.loc[val_mask, "target"]
    X_test, y_test = data.loc[test_mask, feature_cols], data.loc[test_mask, "target"]

    print(f"Train: {len(X_train)} rows  Val: {len(X_val)} rows  Test: {len(X_test)} rows")

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

    print("Training XGBoost model...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"Best iteration: {model.best_iteration} (of {model.n_estimators} max)")

    y_pred = model.predict(X_test)

    print("Test set performance:")
    overall = evaluate(y_test.to_numpy(), y_pred, "Overall")

    print("Per-region test performance:")
    per_region = {}
    region_str = data.loc[test_mask, "region"].astype(str).to_numpy()
    y_test_arr = y_test.to_numpy()
    for region in sorted(set(region_str)):
        mask = region_str == region
        per_region[region] = evaluate(y_test_arr[mask], y_pred[mask], region)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_file = MODEL_DIR / f"xgb_model_{label}.json"
    model.save_model(str(model_file))
    print(f"Saved model to {model_file}")

    test_out = data.loc[test_mask, ["target_datetime", "target", "region"]].copy()
    test_out["predicted"] = y_pred
    test_out.to_csv(REPORTS_DIR / f"xgb_test_predictions_{label}.csv", index=False)

    plot_feature_importance(model, feature_cols, label, REPORTS_DIR / f"xgb_feature_importance_{label}.png")
    plot_actual_vs_predicted(test_out, PLOT_REGION, label, REPORTS_DIR / f"xgb_actual_vs_predicted_{label}.png")

    return {"overall": overall, "per_region": per_region}


def main():
    df = load_and_clean(INPUT_FILE)
    df = add_calendar_features(df)

    metrics = {}
    for h, label in HORIZONS.items():
        metrics[label] = train_and_eval(df, h, label)

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {METRICS_FILE}")


if __name__ == "__main__":
    main()
