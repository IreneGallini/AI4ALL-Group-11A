"""
Electricity demand forecasting with Random Forest — multi-horizon (1 day / 1 week / 1 month ahead)

Approach: "direct" multi-horizon forecasting. For each horizon h (in hours), we train a
separate Random Forest that predicts demand at time T using ONLY information that would
actually be available at forecast time (T - h). This avoids leakage: for a 30-day-ahead
forecast, the model never sees demand data from the last 30 days before T.

Feature groups per horizon h:
  - Lagged demand / rolling stats, all computed as of the forecast origin (T - h), then
    shifted forward by h so they line up with the target row.
  - Calendar features of the TARGET time T (hour, day-of-week, month, weekend, US holiday,
    cyclical encodings) — these are fully known in advance, no forecast needed.
  - Weather at the TARGET time T (temperature, humidity, wind, precipitation). In this
    dataset these are historical actuals. In production these must be replaced with a
    weather FORECAST for T (see README for caveats on forecast horizon limits).
  - Region, one-hot encoded (single pooled model covers all 8 regions).
"""

import json
import pandas as pd
import numpy as np
import holidays
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR/"data"/"processed"/"eia_with_features.csv"
MODEL_DIR = BASE_DIR/"models"
REPORTS_DIR = BASE_DIR/"reports"
METRICS_FILE = MODEL_DIR/"rf_demand_metrics.json"
MODEL_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

HORIZONS = {24: "1_day", 168: "1_week", 720: "1_month"}
TEST_DAYS = 45  # holdout: last 45 days of the dataset, by target timestamp

us_holidays = holidays.US(years=[2025, 2026])


def add_calendar_features(df):
    df = df.copy()
    df["is_holiday"] = df["datetime_utc"].dt.date.isin(us_holidays).astype(int)
    doy = df["datetime_utc"].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def load_and_clean(path):
    df = pd.read_csv(path, parse_dates=["datetime_utc"])
    df = df.sort_values(["region", "datetime_utc"]).reset_index(drop=True)

    cleaned = []
    weather_cols = ["temperature_F", "apparent_temp_F", "humidity_pct",
                     "precipitation_mm", "wind_speed_kmh"]
    gen_cols = ["solar_gen_mwh", "wind_gen_mwh"]
    for region, g in df.groupby("region", observed=True):
        g = g.sort_values("datetime_utc").copy()
        # interior gaps -> linear interpolation; trailing gap (end of series) -> forward fill
        g[weather_cols] = g[weather_cols].interpolate(limit_direction="both")
        g[gen_cols] = g[gen_cols].interpolate(limit_direction="both")
        cleaned.append(g)
    return pd.concat(cleaned, ignore_index=True)


def build_horizon_dataset(df, h):
    """Build feature/target rows for a single forecast horizon h (hours)."""
    rows = []
    for region, g in df.groupby("region", observed=True):
        g = g.sort_values("datetime_utc").reset_index(drop=True)
        demand = g["demand_mwh"]

        feat = pd.DataFrame(index=g.index)
        # information available at the forecast origin (T - h), shifted to align with target row T
        feat["origin_demand"] = demand.shift(h)
        feat["origin_demand_lag24"] = demand.shift(h + 24)
        feat["origin_demand_lag168"] = demand.shift(h + 168)
        feat["origin_roll24_mean"] = demand.shift(h).rolling(24).mean()
        feat["origin_roll24_std"] = demand.shift(h).rolling(24).std()
        feat["origin_roll168_mean"] = demand.shift(h).rolling(168).mean()

        # target-time info, fully known in advance
        for c in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
                  "is_weekend", "is_holiday", "month",
                  "temperature_F", "apparent_temp_F", "humidity_pct",
                  "precipitation_mm", "wind_speed_kmh"]:
            feat[c] = g[c]

        feat["region"] = region
        feat["target"] = demand
        feat["target_datetime"] = g["datetime_utc"]
        rows.append(feat)

    out = pd.concat(rows, ignore_index=True)
    out = out.dropna().reset_index(drop=True)
    return out


def evaluate(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true))) * 100
    r2 = r2_score(y_true, y_pred)
    print(f"{label:>12}  MAE={mae:8.1f} MWh  RMSE={rmse:8.1f} MWh  MAPE={mape:5.2f}%  R2={r2:.4f}")
    return {"mae": float(mae), "rmse": float(rmse), "mape": mape, "r2": float(r2)}


def train_and_eval(df, h, label):
    print(f"\n=== Horizon: {label} ({h}h) ===")
    data = build_horizon_dataset(df, h)
    region_col = data["region"].copy()
    data = pd.get_dummies(data, columns=["region"], prefix="region")

    cutoff = data["target_datetime"].max() - pd.Timedelta(days=TEST_DAYS)
    train_mask = data["target_datetime"] <= cutoff
    test_mask = ~train_mask

    feature_cols = [c for c in data.columns if c not in ("target", "target_datetime")]
    X_train, y_train = data.loc[train_mask, feature_cols], data.loc[train_mask, "target"]
    X_test, y_test = data.loc[test_mask, feature_cols], data.loc[test_mask, "target"]

    print(f"Train rows: {len(X_train)}  Test rows: {len(X_test)}")

    model = RandomForestRegressor(
        n_estimators=150, max_depth=14, min_samples_leaf=10,
        n_jobs=1, random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("Test set performance:")
    overall = evaluate(y_test.to_numpy(), preds, "Overall")

    print("Per-region test performance:")
    per_region = {}
    region_str = region_col.loc[test_mask].astype(str).to_numpy()
    y_test_arr = y_test.to_numpy()
    for region in sorted(set(region_str)):
        mask = region_str == region
        per_region[region] = evaluate(y_test_arr[mask], preds[mask], region)

    joblib.dump({"model": model, "feature_cols": feature_cols}, MODEL_DIR / f"rf_model_{label}.joblib")
    print(f"Saved model to {MODEL_DIR / f'rf_model_{label}.joblib'}")

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    importances.to_csv(REPORTS_DIR / f"rf_feature_importance_{label}.csv")

    test_out = data.loc[test_mask, ["target_datetime", "target"]].copy()
    test_out["prediction"] = preds
    test_out["region"] = region_col.loc[test_mask].values
    test_out.to_csv(REPORTS_DIR / f"rf_test_predictions_{label}.csv", index=False)

    return {"overall": overall, "per_region": per_region}


def main():
    df = load_and_clean(DATA_PATH)
    df = add_calendar_features(df)

    metrics = {}
    for h, label in HORIZONS.items():
        metrics[label] = train_and_eval(df, h, label)

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {METRICS_FILE}")

    summary = pd.DataFrame([
        {"horizon": label, **m["overall"]} for label, m in metrics.items()
    ])
    summary.to_csv(REPORTS_DIR / "rf_metrics_summary.csv", index=False)
    print("\n=== Summary ===")
    print(summary)


if __name__ == "__main__":
    main()
