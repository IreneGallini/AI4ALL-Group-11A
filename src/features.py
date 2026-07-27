"""
Shared feature engineering for demand forecasting — used by train_demand_forecast.py
(Random Forest) and xgboost_model.py (XGBoost), and by app.py for reconstructing
the same feature vectors at inference time.

Multi-horizon "direct" forecasting: for each horizon h (in hours), a model predicts
demand at time T using ONLY information that would actually be available at forecast
time (T - h) — see build_horizon_dataset() for the feature groups.
"""

import numpy as np
import pandas as pd
import holidays

HORIZONS = {24: "1_day", 168: "1_week", 720: "1_month"}

CALENDAR_FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    "is_weekend", "is_holiday", "month",
]
WEATHER_FEATURES = [
    "temperature_F", "apparent_temp_F", "humidity_pct",
    "precipitation_mm", "wind_speed_kmh",
]
LAG_FEATURE_NAMES = [
    "origin_demand", "origin_demand_lag24", "origin_demand_lag168",
    "origin_roll24_mean", "origin_roll24_std", "origin_roll168_mean",
]

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
    gen_cols = ["solar_gen_mwh", "wind_gen_mwh"]
    for region, g in df.groupby("region", observed=True):
        g = g.sort_values("datetime_utc").copy()
        # interior gaps -> linear interpolation; trailing gap (end of series) -> forward fill
        g[WEATHER_FEATURES] = g[WEATHER_FEATURES].interpolate(limit_direction="both")
        g[gen_cols] = g[gen_cols].interpolate(limit_direction="both")
        cleaned.append(g)
    return pd.concat(cleaned, ignore_index=True)


def build_horizon_dataset(df, h):
    """Build feature/target rows for a single forecast horizon h (hours).

    `region` is left as a plain column — callers encode it however their
    model needs (one-hot for Random Forest, pandas categorical for XGBoost).
    """
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
        for c in CALENDAR_FEATURES + WEATHER_FEATURES:
            feat[c] = g[c]

        feat["region"] = region
        feat["target"] = demand
        feat["target_datetime"] = g["datetime_utc"]
        rows.append(feat)

    out = pd.concat(rows, ignore_index=True)
    out = out.dropna().reset_index(drop=True)
    return out
