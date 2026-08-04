import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "eia_with_features.csv"
MODEL_DIR = BASE_DIR / "models"
DEPLOYMENT_METRICS_FILE = MODEL_DIR / "deployment_metrics.json"

FEATURES = [
    "region",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "apparent_temp_F",
]

TARGET = "demand_mwh"


def evaluate(y_true, y_pred, name):
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    rmse = mean_squared_error(y_true_arr, y_pred_arr) ** 0.5

    # Exclude zero-demand rows (a handful of bad readings) from MAPE to
    # avoid division by zero
    nonzero = y_true_arr != 0
    mape = float(
        np.mean(np.abs((y_true_arr[nonzero] - y_pred_arr[nonzero]) / y_true_arr[nonzero]))
    ) * 100

    r2 = r2_score(y_true_arr, y_pred_arr)

    print(f"\n{name}")
    print(f"MAE: {mae:.2f} MWh")
    print(f"RMSE: {rmse:.2f} MWh")
    print(f"MAPE: {mape:.2f}%")
    print(f"R2: {r2:.4f}")

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": mape,
        "r2": float(r2),
    }


def evaluate_with_per_region(y_true, y_pred, regions, name):
    overall = evaluate(y_true, y_pred, name)

    y_true_arr = np.asarray(y_true)
    pred_arr = np.asarray(y_pred)
    regions_arr = np.asarray(regions)

    per_region = {}

    for region in sorted(set(regions_arr)):
        mask = regions_arr == region
        per_region[region] = evaluate(
            y_true_arr[mask], pred_arr[mask], f"{name} - {region}"
        )

    return {"overall": overall, "per_region": per_region}


def main():

    df = pd.read_csv(DATA_PATH, parse_dates=["datetime_utc"])

    df = df.sort_values("datetime_utc")

    # Preserve region label before one-hot encoding, for stratification
    # and per-region metrics
    region_labels = df["region"]

    # One hot encode region
    df = pd.get_dummies(
        df,
        columns=["region"],
        drop_first=False
    )

    region_columns = [
        col for col in df.columns
        if col.startswith("region_")
    ]

    features = [
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "apparent_temp_F",
    ] + region_columns

    df = df.dropna(
        subset=features + [TARGET]
    )

    region_labels = region_labels.loc[df.index]

    X = df[features]
    y = df[TARGET]

    # Random stratified split (by region) rather than chronological, since
    # these features have no lag/sequential dependency
    X_train, X_test, y_train, y_test, region_train, region_test = train_test_split(
        X,
        y,
        region_labels,
        test_size=0.2,
        stratify=region_labels,
        random_state=42,
    )

    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)

    rf_metrics = evaluate_with_per_region(
        y_test,
        rf_pred,
        region_test,
        "Random Forest"
    )

    joblib.dump(
        {
            "model": rf,
            "features": features
        },
        MODEL_DIR / "deployment_rf_model.joblib"
    )

    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    xgb_model.fit(
        X_train,
        y_train
    )

    xgb_pred = xgb_model.predict(X_test)

    xgb_metrics = evaluate_with_per_region(
        y_test,
        xgb_pred,
        region_test,
        "XGBoost"
    )

    xgb_model.save_model(
        MODEL_DIR / "deployment_xgb_model.json"
    )

    with open(MODEL_DIR / "deployment_features.json", "w") as f:
        json.dump(
            {
                "features": features
            },
            f,
            indent=2
        )

    deployment_metrics = {
        "Random Forest": rf_metrics,
        "XGBoost": xgb_metrics,
    }

    with open(DEPLOYMENT_METRICS_FILE, "w") as f:
        json.dump(deployment_metrics, f, indent=2)

    print(f"Saved metrics to {DEPLOYMENT_METRICS_FILE}")

    print("\nDeployment models saved!")


if __name__ == "__main__":
    main()