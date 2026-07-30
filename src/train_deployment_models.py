import json
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "eia_with_features.csv"
MODEL_DIR = BASE_DIR / "models"


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
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name}")
    print(f"MAE: {mae:.2f} MWh")
    print(f"RMSE: {rmse:.2f} MWh")
    print(f"R2: {r2:.4f}")


def main():

    df = pd.read_csv(DATA_PATH, parse_dates=["datetime_utc"])

    df = df.sort_values("datetime_utc")

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


    # Last 20% is future test data
    split = int(len(df) * 0.8)

    train = df.iloc[:split]
    test = df.iloc[split:]


    X_train = train[features]
    y_train = train[TARGET]

    X_test = test[features]
    y_test = test[TARGET]


    print("Training rows:", len(train))
    print("Testing rows:", len(test))


    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)

    evaluate(
        y_test,
        rf_pred,
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

    evaluate(
        y_test,
        xgb_pred,
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


    print("\nDeployment models saved!")


if __name__ == "__main__":
    main()