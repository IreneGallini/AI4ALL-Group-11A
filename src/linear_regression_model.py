import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
BASE_DIR = Path(__file__).resolve().parent.parent

data_path = BASE_DIR/"data"/"processed"/"eia_with_weather.csv"
df = pd.read_csv(data_path)

# Convert time and sort
df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
df = df.sort_values("datetime_utc").reset_index(drop=True)

# Add cyclical time features
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# One-hot encode categories
df = pd.get_dummies(
    df,
    columns=["region"],
    drop_first=True
)

region_columns = [
    col for col in df.columns 
    if col.startswith("region_")
]

# Features
feature_columns = [
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "day_of_week",
    "is_weekend",

    "temperature_F",
    "apparent_temp_F",
    "humidity_pct",
    "precipitation_mm",
    "wind_speed_kmh",

    "solar_gen_mwh",
    "wind_gen_mwh"
] + region_columns

target_column = "demand_mwh"

# Remove missing values
df_clean = df.dropna(
    subset=feature_columns + [target_column]
)

# Time-based train/test split (Last 20% of data is future test data)
split_index = int(len(df_clean) * 0.8)

train = df_clean.iloc[:split_index]
test = df_clean.iloc[split_index:]

X_train = train[feature_columns]
y_train = train[target_column]

X_test = test[feature_columns]
y_test = test[target_column]

print("Training period:")
print(train["datetime_utc"].min(), "to", train["datetime_utc"].max())

print("\nTesting period:")
print(test["datetime_utc"].min(), "to", test["datetime_utc"].max())

# Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predict future
predictions = model.predict(X_test)

# Evaluate
rmse = mean_squared_error(y_test, predictions) ** 0.5
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nLinear Regression Future Forecast Results: ")
print(f"Rows tested: {len(test):,}")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f} MWh")
print(f"MAE: {mae:.2f} MWh")

