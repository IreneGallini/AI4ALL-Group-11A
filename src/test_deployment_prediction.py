from pathlib import Path
import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent


model_data = joblib.load(
    BASE_DIR / "models" / "deployment_rf_model.joblib"
)

model = model_data["model"]
features = model_data["features"]


# Example user input
user_input = {
    "hour": 14,
    "day_of_week": 2,
    "month": 8,
    "is_weekend": 0,
    "apparent_temp_F": 95,
    "region_ERCO": 1,
}


# add missing region columns
for feature in features:
    if feature not in user_input:
        user_input[feature] = 0


input_df = pd.DataFrame([user_input])

input_df = input_df[features]


prediction = model.predict(input_df)[0]

print(f"Predicted demand: {prediction:.2f} MWh")