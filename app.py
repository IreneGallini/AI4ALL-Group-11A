"""
Streamlit dashboard for AI4ALL Group 11A electricity demand forecasting.

Users can:
- Select a model (Random Forest or XGBoost)
- Select a region
- Enter month, day type, time, and apparent temperature
- Get a typical electricity demand scenario

The deployment models use:
- region
- hour
- day_of_week
- month
- is_weekend
- apparent_temp_F
"""

import json
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
import xgboost as xgb


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
WEATHER_DATA_PATH = BASE_DIR / "data" / "processed" / "eia_with_features.csv"

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# Representative day_of_week values for the weekday/weekend toggle
# (model requires a specific day_of_week, but the UI only exposes
# weekday vs. weekend)
WEEKDAY_REPRESENTATIVE_DAY = 2  # Wednesday
WEEKEND_REPRESENTATIVE_DAY = 5  # Saturday


st.set_page_config(
    page_title="Electricity Demand Forecast",
    layout="wide"
)


st.title("Electricity Demand Forecasting")

st.markdown(
    """
    Predict hourly electricity demand for major US grid regions
    using Random Forest and XGBoost models.
    """
)

st.caption(
    "This estimates typical electricity demand based on time of day, day "
    "of week, season, and temperature. It does not use real-time data or "
    "weather forecasts."
)


# Load models
@st.cache_resource
def load_rf_model():
    data = joblib.load(
        MODEL_DIR / "deployment_rf_model.joblib"
    )

    return data["model"], data["features"]


@st.cache_resource
def load_xgb_model():
    model = xgb.XGBRegressor()

    model.load_model(
        MODEL_DIR / "deployment_xgb_model.json"
    )

    with open(MODEL_DIR / "deployment_features.json") as f:
        features = json.load(f)["features"]

    return model, features


@st.cache_resource
def load_deployment_metrics():
    with open(MODEL_DIR / "deployment_metrics.json") as f:
        return json.load(f)


@st.cache_data
def load_temperature_stats():
    df = pd.read_csv(
        WEATHER_DATA_PATH,
        usecols=["region", "month", "apparent_temp_F"]
    )

    return df.groupby(["region", "month"])["apparent_temp_F"].agg(["mean", "std"])


# User Inputs
st.header("Forecast Settings")


model_choice = st.selectbox(
    "Choose Model",
    ["Random Forest", "XGBoost"]
)


regions = [
    "BPAT",
    "CISO",
    "ERCO",
    "ISNE",
    "MISO",
    "NYIS",
    "PJM",
    "SWPP",
]


region = st.selectbox(
    "Choose Region",
    regions
)


month_name = st.selectbox(
    "Month",
    MONTHS
)
month = MONTHS.index(month_name) + 1


day_type = st.radio(
    "Day Type",
    ["Weekday", "Weekend"],
    horizontal=True
)
is_weekend = int(day_type == "Weekend")
day_of_week = (
    WEEKEND_REPRESENTATIVE_DAY if is_weekend else WEEKDAY_REPRESENTATIVE_DAY
)


prediction_time = st.time_input(
    "Prediction Time"
)
hour = prediction_time.hour


st.subheader("Temperature")

temp_stats = load_temperature_stats()
stats_row = temp_stats.loc[(region, month)]
typical_temp = float(stats_row["mean"])
temp_std = float(stats_row["std"]) if pd.notna(stats_row["std"]) else 0.0
colder_temp = typical_temp - temp_std
hotter_temp = typical_temp + temp_std

temp_choice = st.radio(
    "Apparent Temperature",
    ["Typical", "Colder than usual", "Hotter than usual", "Custom"],
)

if temp_choice == "Typical":
    temperature = typical_temp
    st.caption(f"Using {temperature:.0f}°F (historical average for {region} in {month_name})")
elif temp_choice == "Colder than usual":
    temperature = colder_temp
    st.caption(f"Using {temperature:.0f}°F (~1 std dev below the {region} {month_name} average)")
elif temp_choice == "Hotter than usual":
    temperature = hotter_temp
    st.caption(f"Using {temperature:.0f}°F (~1 std dev above the {region} {month_name} average)")
else:
    temperature = st.slider(
        "Apparent Temperature (°F)",
        min_value=-20.0,
        max_value=110.0,
        value=typical_temp,
    )
    st.caption(f"Using {temperature:.0f}°F")


# Prediction
if st.button("See Typical Demand Scenario"):

    user_input = {
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": is_weekend,
        "apparent_temp_F": temperature,
    }


    # Add region one-hot encoding
    for r in regions:
        user_input[f"region_{r}"] = (
            1 if r == region else 0
        )


    input_df = pd.DataFrame([user_input])


    if model_choice == "Random Forest":

        model, features = load_rf_model()

    else:

        model, features = load_xgb_model()


    # Ensure same feature order as training
    for feature in features:
        if feature not in input_df.columns:
            input_df[feature] = 0


    input_df = input_df[features]


    prediction = model.predict(input_df)[0]

    metrics = load_deployment_metrics()
    region_mae = metrics[model_choice]["per_region"][region]["mae"]

    prediction_rounded = round(prediction, -1)
    mae_rounded = round(region_mae, -1)

    st.success(
        f"~{prediction_rounded:,.0f} MWh "
        f"(typically within ±{mae_rounded:,.0f} MWh)"
    )


    st.subheader("Input Summary")

    summary = pd.DataFrame(
        {
            "Feature": [
                "Region",
                "Month",
                "Day Type",
                "Time",
                "Hour",
                "Apparent Temperature"
            ],
            "Value": [
                region,
                month_name,
                day_type,
                prediction_time,
                hour,
                f"{temperature:.0f} °F"
            ]
        }
    )

    st.table(summary)
