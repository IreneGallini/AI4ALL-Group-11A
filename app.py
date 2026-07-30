"""
Streamlit dashboard for AI4ALL Group 11A electricity demand forecasting.

Users can:
- Select a model (Random Forest or XGBoost)
- Select a region
- Enter date, time, and apparent temperature
- Get a predicted electricity demand value

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


prediction_date = st.date_input(
    "Prediction Date"
)


prediction_time = st.time_input(
    "Prediction Time"
)


temperature = st.number_input(
    "Apparent Temperature (°F)",
    value=70.0
)



# Prediction
if st.button("Predict Demand"):

    dt = datetime.combine(
        prediction_date,
        prediction_time
    )


    hour = dt.hour
    day_of_week = dt.weekday()
    month = dt.month
    is_weekend = int(day_of_week >= 5)


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


    st.success(
        f"Predicted Electricity Demand: {prediction:,.2f} MWh"
    )


    st.subheader("Input Summary")

    summary = pd.DataFrame(
        {
            "Feature": [
                "Region",
                "Date",
                "Time",
                "Hour",
                "Day of Week",
                "Month",
                "Weekend",
                "Apparent Temperature"
            ],
            "Value": [
                region,
                prediction_date,
                prediction_time,
                hour,
                day_of_week,
                month,
                "Yes" if is_weekend else "No",
                f"{temperature} °F"
            ]
        }
    )

    st.table(summary)