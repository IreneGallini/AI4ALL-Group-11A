"""
Streamlit dashboard for AI4ALL Group 11A electricity demand forecasting.

Two-screen flow, toggled via st.session_state.view:
- "input": model/region/month/day-type/hour/temperature controls
- "results": compact summary, predicted demand, region map, demand chart

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

import altair as alt
import joblib
import pandas as pd
import pydeck as pdk
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

# Region -> (city, latitude, longitude), reused from src/weather_merge.py
REGION_COORDS = {
    "BPAT": ("Seattle",       47.6062, -122.3321),
    "CISO": ("Sacramento",    38.5816, -121.4944),
    "ERCO": ("Dallas",        32.7767,  -96.7970),
    "ISNE": ("Boston",        42.3601,  -71.0589),
    "MISO": ("Chicago",       41.8781,  -87.6298),
    "NYIS": ("New York",      40.7128,  -74.0060),
    "PJM":  ("Philadelphia",  39.9526,  -75.1652),
    "SWPP": ("Oklahoma City", 35.4676,  -97.5164),
}

# Human-readable names for each region code, per the README/CLAUDE.md
# region descriptions — acronyms alone aren't meaningful to most users
REGION_NAMES = {
    "BPAT": "Pacific Northwest",
    "CISO": "California",
    "ERCO": "Texas",
    "ISNE": "New England",
    "MISO": "Midwest",
    "NYIS": "New York",
    "PJM": "Mid-Atlantic",
    "SWPP": "Southwest Power Pool",
}


def region_label(code):
    city = REGION_COORDS[code][0]
    return f"{REGION_NAMES[code]} ({code}, {city})"


MAP_KEY = "region_map"

LOW_COLOR = (33, 102, 172)   # blue = lower demand
HIGH_COLOR = (178, 24, 43)   # red = higher demand
HIGHLIGHT_COLOR = [255, 215, 0]  # gold outline for the selected region


st.set_page_config(
    page_title="Electricity Demand Forecast",
    layout="wide"
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


@st.cache_data
def load_demand_profile():
    df = pd.read_csv(
        WEATHER_DATA_PATH,
        usecols=["region", "month", "is_weekend", "hour", "demand_mwh"]
    )

    return df.groupby(["region", "month", "is_weekend", "hour"])["demand_mwh"].mean()


def resolve_temperature(target_region, month, temp_choice, custom_value, temp_stats):
    if temp_choice == "Custom":
        return custom_value

    stats_row = temp_stats.loc[(target_region, month)]
    mean_temp = float(stats_row["mean"])
    std_temp = float(stats_row["std"]) if pd.notna(stats_row["std"]) else 0.0

    if temp_choice == "Typical":
        return mean_temp
    elif temp_choice == "Colder than usual":
        return mean_temp - std_temp
    else:  # "Hotter than usual"
        return mean_temp + std_temp


def predict_demand(model, features, target_region, hour, day_of_week, month, is_weekend, temperature):
    user_input = {
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": is_weekend,
        "apparent_temp_F": temperature,
    }

    for r in REGION_COORDS:
        user_input[f"region_{r}"] = (
            1 if r == target_region else 0
        )

    input_df = pd.DataFrame([user_input])

    # Ensure same feature order as training
    for feature in features:
        if feature not in input_df.columns:
            input_df[feature] = 0

    input_df = input_df[features]

    return model.predict(input_df)[0]


def demand_to_color(value, vmin, vmax):
    if vmax == vmin:
        t = 0.5
    else:
        t = (value - vmin) / (vmax - vmin)

    r = LOW_COLOR[0] + t * (HIGH_COLOR[0] - LOW_COLOR[0])
    g = LOW_COLOR[1] + t * (HIGH_COLOR[1] - LOW_COLOR[1])
    b = LOW_COLOR[2] + t * (HIGH_COLOR[2] - LOW_COLOR[2])

    return [int(r), int(g), int(b), 200]


regions = list(REGION_COORDS.keys())

st.session_state.setdefault("view", "input")

DEFAULT_INPUTS = {
    "model_choice": "Random Forest",
    "region": regions[0],
    "month": MONTHS[0],
    "day_type": "Weekday",
    "temp_choice": "Typical",
}

# Streamlit prunes a widget-tied session_state entry once that widget stops
# being instantiated for a run (true for every screen-1 input while screen 2
# is showing) — restore from the plain, never-pruned "saved_inputs" snapshot
# (written when "See Typical Demand Scenario" is clicked) before screen 1's
# widgets are (re)created, falling back to hard defaults on first load.
for key, value in st.session_state.get("saved_inputs", {}).items():
    st.session_state.setdefault(key, value)
for key, value in DEFAULT_INPUTS.items():
    st.session_state.setdefault(key, value)

# Apply any pending map click, so the dropdown, session state, and map stay
# in sync in both directions — and clicking the map works whether we're on
# screen 1 or screen 2. Also mirror it into saved_inputs immediately: the
# region key is itself widget-tied (only the screen-1 selectbox registers
# it) and gets pruned the same way as the other inputs once screen 1 stops
# rendering, so saved_inputs — not the raw "region" key — is what survives
# a later trip back to screen 1 via "Edit Settings".
map_selection = st.session_state.get(MAP_KEY)
if map_selection:
    selected_objs = map_selection.get("selection", {}).get("objects", {}).get("region_layer", [])
    if selected_objs:
        clicked_region = selected_objs[0]["region"]
        st.session_state["region"] = clicked_region
        if "saved_inputs" in st.session_state:
            st.session_state.saved_inputs["region"] = clicked_region


# ============================== Screen 1: Input ==============================
if st.session_state.view == "input":

    st.title("Electricity Demand Forecasting")

    st.markdown(
        "Predict hourly electricity demand for major US grid regions "
        "using Random Forest and XGBoost models."
    )

    st.caption(
        "This estimates typical electricity demand based on time of day, day "
        "of week, season, and temperature. It does not use real-time data or "
        "weather forecasts."
    )

    st.header("Forecast Settings")

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        model_choice = st.selectbox(
            "Choose Model",
            ["Random Forest", "XGBoost"],
            key="model_choice"
        )
    with row1_col2:
        region = st.selectbox(
            "Choose Region",
            regions,
            key="region",
            format_func=region_label
        )

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        month_name = st.selectbox(
            "Month",
            MONTHS,
            key="month"
        )
    with row2_col2:
        day_type = st.radio(
            "Day Type",
            ["Weekday", "Weekend"],
            horizontal=True,
            key="day_type"
        )
    month = MONTHS.index(month_name) + 1

    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        prediction_time = st.time_input(
            "Prediction Time",
            key="prediction_time"
        )
    with row3_col2:
        temp_choice = st.radio(
            "Apparent Temperature",
            ["Typical", "Colder than usual", "Hotter than usual", "Custom"],
            key="temp_choice"
        )

    temp_stats = load_temperature_stats()
    stats_row = temp_stats.loc[(region, month)]
    typical_temp = float(stats_row["mean"])
    temp_std = float(stats_row["std"]) if pd.notna(stats_row["std"]) else 0.0
    colder_temp = typical_temp - temp_std
    hotter_temp = typical_temp + temp_std

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
        custom_default = st.session_state.get("custom_temp")
        if custom_default is None:
            custom_default = typical_temp
        temperature = st.slider(
            "Apparent Temperature (°F)",
            min_value=-20.0,
            max_value=110.0,
            value=custom_default,
            key="custom_temp"
        )
        st.caption(f"Using {temperature:.0f}°F")

    if st.button("See Typical Demand Scenario", key="predict_btn", use_container_width=True):
        saved_inputs = {
            "model_choice": model_choice,
            "region": region,
            "month": month_name,
            "day_type": day_type,
            "prediction_time": prediction_time,
            "temp_choice": temp_choice,
        }
        if st.session_state.get("custom_temp") is not None:
            saved_inputs["custom_temp"] = st.session_state["custom_temp"]

        st.session_state.saved_inputs = saved_inputs
        st.session_state.view = "results"
        st.rerun()


# ============================== Screen 2: Results ==============================
elif st.session_state.view == "results":

    # "region" is read live (kept fresh every run by the map-click sync
    # logic above) rather than from the frozen snapshot, so clicking a
    # different marker on this screen updates results immediately.
    region = st.session_state["region"]

    saved = st.session_state.saved_inputs
    model_choice = saved["model_choice"]
    month_name = saved["month"]
    month = MONTHS.index(month_name) + 1
    day_type = saved["day_type"]
    is_weekend = int(day_type == "Weekend")
    day_of_week = (
        WEEKEND_REPRESENTATIVE_DAY if is_weekend else WEEKDAY_REPRESENTATIVE_DAY
    )
    prediction_time = saved["prediction_time"]
    hour = prediction_time.hour
    temp_choice = saved["temp_choice"]
    custom_temp = saved.get("custom_temp")

    temp_stats = load_temperature_stats()
    temperature = resolve_temperature(region, month, temp_choice, custom_temp, temp_stats)

    summary_col, edit_col = st.columns([5, 1])
    with summary_col:
        st.markdown(
            f"**{model_choice} · {region} · {month_name} · {day_type} · "
            f"{hour:02d}:00 · {temperature:.0f}°F**"
        )
    with edit_col:
        if st.button("← Edit Settings", key="edit_settings_btn"):
            st.session_state.view = "input"
            st.rerun()

    if model_choice == "Random Forest":
        model, features = load_rf_model()
    else:
        model, features = load_xgb_model()

    prediction = predict_demand(
        model, features, region, hour, day_of_week, month, is_weekend, temperature
    )

    metrics = load_deployment_metrics()
    region_mae = metrics[model_choice]["per_region"][region]["mae"]

    prediction_rounded = round(prediction, -1)
    mae_rounded = round(region_mae, -1)

    st.metric("Predicted Demand", f"~{prediction_rounded:,.0f} MWh")
    st.caption(f"Typically within ±{mae_rounded:,.0f} MWh for {region}")

    map_col, chart_col = st.columns(2)

    with map_col:
        map_predictions = []
        for r in regions:
            r_temperature = resolve_temperature(r, month, temp_choice, custom_temp, temp_stats)
            r_prediction = predict_demand(
                model, features, r, hour, day_of_week, month, is_weekend, r_temperature
            )
            city, lat, lon = REGION_COORDS[r]
            map_predictions.append({
                "region": r,
                "region_name": REGION_NAMES[r],
                "city": city,
                "lat": lat,
                "lon": lon,
                "predicted_demand": r_prediction,
            })

        map_df = pd.DataFrame(map_predictions)
        vmin = map_df["predicted_demand"].min()
        vmax = map_df["predicted_demand"].max()
        map_df["color"] = map_df["predicted_demand"].apply(lambda v: demand_to_color(v, vmin, vmax))
        is_selected = map_df["region"] == region
        map_df["radius"] = is_selected.map({True: 70000, False: 45000})
        map_df["line_color"] = is_selected.apply(lambda sel: HIGHLIGHT_COLOR if sel else [255, 255, 255])
        map_df["line_width"] = is_selected.map({True: 4, False: 2})

        region_layer = pdk.Layer(
            "ScatterplotLayer",
            id="region_layer",
            data=map_df,
            get_position=["lon", "lat"],
            get_fill_color="color",
            get_radius="radius",
            get_line_color="line_color",
            line_width_min_pixels=2,
            get_line_width="line_width",
            stroked=True,
            pickable=True,
            auto_highlight=True,
        )

        label_layer = pdk.Layer(
            "TextLayer",
            data=map_df,
            get_position=["lon", "lat"],
            get_text="region_name",
            get_size=14,
            get_color=[20, 20, 20],
            get_pixel_offset=[0, -22],
            get_text_anchor="'middle'",
            pickable=False,
        )

        deck = pdk.Deck(
            layers=[region_layer, label_layer],
            initial_view_state=pdk.ViewState(latitude=39, longitude=-98, zoom=3),
            tooltip={"html": "<b>{region_name}</b> ({region} — {city})<br/>~{predicted_demand} MWh"},
            map_provider="carto",
            map_style="light",
        )

        st.pydeck_chart(
            deck,
            on_select="rerun",
            selection_mode="single-object",
            key=MAP_KEY,
            height=380,
        )

        st.caption("Click a marker to select that region.")

        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:8px; max-width:400px;">
                <span>Lower demand</span>
                <div style="flex:1; height:12px; border-radius:6px;
                            background:linear-gradient(to right, rgb(33,102,172), rgb(178,24,43));"></div>
                <span>Higher demand</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with chart_col:
        demand_profile = load_demand_profile()

        try:
            profile_series = demand_profile.loc[(region, month, is_weekend)]
            profile_df = profile_series.reset_index()
            profile_df.columns = ["hour", "demand_mwh"]

            line = alt.Chart(profile_df).mark_line(color="#4C78A8").encode(
                x=alt.X("hour:Q", title="Hour of Day"),
                y=alt.Y("demand_mwh:Q", title="Demand (MWh)"),
                tooltip=[alt.Tooltip("hour:Q", title="Hour"), alt.Tooltip("demand_mwh:Q", title="Avg Demand (MWh)", format=",.0f")],
            )

            point_df = pd.DataFrame({"hour": [hour], "demand_mwh": [prediction]})
            point = alt.Chart(point_df).mark_point(size=180, filled=True, color="#B2182B").encode(
                x="hour:Q",
                y="demand_mwh:Q",
                tooltip=[alt.Tooltip("hour:Q", title="Selected Hour"), alt.Tooltip("demand_mwh:Q", title="Predicted Demand (MWh)", format=",.0f")],
            )

            st.altair_chart(
                (line + point).properties(
                    title=f"Typical {day_type} Demand — {region}, {month_name}",
                    height=380
                ),
                use_container_width=True
            )
        except KeyError:
            st.info("Not enough historical data to build a demand profile for this region/month/day type.")
