"""
Fetches hourly weather data from Open-Meteo for all 8 EIA grid regions
and merges it with the EIA electricity demand data.

Usage:
    python src/weather_merge.py
"""

import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd
import os

# Setup connection to Open-Meteo
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Region -> (city, latitude, longitude)
regions = {
    "BPAT": ("Seattle",       47.6062, -122.3321),
    "CISO": ("Sacramento",    38.5816, -121.4944),
    "ERCO": ("Dallas",        32.7767,  -96.7970),
    "ISNE": ("Boston",        42.3601,  -71.0589),
    "MISO": ("Chicago",       41.8781,  -87.6298),
    "NYIS": ("New York",      40.7128,  -74.0060),
    "PJM":  ("Philadelphia",  39.9526,  -75.1652),
    "SWPP": ("Oklahoma City", 35.4676,  -97.5164),
}

START = "2025-06-20"
END   = "2026-06-20"

url = "https://archive-api.open-meteo.com/v1/archive"
all_weather = []

for region, (city, lat, lon) in regions.items():
    print(f"Fetching {city} for {region}...")

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START,
        "end_date": END,
        "hourly": "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,wind_speed_10m",
        "temperature_unit": "fahrenheit",
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    hourly = response.Hourly()

    hourly_data = {
        "datetime_utc": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "temperature":   hourly.Variables(0).ValuesAsNumpy(),
        "apparent_temp": hourly.Variables(1).ValuesAsNumpy(),
        "humidity":      hourly.Variables(2).ValuesAsNumpy(),
        "precipitation": hourly.Variables(3).ValuesAsNumpy(),
        "wind_speed":    hourly.Variables(4).ValuesAsNumpy(),
    }
    hourly_data["region"] = region

    df = pd.DataFrame(data=hourly_data)
    all_weather.append(df)

# Combine all regions
weather_df = pd.concat(all_weather, ignore_index=True)
weather_df["datetime_utc"] = weather_df["datetime_utc"].dt.tz_localize(None)

# Rename columns
weather_df = weather_df.rename(columns={
    "temperature":   "temperature_F",
    "apparent_temp": "apparent_temp_F",
    "humidity":      "humidity_pct",
    "precipitation": "precipitation_mm",
    "wind_speed":    "wind_speed_kmh",
})

print(f"\nDone! Weather data shape: {weather_df.shape}")

# Save raw weather
os.makedirs("data/raw", exist_ok=True)
weather_df.to_csv("data/raw/weather_data.csv", index=False)
print("Saved weather_data.csv")

# Merge with EIA data
eia = pd.read_csv("data/raw/eia_energy_data.csv", parse_dates=["datetime_utc"])
merged = eia.merge(weather_df, on=["datetime_utc", "region"], how="left")

print(f"Merged shape: {merged.shape}")
print(f"Missing weather values: {merged['temperature_F'].isna().sum()}")

# Save merged
os.makedirs("data/processed", exist_ok=True)
merged.to_csv("data/processed/eia_with_weather.csv", index=False)
print("Saved eia_with_weather.csv")