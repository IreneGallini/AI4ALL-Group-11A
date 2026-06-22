"""
Pulls hourly electricity demand, solar generation, and wind generation
for US grid regions (Balancing Authorities) and saves to CSV.

Usage:
    1. Copy .env.example to .env and paste your EIA API key (free at https://www.eia.gov/opendata/)
    2. Run: python eia_api_script.py
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import time

load_dotenv()
EIA_API_KEY = os.getenv("EIA_API_KEY", "")  # set in .env file

# Date range (ISO 8601, UTC). Adjust as needed.
# EIA hourly data typically has a ~1-2 day lag.
END_DATE   = datetime.now(timezone.utc) - timedelta(days=2)
START_DATE = END_DATE - timedelta(days=365)      # 1 year ≈ 8,760 rows per region

START_STR = START_DATE.strftime("%Y-%m-%dT%H")
END_STR   = END_DATE.strftime("%Y-%m-%dT%H")

# Balancing authorities to pull. These are major US grid regions.
# Full list: https://www.eia.gov/electricity/gridmonitor/about
REGIONS = [
    "CISO",   # California ISO
    "ERCO",   # ERCOT (Texas)
    "MISO",   # Midcontinent ISO
    "PJM",    # PJM Interconnection (Mid-Atlantic/Midwest)
    "NYIS",   # New York ISO
    "ISNE",   # ISO New England
    "SWPP",   # Southwest Power Pool
    "BPAT",   # Bonneville Power (Pacific Northwest)
]

OUTPUT_FILE = "data/raw/eia_energy_data.csv"

# EIA Grid Monitor endpoint (v2)
BASE_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"

# respondent = balancing authority code
# type codes:
#   D  = Demand (electricity consumption)
#   NG = Net Generation
#   SUN = Solar generation
#   WND = Wind generation

TYPE_LABELS = {
    "D":   "demand_mwh",
    "SUN": "solar_gen_mwh",
    "WND": "wind_gen_mwh",
}


def fetch_series(region: str, type_code: str) -> pd.DataFrame:
    """Fetch a single (region, type) time series from the EIA v2 API."""
    params = {
        "api_key":          EIA_API_KEY,
        "frequency":        "hourly",
        "data[0]":          "value",
        "facets[respondent][]": region,
        "facets[type][]":   type_code,
        "start":            START_STR,
        "end":              END_STR,
        "sort[0][column]":  "period",
        "sort[0][direction]": "asc",
        "length":           5000,           # max rows per request
        "offset":           0,
    }

    all_rows = []
    while True:
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        body = resp.json()

        data = body.get("response", {}).get("data", [])
        if not data:
            break

        all_rows.extend(data)

        # Paginate if needed
        total   = int(body["response"].get("total", len(all_rows)))
        offset  = params["offset"] + len(data)
        if offset >= total:
            break
        params["offset"] = offset
        time.sleep(0.2)   # be polite to the API

    if not all_rows:
        print(f"  ⚠  No data for {region} / {type_code}")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)[["period", "value"]]
    df.rename(columns={"period": "datetime_utc", "value": TYPE_LABELS[type_code]}, inplace=True)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df[TYPE_LABELS[type_code]] = pd.to_numeric(df[TYPE_LABELS[type_code]], errors="coerce")
    return df.set_index("datetime_utc")


def main():
    if not EIA_API_KEY:
        raise ValueError("EIA_API_KEY not set. Copy .env.example to .env and add your key.")

    all_region_dfs = []

    for region in REGIONS:
        print(f"\n📡 Pulling {region}...")
        series = {}

        for type_code, col_name in TYPE_LABELS.items():
            print(f"   → {col_name}")
            df = fetch_series(region, type_code)
            if not df.empty:
                series[col_name] = df[col_name]
            time.sleep(0.3)

        if not series:
            print(f"   ⚠  Skipping {region} — no data returned")
            continue

        region_df = pd.DataFrame(series)
        region_df["region"] = region
        region_df = region_df.reset_index()
        all_region_dfs.append(region_df)
        print(f"   ✅ {len(region_df)} rows")

    if not all_region_dfs:
        print("\n❌ No data collected. Check your API key and internet connection.")
        return

    combined = pd.concat(all_region_dfs, ignore_index=True)


    combined["hour"]          = combined["datetime_utc"].dt.hour
    combined["day_of_week"]   = combined["datetime_utc"].dt.dayofweek   # 0=Mon
    combined["month"]         = combined["datetime_utc"].dt.month
    combined["is_weekend"]    = combined["day_of_week"].isin([5, 6]).astype(int)

    # Reorder columns cleanly
    col_order = [
        "datetime_utc", "region",
        "demand_mwh", "solar_gen_mwh", "wind_gen_mwh",
        "renewable_pct",
        "hour", "day_of_week", "month", "is_weekend",
    ]
    combined = combined[[c for c in col_order if c in combined.columns]]
    combined.sort_values(["region", "datetime_utc"], inplace=True)

    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\n🎉 Saved {len(combined):,} rows × {len(combined.columns)} columns → {OUTPUT_FILE}")
    print("\nColumn summary:")
    print(combined.dtypes.to_string())
    print("\nMissing values:")
    print(combined.isnull().sum().to_string())
    print(f"\nDate range: {combined['datetime_utc'].min()} → {combined['datetime_utc'].max()}")
    print(f"Regions:    {sorted(combined['region'].unique())}")


if __name__ == "__main__":
    main()