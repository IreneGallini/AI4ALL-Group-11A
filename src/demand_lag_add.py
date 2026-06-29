"""
Adds lag features to the merged EIA and weather dataset.

Usage:
    python src/demand_lag_add.py
"""

import pandas as pd
import os

INPUT_FILE = "data/processed/eia_with_weather.csv"
OUTPUT_FILE = "data/processed/eia_with_features.csv"

def main():
    print(f"📂 Loading data from {INPUT_FILE}...")
    # Load data and ensure datetime_utc is a proper datetime object
    df = pd.read_csv(INPUT_FILE, parse_dates=["datetime_utc"])

    # 1. Sort the data
    # This is critical. We must ensure the data is ordered chronologically 
    # within each specific region before calculating time-based differences.
    df = df.sort_values(by=["region", "datetime_utc"]).reset_index(drop=True)

    print("⚙️ Engineering lag features...")
    
    # 2. Calculate Lags
    # Using groupby("region") ensures that row 1 of ERCOT doesn't subtract row N of CISO
    # .diff(1) subtracts the value 1 row (hour) ago from the current row
    df["demand_lag_1h"] = df.groupby("region")["demand_mwh"].diff(1)
    
    # .diff(24) subtracts the value 24 rows (hours) ago from the current row
    df["demand_lag_24h"] = df.groupby("region")["demand_mwh"].diff(24)

    # 3. Save the Output
    print(f"💾 Saving new dataset to {OUTPUT_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    
    # Print a quick summary of the new columns to verify
    print("\n🎉 Done! Feature summary:")
    print(df[["region", "datetime_utc", "demand_mwh", "demand_lag_1h", "demand_lag_24h"]].head())
    
    # Show how many NaNs were created (the first hour will have NaN for 1h, first 24 for 24h)
    print("\nNull values created by lagging:")
    print(df[["demand_lag_1h", "demand_lag_24h"]].isna().sum())

if __name__ == "__main__":
    main()