import sqlite3
import pandas as pd

DB_PATH = "db/air_quality.db"

PARAM_MAP = {
    "pm25": "PM2.5",
    "pm10": "PM10",
    "no2": "NO2",
    "so2": "SO2",
    "co": "CO",
    "o3": "O3"
}

def pivot_raw_api_data():
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql(
        "SELECT location, date_utc, parameter, value FROM raw_api_data",
        conn
    )

    # Rename columns
    df.rename(columns={
        "location": "StationId",
        "date_utc": "Datetime"
    }, inplace=True)

    # Normalize parameter names
    df["parameter"] = df["parameter"].map(PARAM_MAP)

    # Drop unsupported parameters
    df = df.dropna(subset=["parameter"])

    # Convert datetime
    df["Datetime"] = pd.to_datetime(df["Datetime"]).dt.floor("H")

    # Pivot to wide format
    df_wide = df.pivot_table(
        index=["StationId", "Datetime"],
        columns="parameter",
        values="value",
        aggfunc="mean"
    ).reset_index()

    # Save to DB
    df_wide.to_sql(
        "station_hour_api",
        conn,
        if_exists="replace",
        index=False
    )

    conn.close()
    print("API data pivoted to station-hour format.")

if __name__ == "__main__":
    pivot_raw_api_data()
