import os
import requests
import pandas as pd
import sqlite3
from datetime import datetime

API_KEY = os.getenv("OPENAQ_API_KEY")
BASE_URL = "https://api.openaq.org/v2/measurements"

def fetch_india_station_data(limit=1000):
    headers = {
        "X-API-Key": API_KEY
    }

    params = {
        "country": "IN",
        "parameter": ["pm25", "pm10", "no2", "so2", "co", "o3"],
        "limit": limit,
        "sort": "desc"
    }

    r = requests.get(BASE_URL, headers=headers, params=params)
    r.raise_for_status()

    data = r.json()["results"]
    df = pd.json_normalize(data)

    return df

def save_to_db(df):
    conn = sqlite3.connect("db/air_quality.db")

    df_clean = df[[
        "location", "date.utc", "parameter", "value"
    ]]

    df_clean.to_sql(
        "raw_api_data",
        conn,
        if_exists="append",
        index=False
    )

    conn.close()

if __name__ == "__main__":
    df = fetch_india_station_data()
    save_to_db(df)
    print("API data fetched and stored successfully.")
