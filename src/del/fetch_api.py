import os
import requests
import pandas as pd
import sqlite3
from datetime import datetime

API_KEY = os.getenv("OPENAQ_API_KEY")

url = "https://api.openaq.org/v3/locations?limit=1"
headers = {
    "X-API-Key": API_KEY,
    "Accept": "application/json"
}


def fetch_india_locations():
    url = "https://api.openaq.org/v3/locations"

    params = {
        "country": "IN",
        "limit": 500
    }

    r = requests.get(url, headers=headers, params=params)
    r.raise_for_status()

    data = r.json()["results"]
    station_ids = [loc["id"] for loc in data if loc.get("isMonitor")]
    return station_ids

def fetch_india_measurements(limit=1000):
    url = "https://api.openaq.org/v3/measurements"

    params = {
        "coordinates": "28.63576,77.22445",
        "radius": 10000,
        "country": "IN",
        "limit": limit,
        "parameters": "pm25,pm10,no2,so2,co,o3"
    }

    r = requests.get(url, headers=headers, params=params)
    #r.raise_for_status()
    print(r.status_code)
    print(r.text[:500])

    #return pd.json_normalize(r.json()["results"])



def fetch_india_station_data(limit=1000,page=1):
    headers = {
        "X-API-Key": API_KEY
    }

    params = {
        #"country": "IN",
        "limit": limit,
        "page": page,
        "sort": "desc",
        "parameter": "pm25,pm10,no2,so2,co,o3"
    }

    r = requests.get(BASE_URL, headers=headers, params=params)
    r.raise_for_status()

    data = r.json()["results"]
    df = pd.json_normalize(data)

    return df


def save_to_db(df):
    print("\nCOLUMNS:", df.columns.tolist())
    
   
    if "locationId" in df.columns:
        df = df.rename(columns={"locationId": "StationId"})
    elif "location.id" in df.columns:
        df = df.rename(columns={"location.id": "StationId"})
    else:
        raise KeyError("No location column found!")

    if "date.utc" in df.columns:
        df = df.rename(columns={"date.utc": "Datetime"})
    elif "dateUTC" in df.columns:
        df = df.rename(columns={"dateUTC": "Datetime"})
    else:
        raise KeyError("No datetime column found!")

    df_clean = df[["StationId", "Datetime", "parameter", "value"]]

    conn = sqlite3.connect("db/air_quality.db")
    df_clean.to_sql("raw_api_data", conn, if_exists="append", index=False)
    conn.close()

    print("Saved", len(df_clean), "rows.")

if __name__ == "__main__":
    df = fetch_india_measurements()
    print(df.columns.tolist())
    save_to_db(df)
