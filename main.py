import pandas as pd
from src.aqi_calculator import calculate_AQI

# load CSV
df = pd.read_csv("./data/raw/station_hour.csv", low_memory=False)

# ---- 1. Convert pollutant columns to numeric ----
pollutants = ["PM2.5","PM10","NO","NO2","NOx","NH3","CO","SO2","O3"]
for col in pollutants:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ---- 2. Datetime parsing ----
df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
df["hour"] = df["Datetime"].dt.hour
df["day"] = df["Datetime"].dt.day
df["month"] = df["Datetime"].dt.month
df["year"] = df["Datetime"].dt.year

# ---- 3. Calculate AQI where missing ----
missing_mask = df["AQI"].isna()
df.loc[missing_mask, "AQI"] = df[missing_mask].apply(calculate_AQI, axis=1)

# ---- 4. Fill AQI_Bucket where missing ----
bucket_labels = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
bucket_breaks = [(0,50),(51,100),(101,200),(201,300),(301,400),(401,500)]

def get_bucket(aqi):
    if pd.isna(aqi):
        return None
    for label, (lo, hi) in zip(bucket_labels, bucket_breaks):
        if lo <= aqi <= hi:
            return label
    return None

df.loc[missing_mask, "AQI_Bucket"] = df.loc[missing_mask, "AQI"].apply(get_bucket)

# ---- 5. Handle remaining missing pollutant values (mean imputation) ----
for col in pollutants:
    df[col] = df[col].fillna(df[col].mean())

# ---- 6. Final cleaned DataFrame ----
print(df.head())                      # preview
print(df.info())                      # schema check
df.to_csv("./data/preprocessed/station_hour_clean.csv", index=False)   # save
print("\n✨ Preprocessing done — Cleaned file saved!")
