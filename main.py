#DATA PREPROCESSING AND AQI CALCULATION 
import pandas as pd
'''from src.aqi_calculator import calculate_AQI

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
print("\n Preprocessing done — Cleaned file saved!")'''

#----------------------------REGRESSION MODEL TRAINING -------------------------------------
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load cleaned dataframe
df = pd.read_csv("./data/preprocessed/station_hour_clean.csv")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("./data/preprocessed/station_hour_clean.csv")

df = df.dropna(subset=["AQI"])

X = df.drop(columns=["AQI", "AQI_Bucket", "Datetime", "StationId"], errors="ignore")
y = df["AQI"]

X = X.apply(pd.to_numeric, errors='coerce')

imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5
r2 = r2_score(y_test, y_pred)

print("\n===== Regression Performance =====")
print(f"MAE : {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²  : {r2:.3f}")
