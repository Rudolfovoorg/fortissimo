import os
import pandas as pd
import streamlit as st
import altair as alt
import sys
from pathlib import Path
import glob
from db import GetClientDataInTimeframe2,GetMeasurementsInTimeFrame 
import datetime
import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pytz
# ---------- PATH / DATA ----------
# DATA_FILE = os.path.join("", "load_calculated.csv")
# data_all = pd.read_csv(DATA_FILE,  low_memory=False)

def prep_energy(clientId):
    data_all = GetClientDataInTimeframe2(clientId,datetime(year=2025,month=8,day=30),datetime.now(tz=pytz.timezone("Europe/Ljubljana")).replace(minute=0,second=0,hour=0))
    data_all["TimeStampMeasured"] = pd.to_datetime(
        data_all["TimeStampMeasured"],
        utc=True,
        errors="coerce"
    ).dt.tz_convert(None)

    # Drop rows where timestamp failed to parse
    data_all = data_all.dropna(subset=["TimeStampMeasured"])

    # Always sort for correct resampling / ffill
    data_all = data_all.sort_values("TimeStampMeasured")

    # ---------- FILTER ----------
    cutoff = pd.Timestamp("2025-09-01")
    subset = data_all.loc[data_all["TimeStampMeasured"] > cutoff, ["TimeStampMeasured", "LoadEnergyCalculated"]].copy()

    # ---------- CLEAN SPIKES (HARD CODED) CAREFUL!!! - discuss this with ROBOTINA: ----------
    subset.loc[subset["LoadEnergyCalculated"] > 40000, "LoadEnergyCalculated"] = pd.NA
    subset["LoadEnergyCalculated"] = subset["LoadEnergyCalculated"].ffill()

    # ---------- RESAMPLE 15 MIN ----------
    # Set index for resample
    energy_ts = subset.set_index("TimeStampMeasured")["LoadEnergyCalculated"]

    df_energy_15 = (
        energy_ts
        .resample("15T")
        .sum()
        .rename("LoadEnergyCalculated_15")
        .reset_index()
    )

    # CAREFUL!!! - discuss this with ROBOTINA: apply the same spike rule after aggregation
    df_energy_15.loc[df_energy_15["LoadEnergyCalculated_15"] > 40000, "LoadEnergyCalculated_15"] = pd.NA
    df_energy_15["LoadEnergyCalculated_15"] = df_energy_15["LoadEnergyCalculated_15"].ffill()
    df_energy_15=df_energy_15.reset_index()
    return df_energy_15
def prep_weather(clientId):
    df_w = GetMeasurementsInTimeFrame(clientId,datetime(year=2025,month=8,day=30),datetime.now(tz=pytz.timezone("Europe/Ljubljana"))); 
    df_w['Time'] = pd.to_datetime(df_w['Time'],utc=True)

    df_w =df_w[['Time','Temperature', 'Humidity']]

    df_w.set_index('Time', inplace=True)

    df_w.index = [ts.replace(tzinfo=None) for ts in df_w.index]

    df_w.index = pd.to_datetime(df_w.index)

    merged_15min_mean_w = df_w.resample('15T').mean()
    merged_15min_mean_w.reset_index(inplace=True)

    merged_15min_mean_w.rename(columns={'index': 'Time'}, inplace=True)

    return merged_15min_mean_w 
    #merged_15min_mean_w.to_csv('weather_15.csv')

    # Save only final output (if you still want this file)
    #df_energy_15.to_csv("energy_wh_15.csv", index=False)
def prep_features(data_energy:pd.DataFrame,data_weather:pd.DataFrame):


    df_weather = data_weather.copy()
    df_energy = data_energy.copy()

    # ensure both are real datetimes and aligned to same resolution
    df_weather["Time"] = pd.to_datetime(df_weather["Time"])
    df_energy["TimeStampMeasured"] = pd.to_datetime(df_energy["TimeStampMeasured"])

    # rename to same name for easier merging
    df_weather = df_weather.rename(columns={"Time": "Timestamp"})
    df_energy = df_energy.rename(columns={"TimeStampMeasured": "Timestamp"})

    # perform inner join (only timestamps that exist in both)
    df_merged = pd.merge(df_energy, df_weather, on="Timestamp", how="inner")

    #df_merged.drop('Unnamed: 0', axis=1, inplace=True)


    # Extract year, month, day, day of year, and day of week
    #df_merged['Year'] = df_merged['Time'].dt.year
    df_merged['Month'] = df_merged['Timestamp'].dt.month
    df_merged['Day'] = df_merged['Timestamp'].dt.day
    #df_merged['Day_of_year'] = df_merged['Time'].dt.dayofyear
    df_merged['DayOfWeek'] = df_merged['Timestamp'].dt.dayofweek
    df_merged['Hour'] = df_merged['Timestamp'].dt.hour

    # Cyclical transformations of time features (hour, day of month, and day of week)
    df_merged['hour_sin'] = np.sin(2 * np.pi * df_merged['Hour'] / 24)
    df_merged['hour_cos'] = np.cos(2 * np.pi * df_merged['Hour'] / 24)

    df_merged['sin_day_of_month'] = np.sin(2 * np.pi * df_merged['Day'] / 30)  # Assume 30 days in month
    df_merged['cos_day_of_month'] = np.cos(2 * np.pi * df_merged['Day'] / 30)


    # ---------- Step 4: Create lag features ----------
    def create_lag_features(df, target_col, max_lag=15):
        for lag in range(1, max_lag + 1):
            df[f'lag_{lag}'] = df[target_col].shift(lag)
        return df

    df_merged = create_lag_features(df_merged, 'LoadEnergyCalculated_15', max_lag=15)
    df_merged.dropna(inplace=True)

    df_merged.columns
    # ---------- Step 5: Hourly and Daily Interactions ----------
    # Interaction feature between temperature and hour (Temperature * Hour)
    df_merged['Temperature_Hour'] = df_merged['Temperature'] * df_merged['Hour']

    # ---------- Step 6: One-Hot Encoding for DayOfWeek ----------
    df_merged = pd.get_dummies(df_merged, columns=['DayOfWeek'], drop_first=True)

    #df_merged['price_payed_per_interval'] = (df_merged['consumerPower1'] / 1e6) * df_merged['EnergyPricePrediction']

    # ---------- Step 7: Create features for 'target' variable and drop unnecessary columns ----------
    # If target variable is 'consumerPower1' (the power consumption)
    # Drop columns not necessary for modeling
    #df_merged = df_merged.drop(columns=['Unnamed: 0_y'])
    df_merged.to_csv('csvFiles/features_training_dom_mm.csv')
    return df_merged

def get_latest_model_path(clientId):
    model_files = glob.glob(f"modelsV2/bilstm_{clientId}_dommm_*.keras")

    if not model_files:
        raise FileNotFoundError("No model files found in modelsV2.")

    def extract_date(filepath):
        filename = os.path.basename(filepath)
        date_str = filename.replace(f"bilstm_{clientId}_dommm_", "").replace(".keras", "")
        return datetime.strptime(date_str, "%Y-%m-%dT%H-%M-%S")

    latest_model = max(model_files, key=extract_date)
    return latest_model
def forecast_next_day(clientId,df_full):
    #DATA_FILE = "features_training_dom_mm.csv"
    latest_model_path=get_latest_model_path(clientId)
    latest_model_date= latest_model_path.replace(".keras","").split("_")[-1]

    MODEL_FILE = f"modelsV2/bilstm_{clientId}_dommm_{latest_model_date}.keras"
    SCALER_X_FILE = f"modelsV2/scaler_{clientId}_X_{latest_model_date}.pkl"
    SCALER_Y_FILE = f"modelsV2/scaler_{clientId}_y_{latest_model_date}.pkl"
    FEATURE_COLS_FILE = "modelsV2/feature_columns.pkl"

    FORECAST_FILE = "forecastV2/forecast_next_day.csv"

    SEQ_LEN = 288
    PRED_STEPS = 96
    model = tf.keras.models.load_model(MODEL_FILE)
    scaler_X = joblib.load(SCALER_X_FILE)
    scaler_y = joblib.load(SCALER_Y_FILE)
    feature_cols = joblib.load(FEATURE_COLS_FILE)
    df_full = df_full.copy()
    df_full["Timestamp"] = pd.to_datetime(df_full["Timestamp"], errors="coerce")
    df_full = df_full.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    data_prepared = df_full.drop(columns=["Timestamp", "Unnamed: 0"], errors="ignore")
    X_all = data_prepared.drop(columns=["LoadEnergyCalculated_15"])

    # ensure same columns/order as training
    X_all = X_all.reindex(columns=feature_cols, fill_value=0)

    X_all_scaled = scaler_X.transform(X_all)

    if len(X_all_scaled) < SEQ_LEN:
        raise ValueError(f"Not enough rows for SEQ_LEN={SEQ_LEN}. Got {len(X_all_scaled)}")

    x_input = np.expand_dims(X_all_scaled[-SEQ_LEN:], axis=0)  # (1, seq_len, n_features)

    y_next_scaled = model.predict(x_input)  # (1, 96)
    y_next = scaler_y.inverse_transform(y_next_scaled.reshape(-1, 1)).flatten()

    last_ts = df_full["Timestamp"].iloc[-1]
    future_index = pd.date_range(
        start=last_ts + pd.Timedelta(minutes=15),
        periods=PRED_STEPS,
        freq="15min"
    )

    forecast_df = pd.DataFrame({
        "Timestamp": future_index,
        "LoadEnergyForecast_15": y_next
    })

    # Extra 4 steps
    last_forecast_value = float(y_next[-1])

    future_index_4 = pd.date_range(
        start=future_index[-1] + pd.Timedelta(minutes=15),
        periods=4,
        freq="15min"
    )

    extra_df = pd.DataFrame({
        "Timestamp": future_index_4,
        "LoadEnergyForecast_15": [last_forecast_value] * 4
    })

    # Append (modern pandas way)
    forecast_df = pd.concat([forecast_df, extra_df], ignore_index=True)
    forecast_df.to_csv(FORECAST_FILE, index=False)
    return forecast_df

