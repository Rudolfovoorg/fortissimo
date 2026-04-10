import os
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)


data_energy= pd.read_csv('energy_wh_15.csv')
data_weather = pd.read_csv('weather_15.csv')


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

df_merged.drop('Unnamed: 0', axis=1, inplace=True)


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

df_merged.to_csv('features_training_dom_mm.csv')
