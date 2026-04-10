import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import os
import db 
import datetime

#df_w = pd.read_csv('weather.csv')
df_w = db.GetMeasurementsInTimeFrame(4,datetime.datetime(year=2025,month=9,day=1),datetime.datetime.now()); 
df_w['Time'] = pd.to_datetime(df_w['Time'])

df_w =df_w[['Time','Temperature', 'Humidity']]

df_w.set_index('Time', inplace=True)

df_w.index = [ts.replace(tzinfo=None) for ts in df_w.index]

df_w.index = pd.to_datetime(df_w.index)

merged_15min_mean_w = df_w.resample('15T').mean()
merged_15min_mean_w.reset_index(inplace=True)

merged_15min_mean_w.rename(columns={'index': 'Time'}, inplace=True)

merged_15min_mean_w.to_csv('weather_15.csv')

