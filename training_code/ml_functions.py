print("we are at ml func")
import pickle
import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import glob
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime
import random
import db
random.seed(42)
tf.random.set_seed(42)
np.random.seed(42)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras import backend as K
import keras
import xgboost as xgb
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor

from datetime import timedelta

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model


def train_model_solar():
    try:
        #TODO
        return 0
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return 0


def make_predictions_solar():
    try:
        #TODO
        return 0
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return 0
    
def get_season(month):
    if month in [12, 1, 2]:
        return 1
    elif month in [3, 4, 5]:
        return 2
    elif month in [6, 7, 8]:
        return 3
    else:
        return 4

def create_lag_features(df, target_col, max_lag=30):
    for lag in range(1, max_lag + 1):
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    return df


def create_target_series(series, n_steps_out):
    y = []
    for i in range(len(series) - n_steps_out):
        y.append(series[i + 1 : i + 1 + n_steps_out].values)
    return np.array(y)

# Adjust to include 00:00:00 of the next day
#predicted_all = pd.DataFrame()

def get_latest_model_version(client_id,base_path="./MLmodels"):
    """
    Find the latest model version available
    Returns the latest version string (e.g., 'v3') or None if no models exist
    """
    try:
        if not os.path.exists(base_path):
            return None
        
        # Look for existing model files
        existing_models = glob.glob(os.path.join(base_path, f"model_{client_id}_v*.keras"))
        existing_models.extend(glob.glob(os.path.join(base_path, f"model_{client_id}_v*.pkl")))
        
        if not existing_models:
            return None
        
        # Extract version numbers
        versions = []
        for model_path in existing_models:
            filename = os.path.basename(model_path)
            try:
                version_part = filename.split('_v')[1].split('.')[0]
                versions.append(int(version_part))
            except (IndexError, ValueError):
                continue
        
        if not versions:
            return None
        
        # Return latest version
        latest_version = max(versions)
        return f"v{latest_version}"
        
    except Exception as e:
        print(f"Error finding latest version: {e}")
        return None
def get_latest_hpc_model_version(clientId="",base_path="./HPCmodels"):
    """
    Find the latest model version available
    Returns the latest version string (e.g., 'v3') or None if no models exist
    """
    try:
        if not os.path.exists(base_path):
            return None
        
        # Look for existing model files
        existing_models = glob.glob(os.path.join(base_path, f"model_{clientId}_v*.keras"))
        existing_models.extend(glob.glob(os.path.join(base_path, f"model_{clientId}_v*.pkl")))
        
        if not existing_models:
            return None
        
        # Extract version numbers
        versions = []
        for model_path in existing_models:
            filename = os.path.basename(model_path)
            try:
                version_part = filename.split('_v')[1].split('.')[0]
                versions.append(int(version_part))
            except (IndexError, ValueError):
                continue
        
        if not versions:
            return None
        
        # Return latest version
        latest_version = max(versions)
        return f"v{latest_version}"
        
    except Exception as e:
        print(f"Error finding latest version: {e}")
        return None


# Extracting date-related features
def transform_data(df:pd.DataFrame):
    df['TimeOfDay'] = df['TimeStampMeasured'].dt.strftime('%H%M%S')
    df['TimeStampMeasured'] = pd.to_datetime(df['TimeStampMeasured'])
    df['Year'] = df['TimeStampMeasured'].dt.year
    df['Month'] = df['TimeStampMeasured'].dt.month
    df['DayOfWeek'] = df['TimeStampMeasured'].dt.dayofweek + 1
    df['IsWeekend'] = (df['DayOfWeek'] > 5).astype(int)
    df['Season'] = df['Month'].apply(get_season)

    # Extracting lag features
    df = create_lag_features(df, 'PowerFromLoad', max_lag=30)

    # Dropping rows with NaN values introduced by lagging
    df.dropna(inplace=True)

    train_df = df.copy()
    train_df.set_index('TimeStampMeasured', inplace=True)
    #train_df.to_csv(".\\temp\\trainingSet.csv")
    print("saved latest Data")
    return train_df
    # Extract the target variable

def process_or_train(df, scalerPath, modelPath):
    if not os.path.exists(scalerPath):
        print(f"Scaler not found at {scalerPath}. Starting training...")
        train_hourly_model(df)  
        return

def train_hourly_model(client_id:int,df:pd.DataFrame=None):
    if (df is None):
        now= datetime.now()
        start_date=datetime(2025, 3, 11) 
        df = db.GetClientDataInTimeframe(client_id,start_date,now)
        
    scaler = MinMaxScaler()
    if df.empty:
        return None
    df['TimeStampMeasured']= pd.to_datetime(df['TimeStampMeasured'], utc=True)
    train_df = transform_data(df)
    print(train_df.head())
    scaled_data = scaler.fit_transform(train_df)
    target = 'PowerFromLoad'
    target_index = train_df.columns.get_loc(target)

    # Create sequences and targets
    # LSTM parameters
    sequence_length = 24
    prediction_steps = 4

    X, y = [], []
    for i in range(sequence_length, len(scaled_data) - prediction_steps):
        X.append(scaled_data[i - sequence_length:i])
        # Get 4 future values of the target
        y.append([scaled_data[i + j][target_index] for j in range(prediction_steps)])

    X, y = np.array(X), np.array(y)  # y shape: (samples, 4)


    #build model
    model = Sequential()
    model.add(LSTM(150, return_sequences=True, input_shape=(X.shape[1], X.shape[2]), activation='relu'))
    model.add(LSTM(150))
    model.add(Dense(prediction_steps))  # Output 4 values
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    #train
    model.fit(X, y, epochs=2, batch_size=32, verbose=1)
    os.makedirs("./MLmodels", exist_ok=True)
    latestVersion =get_latest_model_version(client_id)
    if (latestVersion is None):
        latestVersion ="v1"
    else:
        version_number = int(latestVersion[1:])  # Remove 'v' and convert to int
        # Increment by 1 and format back to version string
        latestVersion = f"v{version_number + 1}"
    scaler_path = f"./MLmodels/scaler_{client_id}_{latestVersion}.pkl"
    model_path = f"./MLmodels/model_{client_id}_{latestVersion}.keras"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print("scaler Saved")
    model.save(model_path)
    print ("model saved")
    

def run_hourly_predictionv2(client_id=7,df:pd.DataFrame=None,sequence_length = 24,
    prediction_steps = 4,hpc_models=False):
    """
    Run hourly prediction with optional DataFrame input
    
    Args:
        df: Optional DataFrame. If None, will fetch latest data automatically
        sequence_length: Number of historical steps to use for prediction
        prediction_steps: Number of future steps to predict
    Returns:
        Array of predicted values or None if error
    """
    if hpc_models:
        print("using hpc trained models")
        basePath = ".\\HPCModels"
        latestVersion= get_latest_hpc_model_version()
        scalerPath=os.path.join(basePath, f"scaler_{client_id}_{latestVersion}.pkl")
        modelPath = os.path.join(basePath,f"model_{client_id}_{latestVersion}.h5")
    else:
        basePath = ".\\MLmodels"
        latestVersion =get_latest_model_version(client_id)
        scalerPath=os.path.join(basePath, f"scaler_{client_id}_{latestVersion}.pkl")
        modelPath = os.path.join(basePath,f"model_{client_id}_{latestVersion}.keras")
        

    print(modelPath)
    print(scalerPath)
    if not os.path.exists(scalerPath) or not os.path.exists(modelPath):
        df['TimeStampMeasured']= pd.to_datetime(df['TimeStampMeasured'], utc=True)
        train_hourly_model(client_id,df)
    
    if df is None:
        print("No DataFrame provided, fetching latest consumption data...")
        
        clientId=client_id
        now= datetime.now()
        start_date=datetime(2025, 3, 11) 
        df = db.GetClientDataInTimeframe(clientId,start_date,now)
        if df is None or df.empty:
            print("Failed to fetch consumption data")
            return None
    
    with open(scalerPath, 'rb') as f:
        scaler = pickle.load(f)
    df['TimeStampMeasured']= pd.to_datetime(df['TimeStampMeasured'], utc=True)
    #print(df_latest.head(-5))
    transformed_df = transform_data(df) 
    #print(transformed_df.head(-5))
    scaled_data = scaler.transform(transformed_df)

    target = 'PowerFromLoad'
    target_index = transformed_df.columns.get_loc(target)

    print("loading model")
    model=load_model(modelPath)
    last_sequence = scaled_data[-sequence_length:]  # shape: (24, num_features)
    #print(last_sequence)
    #last_scaled = scaler.transform(last_sequence)
    X_input = np.expand_dims(last_sequence, axis=0)  # shape: (1, 24, num_features)

    
    #predict next 4 steps
    y_pred_scaled = model.predict(X_input)
    #print(f"y pred scaled {y_pred_scaled}")
    #dummy array for inverse scaling
    dummy = np.zeros((prediction_steps, scaled_data.shape[1]))
    dummy[:, target_index] = y_pred_scaled[0]

    # Inverse transform
    y_pred_original = scaler.inverse_transform(dummy)[:, target_index]
    print(f" hourly predictor for clientid {client_id} completed successfully")
    return y_pred_original
    #show predicted values
    for i, val in enumerate(y_pred_original, start=1):
        print(f"Predicted value +{i * 15} min: {val:.3f} W")



## hourly predictor
DATA_PATH = "data_energy_weather_agregated.csv"
def load_data():
    df = pd.read_csv(DATA_PATH, index_col=0, dayfirst=True, parse_dates=['Time'])
    df['Time'] = pd.to_datetime(df['Time'])
    return df
def data_agregate_weather(energy_data_cond:pd.DataFrame,measurements_df:pd.DataFrame):
    df = energy_data_cond

    df['TimeStampMeasured'] = pd.to_datetime(df['TimeStampMeasured'],utc=True)

    df =df[['TimeStampMeasured','PowerFromLoad']]

    df.set_index('TimeStampMeasured', inplace=True)

    df.index = [ts.replace(tzinfo=None) for ts in df.index]

    df.index = pd.to_datetime(df.index)

    merged_15min_mean = df.resample('15T').mean()
    merged_15min_mean.reset_index(inplace=True)

    merged_15min_mean.rename(columns={'PowerFromLoad': 'ConsumerPower1'}, inplace=True)
    merged_15min_mean.rename(columns={'index': 'Time'}, inplace=True)

    merged_15min_mean.to_csv('preds_4\\data_agregated_cond.csv')


    df_w = measurements_df

    df_w['Time'] = pd.to_datetime(df_w['Time'],utc=True)

    df_w =df_w[['Time','Temperature', 'Humidity']]

    df_w.set_index('Time', inplace=True)

    df_w.index = [ts.replace(tzinfo=None) for ts in df_w.index]

    df_w.index = pd.to_datetime(df_w.index)

    merged_15min_mean_w = df_w.resample('15T').mean()
    merged_15min_mean_w.reset_index(inplace=True)

    merged_15min_mean_w.rename(columns={'index': 'Time'}, inplace=True)

    merged_15min_mean_w.to_csv('preds_4\\measurements_agregated.csv')

    merged_df = pd.merge(
        merged_15min_mean,
        merged_15min_mean_w[['Time', 'Temperature', 'Humidity']],
        on='Time',
        how='inner'
    )
    merged_df.to_csv('preds_4\\data_energy_weather_agregated.csv')
    return merged_df




def prepare_features(df, target_col='ConsumerPower1', max_lag=24, n_steps_out=4):
    df['Hour'] = df['Time'].dt.hour
    df['hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['Month'] = df['Time'].dt.month
    df['DayOfWeek'] = df['Time'].dt.dayofweek + 1
    df['is_weekend'] = (df['DayOfWeek'] >= 6).astype(int)
    df['Season'] = df['Month'].apply(get_season)
    df = create_lag_features(df, target_col, max_lag)
    df = pd.get_dummies(df, columns=['DayOfWeek'], drop_first=True)
    df['Temperature_Hour'] = df['Temperature'] * df['Hour']
    df.drop(columns=['Hour'], inplace=True)
    df.dropna(inplace=True)
    return df

def make_hourly_prediction(df_weather_agregated:pd.DataFrame,current_time):
    if True:#current_time.minute == 45:
        n_steps_out = 4
        max_lag = 24
        print("It's time to predict.")
        df =df_weather_agregated.copy()
        #print(df)
        stored_time_last45min = current_time

        df = prepare_features(df=df,max_lag=24)

        y = create_target_series(df['ConsumerPower1'], n_steps_out)
        X = df.drop(columns=['ConsumerPower1', 'Time']).iloc[:-n_steps_out]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train = X_scaled
        y_train = y

        base_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.05)
        model = MultiOutputRegressor(base_model)
        model.fit(X_train, y_train)

        x_for_prediction = df[df['Time'] <= stored_time_last45min].iloc[-1:].copy()
        x_for_prediction = x_for_prediction.drop(columns=['ConsumerPower1', 'Time'])
        #print(x_for_prediction)
        x_for_prediction_scaled = scaler.transform(x_for_prediction)
        new_preds = model.predict(x_for_prediction_scaled)
        del df, X, y, X_scaled, scaler, model, x_for_prediction
        gc.collect()
        return new_preds
        # Clean up the memory

    else:
        print("Not prediction time.")




## DAYLI PREDICTION FUNCTIONS

def data_agregation(df:pd.DataFrame):
    df['TimeStampMeasured'] = pd.to_datetime(df['TimeStampMeasured'],utc=True)

    df =df[['TimeStampMeasured','PowerFromLoad']]

    df.set_index('TimeStampMeasured', inplace=True)

    df.index = [ts.replace(tzinfo=None) for ts in df.index]

    df.index = pd.to_datetime(df.index)

    merged_15min_mean:pd.DataFrame = df.resample('15T').mean()
    merged_15min_mean.reset_index(inplace=True)

    merged_15min_mean.rename(columns={'PowerFromLoad': 'ConsumerPower1'}, inplace=True)
    merged_15min_mean.rename(columns={'index': 'Time'}, inplace=True)

    merged_15min_mean.to_csv('./temp/data_agregated_cond.csv')
    return merged_15min_mean

def pv_data_agregation(measurements:pd.DataFrame,energy_data:pd.DataFrame):
    df = measurements

    # Convert the Time column to datetime
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce", utc=True)
    
    # Round down to minute precision (truncate seconds + microseconds)
    df["Time"] = df["Time"].dt.floor("min")


    df["Time"] = df["Time"].dt.strftime("%Y-%m-%d;%H:%M:%S")

    df_pv = pd.read_csv("pv_power.csv", sep=";")   # important, semicolon separated

    df_pv["Time"] = df_pv["date"].astype(str) + ";" + df_pv["time"].astype(str)

    df_pv = df_pv.drop(columns=["date", "time"])
    df_merged = pd.merge(df, df_pv, on="Time", how="inner")
    df_merged = df_merged.rename(columns={"c23525.pv_power": "SolarPower"})

    df["Time"] = pd.to_datetime(df["Time"], utc=True)
    cut_of_date = pd.to_datetime(df_merged["Time"].max(), utc=True)

    energy_data["TimeStampMeasured"] = pd.to_datetime(energy_data["TimeStampMeasured"], utc=True)
    energy_data["TimeStampMeasured"] = energy_data["TimeStampMeasured"].dt.floor("min")
    #energy_data["TimeStampMeasured"] = energy_data["TimeStampMeasured"].dt.strftime("%Y-%m-%d;%H:%M:%S")
    energy_data["PowerToPv"] =energy_data["PowerToPv"]/1000  # Convert to kW
    pv_new_data = energy_data[energy_data["TimeStampMeasured"] >= cut_of_date].copy()
    pv_new_data = pv_new_data.rename(columns={"TimeStampMeasured": "Time"})
    # Don't rename PowerToPv yet

    measurements_new_data = df[df["Time"] >= cut_of_date].copy()
    # Don't rename PowerPV yet

    # Merge without name conflicts
    df_merged_new = pd.merge(measurements_new_data, pv_new_data[["Time", "PowerToPv"]], 
                            on="Time", how="left")

    # Now rename - PowerToPv becomes SolarPower
    df_merged_new = df_merged_new.rename(columns={"PowerToPv": "SolarPower"})

    # If PowerPV column exists and SolarPower is null, use PowerPV
    if 'PowerPV' in df_merged_new.columns:
        df_merged_new['SolarPower'] = df_merged_new['SolarPower'].fillna(df_merged_new['PowerPV'])
        df_merged_new = df_merged_new.drop(columns=['PowerPV'])

    # Concatenate
    df = pd.concat([df_merged, df_merged_new], ignore_index=True)
    df = df.drop(columns=['PowerPV'])
    df["Time"] = pd.to_datetime(df["Time"], utc=True, format="%Y-%m-%d;%H:%M:%S", errors="coerce")
    df = df.sort_values("Time").set_index("Time")

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.columns.difference(num_cols)

    if "ClientId" in num_cols:
        num_cols.remove("ClientId")

    def mode_agg(s: pd.Series):
        m = s.mode(dropna=True)
        return m.iloc[-1] if not m.empty else 'Unknown'

    agg_map = {c: "mean" for c in num_cols}
    agg_map.update({c: mode_agg for c in cat_cols})
    agg_map["ClientId"] = "last"

    # Resample to 15 min
    df_15m = df.resample("15min").agg(agg_map)

    # Format output time column
    df_15m = df_15m.reset_index()

    df_15m["Time"] = pd.to_datetime(df_15m["Time"], errors="coerce")
    df_15m["Time"] = df_15m["Time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Put Time first
    cols = ["Time"] + [c for c in df_15m.columns if c != "Time"]
    df_15m = df_15m[cols]

    for col in ["SunriseDT", "SunsetDT"]:
        s = pd.to_datetime(df_15m[col],utc=True, errors="coerce")      
        s = s.dt.floor("min")                                 
    
        df_15m[col] = s.dt.strftime("%Y-%m-%d %H:%M:%S")     

    df_15m["Temperature"] = df_15m["Temperature"] - 273.15
    df_15m.to_csv("train_test_data.csv", index=False)  # Save to CSV
    train_test_data=df_15m.reset_index(drop=True)
    return train_test_data