print("we are at dl")

import time
import datetime
import os
import pandas as pd
import joblib
import numpy as np
import glob
import tensorflow as tf
import keras
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import random
random.seed(42)
tf.random.set_seed(42)
np.random.seed(42)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras import backend as K
import gc
from datetime import timedelta

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model

def train_model_consumption_lstm(agregated_df:pd.DataFrame,client_id:int):
    try:
        print("started lstm training")
        agregated_df = agregated_df.set_index('Time')
        agregated_df.index = pd.to_datetime(agregated_df.index, dayfirst=True)
        print(agregated_df.head())
        # Parse the 'Time' column as datetime
        
        def load_data():
            df = pd.read_csv("data_agregated_cond.csv", index_col=0, dayfirst=True, parse_dates=['Time'])  # +userId
            df['Time'] = pd.to_datetime(df['Time'])
            return df

        def get_season(month):
            if month in [12, 1, 2]:
                return 1
            elif month in [3, 4, 5]:
                return 2
            elif month in [6, 7, 8]:
                return 3
            else:
                return 4

        def create_lag_features(df, target_col, max_lag=10):
            for lag in range(1, max_lag + 1):
                df[f'lag_{lag}'] = df[target_col].shift(lag)
            return df

        df = agregated_df.copy()
        df = df.reset_index()
        # Extract the time of day
        df['TimeOfDay'] = df['Time'].dt.strftime('%H%M%S')
        # Adjust to include 00:00:00 of the next day
        intervals = pd.date_range(start='00:15', end='23:45', freq='15T').time.tolist()
        intervals.append(pd.to_datetime('00:00').time())

        predicted_all = pd.DataFrame()

        intervalCounter = 0;

        for interval in intervals:

            #comment after testing:
            #inerval = datetime.time(0, 15)

            ts_label = f"ts_{interval.strftime('%H%M%S')}"
            df_interval = df[df['TimeOfDay'] == interval.strftime('%H%M%S')].drop(columns=['TimeOfDay'])


            # Extracting date-related features
            df_interval['Time'] = pd.to_datetime(df_interval['Time'])
            df_interval['Year'] = df_interval['Time'].dt.year
            df_interval['Month'] = df_interval['Time'].dt.month
            df_interval['DayOfWeek'] = df_interval['Time'].dt.dayofweek + 1
            df_interval['IsWeekend'] = (df_interval['DayOfWeek'] > 5).astype(int)
            df_interval['Season'] = df_interval['Month'].apply(get_season)

            # Extracting lag features
            df_interval = create_lag_features(df_interval, 'ConsumerPower1', max_lag=10)

            # Dropping rows with NaN values introduced by lagging
            df_interval.dropna(inplace=True)

            # Filter the DataFrame to use only the data before today
            today_date = datetime.datetime.today().strftime('%Y-%m-%d')


            #COMMENT THESE 2 LINES IN DEPLOYMENT:
            #yesterday_date = (datetime.datetime.today() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            #train_df = df_interval[df_interval['Time'] < yesterday_date]

            #UNCOMMENT THIS IN DEPLOYMENT
            train_df = df_interval[df_interval['Time'] < today_date]

            # Extract the target variable
            target = 'ConsumerPower1'

            # Normalize the data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(train_df.drop(columns=['Time']))

            # Create sequences and targets
            sequence_length = 10
            X, y = [], []
            os.makedirs( "models", exist_ok=True)
            os.makedirs( "scalers",  exist_ok=True)
            
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i - sequence_length:i])
                y.append(scaled_data[i, 0])  # Assuming ConsumerPower1 is the first column after dropping Time

            X, y = np.array(X), np.array(y)

            # Build the LSTM model
            model = Sequential()
            model.add(LSTM(150, return_sequences=True, input_shape=(X.shape[1], X.shape[2]), activation='relu'))
            model.add(LSTM(150))
            model.add(Dense(1))
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

            # Train the model
            model.fit(X, y, epochs=20, batch_size=32, verbose=1)
            
            # Save the model and scaler
            model.save(f"models//client_{client_id}_consumption_lstm_model_{intervalCounter}_.keras")  # +userId
            joblib.dump(scaler, f"scalers//client_{client_id}_scaler_lstm_{intervalCounter}.pkl")  # +userId
            intervalCounter += 1

            # Clear memory
            del df_interval, train_df, scaled_data, X, y, model, scaler
            K.clear_session()
            gc.collect()



    except Exception as e:

        print(f"An unexpected error occurred - train_model_lstm: {e}")

def file_cleanup():
    model_path_pv = f'models//client_*_solar_power_lstm_model.h5'
    scaler_X_path = f'scalers//client_*_pv_scaler_X.pkl'
    scaler_y_path = f'scalers//client_*_pv_scaler_y.pkl'
    model_files = glob.glob("models//client_*_consumption_lstm_model_*.keras")
    for file in model_files:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except OSError as e:
            print(f"Error deleting {file}: {e}")

    # Delete existing scaler files  
    scaler_files = glob.glob("scalers//client_*scaler_lstm_*.pkl")
    for file in scaler_files:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except OSError as e:
            print(f"Error deleting {file}: {e}")

    pv_scaler_files = glob.glob(scaler_X_path)
    for file in pv_scaler_files:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except OSError as e:
            print(f"Error deleting {file}: {e}")
    pv_scaler_filesY = glob.glob(scaler_y_path)
    for file in pv_scaler_filesY:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except OSError as e:
            print(f"Error deleting {file}: {e}")
    modelPV_files = glob.glob(model_path_pv)
    for file in modelPV_files:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except OSError as e:
            print(f"Error deleting {file}: {e}")
    
    modelh_files = glob.glob("client_*_kmeans_hybrid.pkl")
    scalerh_files = glob.glob("client_*_scaler_hybrid.pkl")
    for file in modelh_files:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except OSError as e:
            print(f"Error deleting {file}: {e}")

    #delete label encoder files        
    try:
        os.remove("label_encoder_description.joblib")
        os.remove("label_encoder_weather.joblib")
    except OSError as e:
        print(f"Error deleting label_encoder_description does not exist: {e}")
    # Delete scaler files  
    for file in scalerh_files:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except OSError as e:
            print(f"Error deleting {file}: {e}")
    print("Cleanup completed. Starting training...")


def make_predictions_consumption_lstm(agregated_df: pd.DataFrame, client_id: int):
    try:
        agregated_df = agregated_df.set_index(agregated_df.columns[1])
        agregated_df['Time'] = pd.to_datetime(agregated_df['Time'], dayfirst=True)

        def get_season(month):
            if month in [12, 1, 2]:
                return 1
            elif month in [3, 4, 5]:
                return 2
            elif month in [6, 7, 8]:
                return 3
            else:
                return 4

        def create_lag_features(df, target_col, max_lag=10):
            for lag in range(1, max_lag + 1):
                df[f'lag_{lag}'] = df[target_col].shift(lag)
            return df

        df = agregated_df.copy()
        df = df.reset_index()
        df['TimeOfDay'] = df['Time'].dt.strftime('%H%M%S')
        
        # Generate 25 hours of intervals (100 total)
        intervals = []
        # Standard 96 slots for a full day
        for hour in range(24):  # 0 to 23
            for minute in [0, 15, 30, 45]:
                time_obj = pd.Timestamp(f'{hour:02d}:{minute:02d}:00').time()
                intervals.append(time_obj)

        # Add the 4 next-day slots
        for minute in [15, 30, 45]:
            intervals.append(pd.Timestamp(f'00:{minute:02d}:00').time())
        intervals.append(pd.Timestamp('01:00:00').time())
        print(f"Making predictions for {len(intervals)} intervals using 96 trained models")

        predicted_all = pd.DataFrame()
        
        # Get dates for predictions
        today_date = datetime.datetime.today().strftime('%Y-%m-%d')
        tomorrow_date = (datetime.datetime.today() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')

        for i, interval in enumerate(intervals):
            # Determine which date to use (first 96 are today, last 4 are tomorrow)
            prediction_date = today_date if i < 96 else tomorrow_date
            
            # Use modulo to reuse the first 4 models for next day's 00:00-00:45
            model_index = i % 96
            
            ts_label = f"ts_{interval.strftime('%H%M%S')}"
            df_interval = df[df['TimeOfDay'] == interval.strftime('%H%M%S')].drop(columns=['TimeOfDay'])

            if df_interval.empty:
                print(f"Warning: No historical data found for {interval.strftime('%H:%M:%S')}")
                # Create a default prediction or skip
                predicted_all = pd.concat([predicted_all, pd.DataFrame(
                    {'Time': [f'{prediction_date} {interval.strftime("%H:%M:%S")}'], 'Predicted': [0]})],
                                          ignore_index=True)
                continue

            # Load correct model and scaler using model_index
            try:
                model = load_model(f"models//client_{client_id}_consumption_lstm_model_{model_index}_.keras")
                scaler = joblib.load(f"scalers//client_{client_id}_scaler_lstm_{model_index}.pkl")
            except FileNotFoundError:
                print(f"Model or scaler not found for index {model_index}")
                predicted_all = pd.concat([predicted_all, pd.DataFrame(
                    {'Time': [f'{prediction_date} {interval.strftime("%H:%M:%S")}'], 'Predicted': [0]})],
                                          ignore_index=True)
                continue

            # Extracting date-related features
            df_interval['Time'] = pd.to_datetime(df_interval['Time'])
            df_interval['Year'] = df_interval['Time'].dt.year
            df_interval['Month'] = df_interval['Time'].dt.month
            df_interval['DayOfWeek'] = df_interval['Time'].dt.dayofweek + 1
            df_interval['IsWeekend'] = (df_interval['DayOfWeek'] > 5).astype(int)
            df_interval['Season'] = df_interval['Month'].apply(get_season)

            # Extracting lag features
            df_interval = create_lag_features(df_interval, 'ConsumerPower1', max_lag=10)
            df_interval.dropna(inplace=True)

            if len(df_interval) < 10:  # Need at least sequence_length data
                print(f"Insufficient data for prediction at {interval.strftime('%H:%M:%S')}")
                predicted_all = pd.concat([predicted_all, pd.DataFrame(
                    {'Time': [f'{prediction_date} {interval.strftime("%H:%M:%S")}'], 'Predicted': [0]})],
                                          ignore_index=True)
                continue

            # Prepare the last sequence for prediction
            scaled_data = scaler.transform(df_interval.drop(columns=['Time']))
            sequence_length = 10
            last_sequence = scaled_data[-sequence_length:]

            # Predict the next value
            last_sequence = np.expand_dims(last_sequence, axis=0)
            predicted_value_scaled = model.predict(last_sequence, verbose=0)
            predicted_value = scaler.inverse_transform(
                np.concatenate(
                    (predicted_value_scaled, np.zeros((predicted_value_scaled.shape[0], scaled_data.shape[1] - 1))),
                    axis=1))[:, 0]

            # Add prediction with correct date
            predicted_all = pd.concat([predicted_all, pd.DataFrame(
                {'Time': [f'{prediction_date} {interval.strftime("%H:%M:%S")}'], 'Predicted': predicted_value})],
                                      ignore_index=True)

            # Progress indicator
            if (i + 1) % 25 == 0:
                print(f"Completed {i + 1}/100 predictions")

        print(f"Prediction completed. Generated {len(predicted_all)} predictions")
        print("Sample predictions:")
        print(predicted_all.head())
        print(predicted_all.tail())
        
        predicted_all.to_csv(f"client_{client_id}_consumption_preds.csv", index=False)
        return predicted_all
        
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def train_model_consumption_hybrid(agregated_df: pd.DataFrame, pred_df: pd.DataFrame, client_id):
    try:
        agregated_df["Time"] = pd.to_datetime(agregated_df['Time'])
        pred_df["Time"] = pd.to_datetime(pred_df['Time'])
        
        def load_data_training():
            df = pd.read_csv(f"client_{client_id}_data_agregated_cond.csv")
            df['Time'] = pd.to_datetime(df['Time'])
            return df

        def load_data_pred():
            df = pd.read_csv(f"client_{client_id}_consumption_preds.csv")
            df['Time'] = pd.to_datetime(df['Time'])
            return df

        df_training = agregated_df
        df_prediction = pred_df
        
        # Get the date of the first entry in df_prediction
        cutoff_date = df_prediction['Time'].iloc[0]
        cutoff_date = pd.to_datetime(cutoff_date)
        df_training = df_training[df_training['Time'] < cutoff_date]

        # Create a '25HourPeriod' column to group by 25-hour periods starting from midnight
        df_training['Date'] = df_training['Time'].dt.date
        df_training['TimeOfDay'] = df_training['Time'].dt.strftime('%H:%M:%S')
        df_training['Hour'] = df_training['Time'].dt.hour
        df_training['Minute'] = df_training['Time'].dt.minute
        
        # Create 25-hour periods: each period starts at 00:00 and includes next day's 00:00
        df_training['Period_Start_Date'] = df_training['Date']
        
        # Generate all possible 25-hour time slots (100 intervals of 15 minutes)
        time_slots_25h = []
        for hour in range(24):  # 0 to 23
            for minute in [0, 15, 30, 45]:
                time_obj = pd.Timestamp(f'{hour:02d}:{minute:02d}:00').time()
                time_slots_25h.append(time_obj)

        # Add the 4 next-day slots
        for minute in [15, 30, 45]:
            time_slots_25h.append(pd.Timestamp(f'00:{minute:02d}:00').time())
        time_slots_25h.append(pd.Timestamp('01:00:00').time())
        
        print(f"Total 25-hour time slots: {len(time_slots_25h)}")
        
        # Create extended training data with 25-hour periods
        extended_training_data = []
        
        unique_dates = sorted(df_training['Date'].unique())
        
        for i, date in enumerate(unique_dates[:-1]):  # Exclude last date to ensure we have next day data
            current_date = date
            next_date = unique_dates[i + 1] if i + 1 < len(unique_dates) else None
            
            if next_date is None:
                continue
                
            # Get data for current date (00:00 to 23:45)
            current_day_data = df_training[df_training['Date'] == current_date].copy()
            
            # Get data for next day's first 1h (00:00 to 01:00)
            next_day_first_hour = df_training[
                (df_training['Date'] == next_date) & 
                ((df_training['Hour'] == 0) | ((df_training['Hour'] == 1) & (df_training['Minute'] == 0)))
            ].copy()
            
            # Combine to create 25-hour period
            period_data = pd.concat([current_day_data, next_day_first_hour], ignore_index=True)
            period_data['Period_Start_Date'] = current_date
            
            extended_training_data.append(period_data)
        
        # Combine all 25-hour periods
        if extended_training_data:
            df_25h_training = pd.concat(extended_training_data, ignore_index=True)
        else:
            print("No valid 25-hour periods found in training data")
            return
        
        # Create pivot table with 25-hour periods as rows and time slots as columns
        pivot_df = df_25h_training.pivot_table(
            index='Period_Start_Date', 
            columns='TimeOfDay', 
            values='ConsumerPower1'
        )
        
        # Fill missing values
        pivot_df.fillna(method='ffill', axis=1, inplace=True)
        pivot_df.fillna(method='bfill', axis=1, inplace=True)
        pivot_df.fillna(0, inplace=True)
        
        print(f"Pivot table shape for 25-hour training: {pivot_df.shape}")
        print(f"Expected columns: {len(time_slots_25h)}")
        print(f"Actual columns: {len(pivot_df.columns)}")
        
        # Ensure we have all 100 time slots (pad with zeros if missing)
        for time_slot in time_slots_25h:
            if time_slot not in pivot_df.columns:
                pivot_df[time_slot] = 0
        
        # Reorder columns to match the expected 25-hour sequence
        pivot_df = pivot_df.reindex(columns=time_slots_25h, fill_value=0)
        
        print(f"Final pivot table shape: {pivot_df.shape}")
        
        # Standardize the data
        scaler = StandardScaler()
        pivot_df_scaled = scaler.fit_transform(pivot_df)
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=10, random_state=42)
        clusters = kmeans.fit_predict(pivot_df_scaled)
        pivot_df['Cluster'] = clusters
        
        # Save the scaler and kmeans model
        joblib.dump(scaler, f"client_{client_id}_scaler_hybrid.pkl")
        joblib.dump(kmeans, f"client_{client_id}_kmeans_hybrid.pkl")
        
        print(f"Training completed with {len(pivot_df)} 25-hour periods")
        print(f"Cluster distribution: {np.bincount(clusters)}")
        
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

def train_model_consumption_hybrid(agregated_df: pd.DataFrame, pred_df: pd.DataFrame, client_id):     
    try:
        agregated_df["Time"] = pd.to_datetime(agregated_df['Time'])
        pred_df["Time"] = pd.to_datetime(pred_df['Time'])
        
        def load_data_training():
            df = pd.read_csv(f"client_{client_id}_data_agregated_cond.csv")
            df['Time'] = pd.to_datetime(df['Time'])
            return df 
        
        def load_data_pred():
            df = pd.read_csv(f"client_{client_id}_consumption_preds.csv")
            df['Time'] = pd.to_datetime(df['Time'])
            return df 
        
        df_training = agregated_df
        df_prediction = pred_df
        
        # Get the date of the first entry in df_prediction
        cutoff_date = df_prediction['Time'].iloc[0]
        cutoff_date = pd.to_datetime(cutoff_date)
        df_training = df_training[df_training['Time'] < cutoff_date]
        
        # Sort by time to ensure proper ordering
        df_training = df_training.sort_values('Time').reset_index(drop=True)
        
        # Create features for each day including next day's first hour
        features_list = []
        dates_list = []
        
        # Get unique dates
        df_training['Date'] = df_training['Time'].dt.date
        unique_dates = sorted(df_training['Date'].unique())
        
        for i, current_date in enumerate(unique_dates):
            # Get current day's data
            current_day_data = df_training[df_training['Date'] == current_date].copy()
            
            if len(current_day_data) < 96:  # Skip if not enough data for full day
                continue
                
            # Create time features for current day (96 features)
            current_day_data = current_day_data.sort_values('Time')
            current_day_features = current_day_data['ConsumerPower1'].values[:96]  # Take first 96 readings
            
            # Get next day's first hour (4 features for 15-min intervals)
            next_day_features = []
            if i < len(unique_dates) - 1:  # If not the last date
                next_date = unique_dates[i + 1]
                next_day_data = df_training[df_training['Date'] == next_date].copy()
                
                if len(next_day_data) >= 4:  # Need at least 4 readings for first hour
                    next_day_data = next_day_data.sort_values('Time')
                    next_day_features = next_day_data['ConsumerPower1'].values[:4]
                else:
                    # If next day data is insufficient, use forward fill from last value
                    next_day_features = [current_day_features[-1]] * 4
            else:
                # For the last date, use forward fill from last value
                next_day_features = [current_day_features[-1]] * 4
            
            # Combine current day (96) + next day first hour (4) = 100 features
            combined_features = list(current_day_features) + list(next_day_features)
            
            if len(combined_features) == 100:  # Ensure we have exactly 100 features
                features_list.append(combined_features)
                dates_list.append(current_date)
        
        # Create DataFrame with 100 features
        feature_columns = []
        # Current day features (96)
        for hour in range(24):
            for quarter in ['00', '15', '30', '45']:
                feature_columns.append(f"{hour:02d}:{quarter}:00")
        
        # Next day first hour features (4)
        for quarter in ['00', '15', '30', '45']:
            feature_columns.append(f"24:{quarter}:00")  # Using 24: to indicate next day
        
        # Create the feature matrix
        feature_df = pd.DataFrame(features_list, columns=feature_columns, index=dates_list)
        
        # Handle any remaining NaN values
        feature_df.fillna(method='ffill', inplace=True)
        feature_df.fillna(method='bfill', inplace=True)
        
        # Standardize the data
        scaler = StandardScaler()
        feature_df_scaled = scaler.fit_transform(feature_df)
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=10, random_state=42)
        clusters = kmeans.fit_predict(feature_df_scaled)
        feature_df['Cluster'] = clusters
        
        # Save the scaler and kmeans model
        joblib.dump(scaler, f"client_{client_id}_scaler_hybrid.pkl")
        joblib.dump(kmeans, f"client_{client_id}_kmeans_hybrid.pkl")
        
        print(f"Training completed successfully with {len(feature_df)} samples and 100 features per sample")
        return feature_df, scaler, kmeans
        
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return None, None, None
                                        #consumption_preds
def make_predictions_consumption_hybrid(pred_df: pd.DataFrame, client_id: int):
    try:
        df = pred_df.copy()
        df['Time'] = pd.to_datetime(df['Time'])

        def load_models():
            # Load the scaler and kmeans model
            scaler = joblib.load(f"client_{client_id}_scaler_hybrid.pkl")
            kmeans = joblib.load(f"client_{client_id}_kmeans_hybrid.pkl")
            cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
            return cluster_centers, scaler, kmeans

        df_prediction = df.copy()
        cluster_centers, scaler, kmeans = load_models()

        # Rename column to match training data
        if 'Predicted' in df_prediction.columns:
            df_prediction = df_prediction.rename(columns={'Predicted': 'ConsumerPower1'})

        # Sort by time to ensure proper ordering
        df_prediction = df_prediction.sort_values('Time').reset_index(drop=True)
        
        # Create features for each day including next day's first hour (same as training)
        df_prediction['Date'] = df_prediction['Time'].dt.date
        unique_dates = sorted(df_prediction['Date'].unique())
        
        predictions_list = []
        
        for i, current_date in enumerate(unique_dates):
            # Get current day's data
            current_day_data = df_prediction[df_prediction['Date'] == current_date].copy()
            
            if len(current_day_data) < 96:  # Skip if not enough data for full day
                continue
                
            # Create time features for current day (96 features)
            current_day_data = current_day_data.sort_values('Time')
            current_day_features = current_day_data['ConsumerPower1'].values[:96]
            
            # Get next day's first hour (4 features)
            next_day_features = []
            if i < len(unique_dates) - 1:  # If not the last date
                next_date = unique_dates[i + 1]
                next_day_data = df_prediction[df_prediction['Date'] == next_date].copy()
                
                if len(next_day_data) >= 4:
                    next_day_data = next_day_data.sort_values('Time')
                    next_day_features = next_day_data['ConsumerPower1'].values[:4]
                else:
                    # Use forward fill from last value
                    next_day_features = [current_day_features[-1]] * 4
            else:
                # For the last date, use forward fill
                next_day_features = [current_day_features[-1]] * 4
            
            # Combine current day (96) + next day first hour (4) = 100 features
            combined_features = list(current_day_features) + list(next_day_features)
            
            if len(combined_features) == 100:
                predictions_list.append({
                    'date': current_date,
                    'features': combined_features,
                    'original_data': current_day_data
                })
        
        if not predictions_list:
            raise ValueError("No valid prediction data found")
        
        # Create feature matrix for prediction
        feature_matrix = np.array([pred['features'] for pred in predictions_list])
        
        # Standardize the prediction data using the same scaler fitted on training data
        feature_matrix_scaled = scaler.transform(feature_matrix)
        
        # Predict clusters
        predicted_clusters = kmeans.predict(feature_matrix_scaled)
        
        # Create final predictions DataFrame
        final_predictions = []
        
        for i, pred_data in enumerate(predictions_list):
            current_date = pred_data['date']
            cluster_idx = predicted_clusters[i]
            
            # Get all 100 cluster centroid values for this prediction
            cluster_centroid_full = cluster_centers[cluster_idx]  # All 100 values
            
            # Create timestamps for all 100 predictions (current day + 1 hour next day)
            base_time = pd.Timestamp(current_date)
            
            # Generate 100 timestamps (96 for current day + 4 for next day first hour)
            timestamps = []
            for j in range(100):
                timestamp = base_time + pd.Timedelta(minutes=j*15)  # 15-minute intervals
                timestamps.append(timestamp)
            
            # Get original LSTM predictions for current day (96 values)
            original_data = pred_data['original_data'].copy()
            original_data = original_data.sort_values('Time')
            lstm_values_current_day = original_data['ConsumerPower1'].values[:96]
            
            # For next day's first hour, we need to get LSTM predictions or use extrapolation
            lstm_values_next_hour = []
            if i < len(predictions_list) - 1:
                # Try to get next day's first 4 LSTM values
                next_date = unique_dates[unique_dates.index(current_date) + 1] if current_date in unique_dates else None
                if next_date:
                    next_day_data = df_prediction[df_prediction['Date'] == next_date].copy()
                    if len(next_day_data) >= 4:
                        next_day_data = next_day_data.sort_values('Time')
                        lstm_values_next_hour = next_day_data['ConsumerPower1'].values[:4]
                    else:
                        # Use extrapolation from current day's trend
                        lstm_values_next_hour = [lstm_values_current_day[-1]] * 4
                else:
                    lstm_values_next_hour = [lstm_values_current_day[-1]] * 4
            else:
                # Last day: extrapolate from current day
                lstm_values_next_hour = [lstm_values_current_day[-1]] * 4
            
            # Combine LSTM predictions: current day (96) + next hour (4) = 100
            all_lstm_predictions = list(lstm_values_current_day) + list(lstm_values_next_hour)
            
            # Create predictions for all 100 timestamps
            for j in range(100):
                final_predictions.append({
                    'Time': timestamps[j],
                    'lstm_predictions': all_lstm_predictions[j],
                    'cluster_centroid': cluster_centroid_full[j],
                    'Date': timestamps[j].date()
                })
        
        # Convert to DataFrame
        df_prediction_hybrid = pd.DataFrame(final_predictions)
        
        # Create hybrid predictions
        df_prediction_hybrid['hybrid_predictions'] = (
            df_prediction_hybrid['lstm_predictions'] + df_prediction_hybrid['cluster_centroid']
        ) / 2
        
        # Clean up columns
        df_prediction_hybrid = df_prediction_hybrid.drop(columns=['cluster_centroid', 'Date'])
        
        # Save predictions
        df_prediction_hybrid.to_csv(f"client_{client_id}_consumption_lstm_hybrid_preds.csv", index=False)
        
        # Create hourly aggregations
        hourly_predictions = (
            df_prediction_hybrid.set_index('Time')[['lstm_predictions', 'hybrid_predictions']]
            .resample('H').mean()
            .reset_index()
        )
        
        hourly_predictions.to_csv(f"client_{client_id}_consumption_lstm_hybrid_preds_hour.csv", index=False)
        
        print(f"Predictions completed successfully for {len(df_prediction_hybrid)} timestamps")
        return df_prediction_hybrid
        
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return None


# if __name__ == "__main__":
#     train_model_consumption_lstm()
#     make_predictions_consumption_lstm()
#     train_model_consumption_hybrid()
#     make_predictions_consumption_hybrid()