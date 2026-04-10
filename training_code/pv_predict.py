import numpy as np
import tensorflow as tf
import gc
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.models import load_model
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings("ignore")
import random
random.seed(42)
tf.random.set_seed(42)
np.random.seed(42)
# from datetime import datetime
import os
import joblib
import keras
import datetime
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Read the csv file
#os.chdir("_PROJECT_DIR_")

def train_pv_model(df_selected,df_weather_expanded,train_test_data,clientId):
    tf.keras.backend.clear_session()
    gc.collect()
    df_selected.to_csv("df_selected.csv")
    df_weather_expanded.to_csv("df_weather_expanded.csv")
    train_test_data.to_csv("train_test_data.csv")
    df:pd.DataFrame = df_selected
    #df.columns
    #print(df)  # 7 columns, including the Date.
    try:
        # Separate dates for future plotting
        #df.reset_index()
        #train_dates = pd.to_datetime(df['Time'])
        #print(train_dates.tail(15))  # Check last few dates.

        # Variables for training
        feature_cols = ['Temperature', 'Clouds', 'Visibility', 'Humidity', 'Pressure', 'WindSpeed', 
                'hour_sin', 'hour_cos', 'sin_day_of_month', 'cos_day_of_month', 
                'daylight_duration', 'Weather_encoded', 'WeatherDescription_encoded']
        cols = feature_cols
        # Date and volume columns are not used in training.
        print(cols)  # ['Open', 'High', 'Low', 'Close', 'Adj Close']

        # New dataframe with only training data - 5 columns
        df_for_training = df[cols].astype(float)

        # df_for_plot=df_for_training.tail(5000)
        # df_for_plot.plot.line()

        # LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
        # normalize the dataset
        '''
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler_target = MinMaxScaler(feature_range=(0, 1))

        joblib.dump(scaler, 'SolarPowerPrediction/production/scaler_X.pkl')
        joblib.dump(scaler_target, 'SolarPowerPrediction/production/scaler_y.pkl')

        df_for_training_scaled = scaler.fit_transform(df_for_training)
        target_scaled = scaler_target.fit_transform(df[['SolarPower']])
        '''
        # As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features.
        # In this example, the n_features is 5. We will make timesteps = 14 (past days data used for training).

        scaler_X_path = f'scalers//client_{clientId}_pv_scaler_X.pkl'
        scaler_y_path = f'scalers//client_{clientId}_pv_scaler_y.pkl'

        # Check if the scaler for the features exists
        if os.path.exists(scaler_X_path):
            # Load the existing scaler
            scaler = joblib.load(scaler_X_path)
            print('Load the existing scaler_X')
        else:
            # If the file does not exist, create and fit a new scaler
            print('create and fit a new scaler for the features, scaler_X')
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(df_for_training)  # Fit the scaler to the data
            # Save the scaler for future use
            joblib.dump(scaler, scaler_X_path)

        # Apply the scaler to your data
        df_for_training_scaled = scaler.transform(df_for_training)

        # Check if the scaler for the target exists
        if os.path.exists(scaler_y_path):
            # Load the existing scaler
            scaler_target = joblib.load(scaler_y_path)
            print('Load the existing scaler_y')
        else:
            # If the file does not exist, create and fit a new scaler for the target
            print('create and fit a new scaler for the target, scaler_y')
            scaler_target = MinMaxScaler(feature_range=(0, 1))
            scaler_target.fit(df[['SolarPower']])  # Fit the scaler to the target data
            # Save the scaler for future use
            joblib.dump(scaler_target, scaler_y_path)

        # Apply the scaler to the target
        target_scaled = scaler_target.transform(df[['SolarPower']])

        
        def create_sequences(X, y, sequence_length):
            X_seq, y_seq = [], []
            for i in range(len(X) - sequence_length):
                X_seq.append(X[i:(i + sequence_length)])
                y_seq.append(y[i + sequence_length])
            return np.array(X_seq), np.array(y_seq)

        # Choose a sequence length
        sequence_length = 24  # Example: Representing 4 days of hourly data

        # Create sequences
        x_seq, y_seq = create_sequences(df_for_training_scaled, target_scaled.flatten(), sequence_length)

        # Dataset splitting
        SPLIT = 0.85
        X_train = x_seq[:int(SPLIT * len(x_seq))]
        y_train = y_seq[:int(SPLIT * len(y_seq))]
        X_test = x_seq[int(SPLIT * len(x_seq)):]
        y_test = y_seq[int(SPLIT * len(y_seq)):]


        today = datetime.datetime.now().replace(hour=0,minute=0,second=0,microsecond=0)
        model_path = f'models//client_{clientId}_solar_power_lstm_model.h5'

        # Check if the model file already exists
        if not os.path.exists(model_path):
            # Model does not exist, so build and fit the model
            print('Model does not exist, so build and fit the model...')

            n_features = X_train.shape[2]
            sequence_length = X_train.shape[1]  # should be 24

            multivariate_lstm = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(sequence_length, n_features)),
                tf.keras.layers.LSTM(
                    units=160,
                    activation="tanh",
                    recurrent_activation="sigmoid",
                    unroll=True,          # <<< key for pure TFLite attempts
                    use_bias=True
                ),
                tf.keras.layers.Dense(1)
            ])

            multivariate_lstm.compile(loss='mean_squared_error', optimizer='adam')
            multivariate_lstm.summary()

            # Fit the model
            early_stopping_lstm = EarlyStopping(monitor='val_loss', patience=5)
            history = multivariate_lstm.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, callbacks=[early_stopping_lstm])

            # Save the model
            multivariate_lstm.save(model_path)
            print("Model trained and saved.")

            today = today

            #plt.plot(history.history['loss'])
            #plt.xlabel('Epochs')
            #plt.ylabel('Valuation loss')
            # plt.plot(history.history['val_loss'], label='Validation loss')
            # plt.legend()
            filename = f'training_loss_lstm_{today}.png'
            #plt.savefig(filename)

            # Predicting on test ds and evaluation...
            print("Predicting on test ds and print evaluation metriccs in kWh...")

            predictions_lstm = multivariate_lstm.predict(X_test)
            predictions_lstm_inversed = scaler_target.inverse_transform(predictions_lstm)
            y_test_reshaped = y_test.reshape(-1, 1)  # Ensure y_test_seq is a 2D array
            y_test_inversed = scaler_target.inverse_transform(y_test_reshaped)

            def convert_w_to_kwh(values):
                return ((values) * 0.25 / 1000)


            # Convert y_true and y_pred to kWh
            y_true_kwh = convert_w_to_kwh(y_test_inversed)
            y_predicted_kwh = convert_w_to_kwh(predictions_lstm_inversed)

            #plt.plot(y_test_inversed)
            #plt.plot(predictions_lstm_inversed)
            #plt.show()

            # evaluating...

            # Calculate metrics in kWh for the LSTM model
            mse_lstm_kwh = mean_squared_error(y_true_kwh, y_predicted_kwh)
            mae_lstm_kwh = mean_absolute_error(y_true_kwh, y_predicted_kwh)
            rmse_lstm_kwh = np.sqrt(mse_lstm_kwh)
            r2_lstm_kwh = r2_score(y_true_kwh, y_predicted_kwh)

            # Printing the metrics for the LSTM model in kWh
            print("LSTM Model Performance (in kWh):")
            print(f"MSE (kWh): {mse_lstm_kwh}")
            print(f"MAE (kWh): {mae_lstm_kwh}")
            print(f"RMSE (kWh): {rmse_lstm_kwh}")
            print(f"R^2: {r2_lstm_kwh}\n")

        else:
            # Model already exists, skip building and fitting
            print("Model and val loss plot already exist. Skipping model building, fitting and evaluation on the test set.")


        print('Apply model to the serving dataset...')

        df_serving =df_weather_expanded
        print(len(df_serving))  # 7 columns, including the Date.

        # Separate dates for future plotting
        # serving_dates = pd.to_datetime(df_serving['Unnamed: 0'])

        print('Check last few dates...')
        #(serving_dates.tail(15))  # Check last few dates.


        #cols_serving = list(df_serving)[1:14]

        # New dataframe with only training data - 5 columns
        df_for_prediction = df_serving[cols].astype(float)

        df_for_training_end = df_for_training.iloc[-1152:,]

        df_for_prediction = pd.concat([df_for_training_end,df_for_prediction], axis=0)

        scaler = joblib.load(scaler_X_path)
        df_for_prediction_scaled = scaler.transform(df_for_prediction) # change

        dummy_targets = np.zeros(len(df_for_prediction_scaled))
        df_for_prediction_seq, _ = create_sequences(df_for_prediction_scaled, dummy_targets, sequence_length)
        multivariate_lstm = load_model(model_path)
        print("Model loaded successfully.")

        serving_preds_lstm = multivariate_lstm.predict(df_for_prediction_seq)
        serving_preds_lstm_inversed = scaler_target.inverse_transform(serving_preds_lstm)
        print(len(serving_preds_lstm_inversed))
        print(serving_preds_lstm_inversed.shape)

        difference = len(serving_preds_lstm_inversed) - len(df_serving)
        serving_preds_lstm_inversed_from_tomorrow = serving_preds_lstm_inversed[difference:]
        print(len(serving_preds_lstm_inversed_from_tomorrow))

        serving_preds_lstm_inversed_from_tomorrow_series = pd.Series(serving_preds_lstm_inversed_from_tomorrow.flatten(), index=df_serving.index)

        # Create a new DataFrame with the selected column from the existing DataFrame and the new series
        predicted_df = pd.DataFrame({
            'Time':  df_serving.index,
            #'is_daylight': wf['is_daylight']
            'SolarPower_pred_lstm': serving_preds_lstm_inversed_from_tomorrow_series
        })
        print(predicted_df)

        #We want to extract again is_daylight, but the only table that hs this information is the original training ds from sql
        original_data_training = train_test_data

        original_data_training['SunriseDT'] = pd.to_datetime(original_data_training['SunriseDT'])
        original_data_training['SunsetDT'] = pd.to_datetime(original_data_training['SunsetDT'])

        last_sunrise_time = original_data_training.iloc[-1]['SunriseDT'].time()
        last_sunset_time = original_data_training.iloc[-1]['SunsetDT'].time()

        # Ensure the 'existing_column' is of datetime type for proper plotting
        predicted_df['Time'] = pd.to_datetime(predicted_df['Time'])

        # Use per-timestamp daylight bounds to avoid seasonal clipping.
        predicted_df['Time_predicted'] = today
        predicted_df['SunriseDT'] = pd.to_datetime(df_serving['sunrise_dt']).values
        predicted_df['SunsetDT'] = pd.to_datetime(df_serving['sunset_dt']).values


        predicted_df['SolarPower_pred_lstm'] = predicted_df.apply(
            lambda row: 0 if (
                row['SolarPower_pred_lstm'] < 0
                or row['Time'].time() < row['SunriseDT'].time()
                or row['Time'].time() > row['SunsetDT'].time()
            )
            else row['SolarPower_pred_lstm'],
            axis=1
        )

        # Display the modified DataFrame
        print(predicted_df)
        predicted_df['SolarPower_pred_lstm'].describe()

        # Specify the path where you want to save the CSV file
        
        predictions_file_path = 'predicted_pv_today.csv'

        # Save the modified DataFrame to a CSV file
        predicted_df.to_csv(predictions_file_path, index=False)

        print(f"DataFrame saved successfully to {predictions_file_path}")

        # Sorting the DataFrame by the date might be necessary to get a chronological plot
        predicted_df.sort_values('Time', inplace=True)
        return predicted_df
    except Exception as ex:
        print(ex)
        return None
    finally:
        # Keep long-lived process memory stable between daily runs.
        tf.keras.backend.clear_session()
        gc.collect()
    # print('Plotting the predictions...')
    # plt.figure(figsize=(10, 6))
    # plt.plot(predicted_df['Time'], predicted_df['SolarPower_pred_lstm'], marker='o', linestyle='-')
    # plt.title('Solar Power predicted values over time (W)')
    # plt.xlabel('Date')
    # plt.ylabel('Value')

    # # Customizing x-axis to show every 10th date value
    # ax = plt.gca()  # Get the current Axes instance

    # # Assuming your dataframe is not huge, this method is okay. For very large dataframes, consider down-sampling your data for plotting.
    # # Generate a list of all date labels, then select every 10th one
    # all_dates = predicted_df['Time'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()  # Convert to string if needed
    # sparse_dates = all_dates[::10]  # Select every 10th date

    # # Set the x-ticks to be every 10th label from the sorted dates
    # sparse_dates_positions = predicted_df['Time'][::10]

    # # Set the x-ticks to show every 10th label from the sorted dates, including time
    # ax.set_xticks(sparse_dates_positions)
    # ax.set_xticklabels(sparse_dates, rotation=45, ha="right")
    # # Rotate dates for better readability
    # plt.tight_layout()
    # plt.show()
