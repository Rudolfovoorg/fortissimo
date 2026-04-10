import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import json

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)
strategy = tf.distribute.MirroredStrategy()
print("GPUs in use:", strategy.num_replicas_in_sync)


# ---- outputs dir ----
OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------- Data Loading and Preprocessing --------------------
def load_data():
    print("Current working directory:", os.getcwd())
    print("Files in ./data/:", os.listdir("./data"))
    file_path = "./data/LabtopConsumerPower.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    df['Time'] = pd.to_datetime(df['Time'])
    return df

def get_season(month):
    return 1 if month in [12, 1, 2] else 2 if month in [3, 4, 5] else 3 if month in [6, 7, 8] else 4

def create_lag_features(df, target_col, max_lag=30):
    for lag in range(1, max_lag + 1):
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    return df

def create_sequences(data, seq_len, target_index, prediction_steps):
    if len(data) <= seq_len + prediction_steps:
        return np.array([]), np.array([])
    X, y = [], []
    for i in range(seq_len, len(data) - prediction_steps):
        X.append(data[i - seq_len:i])
        y.append([data[i + j][target_index] for j in range(prediction_steps)])
    return np.array(X), np.array(y)

# Load and preprocess data
df = load_data()
df['Year'] = df['Time'].dt.year
df['Month'] = df['Time'].dt.month
df['DayOfWeek'] = df['Time'].dt.dayofweek + 1
df['IsWeekend'] = (df['DayOfWeek'] > 5).astype(int)
df['Season'] = df['Month'].apply(get_season)

df = create_lag_features(df, 'ConsumerPower1', max_lag=30)
df.dropna(inplace=True)
df.set_index('Time', inplace=True)

# Train-test split (last 3840 points as test)
test_size = 3840
train_df = df.iloc[:-test_size]
test_df = df.iloc[-test_size:]

target = 'ConsumerPower1'
target_index = train_df.columns.get_loc(target)

# Scale data
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_df)
scaled_test = scaler.transform(test_df)

# Prediction steps
prediction_steps = 96

# -------------------- Keras Tuner HyperModel --------------------
def build_model(hp):
    # dynamic search spaces
    seq_len = hp.Int('sequence_length', min_value=192, max_value=768, step=48)
    units = hp.Int('units', min_value=60, max_value=600, step=60)
    layers = hp.Int('layers', min_value=1, max_value=3, step=1)
    dropout_rate = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')

    model = Sequential()
    for i in range(layers):
        return_seq = i < layers - 1
        if i == 0:
            model.add(Bidirectional(
                LSTM(units, return_sequences=return_seq),
                input_shape=(seq_len, scaled_train.shape[1])
            ))
        else:
            model.add(Bidirectional(LSTM(units, return_sequences=return_seq)))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    model.add(Dense(prediction_steps))
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber())
    return model

# -------------------- Custom Tuner --------------------
class TimeSeriesTuner(kt.Hyperband):
    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        seq_len = hp.get('sequence_length')
        X_train_hp, y_train_hp = create_sequences(scaled_train, seq_len, target_index, prediction_steps)
        X_val_hp, y_val_hp = create_sequences(scaled_test, seq_len, target_index, prediction_steps)

        if len(X_train_hp) == 0 or len(X_val_hp) == 0:
            print(f"Skipping trial {trial.trial_id} due to insufficient data for seq_len={seq_len}")
            return
        if np.isnan(X_train_hp).any() or np.isinf(X_train_hp).any():
            print(f"Skipping trial {trial.trial_id}: NaN or Inf in training data")
            return

        return super().run_trial(
            trial,
            X_train_hp, y_train_hp,
            validation_data=(X_val_hp, y_val_hp),
            *args, **kwargs
        )

# -------------------- Instantiate and Run Tuner --------------------
tuner = TimeSeriesTuner(
    build_model,
    objective='val_loss',
    max_epochs=15,
    factor=3,
    directory='nas_results',
    project_name='bilstm_nas'
)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
tuner.search(epochs=15, callbacks=[stop_early])

# -------------------- Best Model & Test Evaluation --------------------
best_model = tuner.get_best_models(num_models=1)[0]
best_hps = tuner.get_best_hyperparameters(1)[0]

best_seq_len = best_hps.get('sequence_length')
X_test, y_test = create_sequences(scaled_test, best_seq_len, target_index, prediction_steps)

# Evaluate on test set
test_loss = float(best_model.evaluate(X_test, y_test, verbose=0))

# Predict all test sequences and compute MSE/MAE (scaled space)
y_pred = best_model.predict(X_test, verbose=0)
mse = float(tf.keras.metrics.mean_squared_error(y_test.flatten(), y_pred.flatten()).numpy())
mae = float(tf.keras.metrics.mean_absolute_error(y_test.flatten(), y_pred.flatten()).numpy())

# ---- save metrics to ONE file (JSON) ----
metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
metrics_payload = {
    "best_hyperparameters": best_hps.values,  # includes sequence_length/units/layers/dropout/learning_rate
    "test_loss_huber_scaled": test_loss,
    "mse_scaled": mse,
    "mae_scaled": mae
}
with open(metrics_path, "w") as f:
    json.dump(metrics_payload, f, indent=2)

# -------------------- Retrain Best Model to Visualize Loss (save PNG) --------------------
X_train_final, y_train_final = create_sequences(scaled_train, best_seq_len, target_index, prediction_steps)
X_val_final, y_val_final = create_sequences(scaled_test, best_seq_len, target_index, prediction_steps)

history = best_model.fit(
    X_train_final, y_train_final,
    epochs=15,
    validation_data=(X_val_final, y_val_final),
    verbose=0
)

plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss (Best Model)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curves.png"), dpi=150)
plt.close()

# -------------------- Visualize NAS Trial Performance (save PNG if available) --------------------
oracle_file = os.path.join('nas_results', 'bilstm_nas', 'oracle.json')
if os.path.exists(oracle_file):
    with open(oracle_file, 'r') as f:
        oracle_data = json.load(f)

    trial_losses = []
    if isinstance(oracle_data, dict) and 'trials' in oracle_data:  # only if this key exists
        for trial_id, trial_data in oracle_data['trials'].items():
            if 'score' in trial_data and trial_data['score'] is not None:
                trial_losses.append(trial_data['score'])

    if trial_losses:
        plt.figure(figsize=(8,5))
        plt.plot(trial_losses, marker='o')
        plt.xlabel('Trial')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss per NAS Trial')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "nas_trials.png"), dpi=150)
        plt.close()
