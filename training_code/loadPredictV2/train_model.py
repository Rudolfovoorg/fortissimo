import os
import time
import json
import gc
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from contextlib import redirect_stdout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

SEQ_LEN = 288
N_FEATURES = 32
PRED_STEPS = 96

def export_tflite_builtins_only(model, out_path="tfLiteModels/model.tflite"):
    seq_len = model.input_shape[1]
    n_features = model.input_shape[2]

    @tf.function(
        input_signature=[tf.TensorSpec([1, seq_len, n_features], tf.float32)]
    )
    def serving(x):
        return model(x)

    concrete = serving.get_concrete_function()

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete])
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.experimental_enable_resource_variables = True

    tflite_model = converter.convert()

    with open(out_path, "wb") as f:
        f.write(tflite_model)

    return out_path

def train_model(clientId:int,train_data:pd.DataFrame):
    # Drop stale graphs/allocations from previous training cycles in this long-running process.
    tf.keras.backend.clear_session()
    gc.collect()

    now= datetime.now().replace(second=0,microsecond=0).strftime("%Y-%m-%dT%H-%M-%S")
    DATA_FILE = "features_training_dom_mm.csv"
    MODEL_FILE = f"modelsV2/bilstm_{clientId}_dommm_{now}.keras"
    SCALER_X_FILE = f"modelsV2/scaler_{clientId}_X_{now}.pkl"
    SCALER_Y_FILE = f"modelsV2/scaler_{clientId}_y_{now}.pkl"
    FEATURE_COLS_FILE = "modelsV2/feature_columns.pkl"
    TRAIN_LOG_FILE = "modelsV2/train_log.txt"
    HISTORY_FILE = f"modelsV2/history_{clientId}.csv"
    METRICS_JSON_FILE = f"modelsV2/metrics_{clientId}.json"
    METRICS_CSV_FILE = f"modelsV2/metrics_{clientId}.csv"
    COMPLEXITY_JSON_FILE = "modelsV2/training_complexity.json"
    SEQ_LEN = 288
    PRED_STEPS = 96
    EPOCHS = 15
    BATCH_SIZE = 32


    def create_sequences(X_arr, y_arr, seq_len, prediction_steps):
        X_seq, y_seq = [], []
        for i in range(seq_len, len(X_arr) - prediction_steps):
            X_seq.append(X_arr[i - seq_len:i])
            y_seq.append(y_arr[i:i + prediction_steps])
        return np.array(X_seq), np.array(y_seq)


    def build_bilstm_model(input_shape, prediction_steps=96):
        model = Sequential()
        model.add(Bidirectional(LSTM(units=60, return_sequences=False, input_shape=input_shape)))
        model.add(Dropout(0.2))
        model.add(Dense(units=prediction_steps))
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
        return model

    def build_lstm_model(input_shape, prediction_steps=96):
        model = Sequential([
            Input(shape=input_shape),

            LSTM(60, return_sequences=True),
            LSTM(30, return_sequences=False),

            Dropout(0.2),
            Dense(prediction_steps),
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
        return model
    
    def smape_custom_100(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return np.mean(
            100.0 * np.abs(y_true - y_pred) /
            (np.abs(y_true) + np.abs(y_pred) + 1e-8)
        )


    def wmape(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        denom = np.sum(np.abs(y_true))
        if denom == 0:
            return np.nan
        return 100.0 * np.sum(np.abs(y_true - y_pred)) / (denom + 1e-8)


    def save_training_complexity(model, X_train_seq, y_train_seq):
        trainable_params = int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))
        non_trainable_params = int(np.sum([np.prod(v.shape) for v in model.non_trainable_variables]))
        total_params = trainable_params + non_trainable_params
        param_memory_mb = total_params * 4 / (1024**2)

        complexity = {
            "X_train_seq_shape": list(X_train_seq.shape),
            "y_train_seq_shape": list(y_train_seq.shape),
            "seq_len": int(SEQ_LEN),
            "prediction_steps": int(PRED_STEPS),
            "trainable_params": trainable_params,
            "non_trainable_params": non_trainable_params,
            "total_params": total_params,
            "approx_param_memory_mb": float(param_memory_mb),
        }

        with open(COMPLEXITY_JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(complexity, f, indent=2)



    # -----------------------
    # Load data
    # -----------------------
    df = train_data.copy() #pd.read_csv(DATA_FILE)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    data_prepared = df.drop(columns=["Timestamp", "Unnamed: 0"], errors="ignore")

    X = data_prepared.drop(columns=["LoadEnergyCalculated_15"])
    y = data_prepared["LoadEnergyCalculated_15"]

    # -----------------------
    # Split
    # -----------------------
    train_size = int(len(data_prepared) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # -----------------------
    # Scale
    # -----------------------
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train).astype(np.float32)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).astype(np.float32)

    X_test_scaled = scaler_X.transform(X_test).astype(np.float32)
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).astype(np.float32)

    # -----------------------
    # Sequences
    # -----------------------
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, SEQ_LEN, PRED_STEPS)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, SEQ_LEN, PRED_STEPS)

    # -----------------------
    # Model
    # -----------------------
    model = build_bilstm_model((X_train_seq.shape[1], X_train_seq.shape[2]), prediction_steps=PRED_STEPS)
    #model = build_lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]), prediction_steps=PRED_STEPS) # for tflite 
    
    # -----------------------
    # Train (log verbose to file) + timing
    # -----------------------
    start = time.perf_counter()
    with open(TRAIN_LOG_FILE, "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            history = model.fit(
                X_train_seq, y_train_seq,
                epochs=EPOCHS,
                validation_data=(X_test_seq, y_test_seq),
                batch_size=BATCH_SIZE,
                verbose=1
            )
    train_time_sec = time.perf_counter() - start

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(HISTORY_FILE, index=False)

    # -----------------------
    # Evaluate on test
    # -----------------------
    y_pred_scaled = model.predict(X_test_seq)  # (n_samples, 96)
    y_pred_inv = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(y_pred_scaled.shape)

    y_test_2d = y_test_seq.squeeze(-1)  # (n_samples, 96)
    y_test_inv = scaler_y.inverse_transform(y_test_2d.reshape(-1, 1)).reshape(y_test_2d.shape)

    y_true_flat = y_test_inv.flatten()
    y_pred_flat = y_pred_inv.flatten()

    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    SMAPE = smape_custom_100(y_true_flat, y_pred_flat)
    WMAPE = wmape(y_true_flat, y_pred_flat)

    metrics = {
        "TrainEnd":datetime.now().isoformat(),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "SMAPE_percent": float(SMAPE),
        "WMAPE_percent": float(WMAPE),
        "seq_len": int(SEQ_LEN),
        "prediction_steps": int(PRED_STEPS),
        "epochs": int(len(hist_df)),
        "batch_size": int(BATCH_SIZE),
        "n_rows_total": int(len(data_prepared)),
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
        "n_features": int(X.shape[1]),
        "training_time_seconds": float(train_time_sec),
        "was_in_use":True,
        "time_per_epoch_seconds": float(train_time_sec / max(1, len(hist_df))),
    }

    if os.path.exists(METRICS_JSON_FILE):
        try:
            with open(METRICS_JSON_FILE, "r", encoding="utf-8") as f:
                existing_metrics = json.load(f)

            
            # If file accidentally contains a single dict, convert to list
            if isinstance(existing_metrics, dict):
                existing_metrics = [existing_metrics]

        except json.JSONDecodeError:
            existing_metrics = []
    else:
        existing_metrics = []
    def key_fn(m: dict):
        s = m.get("TrainEnd", "")
        try:
            return datetime.fromisoformat(s)
        except Exception:
            return datetime.min
    existing_metrics=sorted(existing_metrics, key=key_fn, reverse=True)
    def get_latest_in_use(metrics_list):

        for m in metrics_list:              
            if m.get("was_in_use") is True:
                return m
        return None
    last_in_use_metric=get_latest_in_use(existing_metrics)
    def is_new_model_better(new, old) ->bool:
        if (old is None):
            return True
        return (
            new["WMAPE_percent"] < old["WMAPE_percent"] * 0.99
            and new["RMSE"] <= old["RMSE"] * 1.01
        )
    if(is_new_model_better(metrics,last_in_use_metric)):
        model.save(MODEL_FILE)
        #export_tflite_builtins_only(model, "tfLiteModels/model.tflite")

        joblib.dump(scaler_X, SCALER_X_FILE)
        joblib.dump(scaler_y, SCALER_Y_FILE)
        joblib.dump(list(X.columns), FEATURE_COLS_FILE)
    else:
        print(f"new model wmape is higher so the {now} trained model was not saved")
        metrics["was_in_use"]=False
    existing_metrics.append(metrics)

    # Save back
    with open(METRICS_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(existing_metrics, f, indent=2)
    pd.DataFrame([metrics]).to_csv(METRICS_CSV_FILE, index=False)

    # -----------------------
    # Save artifacts for serving
    # -----------------------

    # Save complexity report
    save_training_complexity(model, X_train_seq, y_train_seq)

    print("Training complete. Artifacts saved:")

    # Best-effort cleanup so resident memory can fall between daily runs.
    del model, history, hist_df
    del X_train_seq, y_train_seq, X_test_seq, y_test_seq
    del X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled
    del y_pred_scaled, y_pred_inv, y_test_2d, y_test_inv, y_true_flat, y_pred_flat
    tf.keras.backend.clear_session()
    gc.collect()
