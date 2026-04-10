import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prep_data import run_prediction_sequnece
from tflite_runtime.interpreter import Interpreter

MODEL_PATH = os.getenv("MODEL_PATH")  # e.g. /models/model.tflite
TFLITE_THREADS = int(os.getenv("TFLITE_THREADS", "1"))

app = FastAPI(title="TFLite Inference API")

# Initialized on startup
interpreter = None
input_details = None
output_details = None


class PredictRequest(BaseModel):
    inputs: list[list[float]]


# @app.on_event("startup")
# def load_model() -> None:
#     global interpreter, input_details, output_details

#     if not MODEL_PATH:
#         raise RuntimeError("MODEL_PATH env var is not set")

#     model_file = Path(MODEL_PATH)
#     if not model_file.exists():
#         raise RuntimeError(f"Model file not found at: {MODEL_PATH}")

#     interpreter = Interpreter(model_path=str(model_file), num_threads=TFLITE_THREADS)
#     interpreter.allocate_tensors()

#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()


@app.get("/health")
def health():
    if not MODEL_PATH:
        return {"status": "error", "reason": "MODEL_PATH_not_set"}, 500
    if not Path(MODEL_PATH).exists():
        return {"status": "error", "reason": "model_file_missing", "model_path": MODEL_PATH}, 500
    return {"status": "is aight", "model_path": MODEL_PATH,"pid": os.getpid()}


@app.get("/predict")
def predict():
    df = run_prediction_sequnece()

    return {
        "predictions": df.to_dict(orient="records")
        }