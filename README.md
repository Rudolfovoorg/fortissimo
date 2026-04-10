# fortissimo

## Data and Code Availability

This repository includes instructions for:
- accessing the original data,
- running the analysis/training pipeline,
- reproducing the reported results.

## Repository Structure
- `training_code/`: model training, data preparation, DB integration, scheduled pipeline logic.
- `multiClusterSync/apps/tflite-infer/base/`: FastAPI inference service (TFLite runtime).
- `multiClusterSync/apps/tflite-infer/overlays/`: per-building model images and K8s overlays.
- `multiClusterSync/.github/workflows/`: CI workflows for building/pushing images.

## 1) Local Python Setup
From repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

PowerShell equivalent:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
```

Install dependencies for training and pipeline scripts:

```bash
pip install numpy pandas joblib requests pytz matplotlib scikit-learn sqlalchemy psycopg2-binary tensorflow keras xgboost pyotp paramiko streamlit altair
```

## 2) Configure Environment Variables
Set placeholders with real values before running:

```bash
export CONNECTION_STRING="_CONNECTION_STRING_"
export ENVIRONMENT="development"
export API_KEY="_API_KEY_"
export DISTRIBUTE_API_URL="_DISTRIBUTE_API_URL_"
export ENERGYDATA_API_URL="_ENERGYDATA_API_URL_"
export MEASUREMENTS_API_URL="_MEASUREMENTS_API_URL_"
```

PowerShell equivalent:

```powershell
$env:CONNECTION_STRING="_CONNECTION_STRING_"
$env:ENVIRONMENT="development"
$env:API_KEY="_API_KEY_"
$env:DISTRIBUTE_API_URL="_DISTRIBUTE_API_URL_"
$env:ENERGYDATA_API_URL="_ENERGYDATA_API_URL_"
$env:MEASUREMENTS_API_URL="_MEASUREMENTS_API_URL_"
```

## 3) Run Training / Analysis Code

### 3.1 Full orchestrator loop (`training_code/main.py`)
```bash
cd training_code
python main.py
```

### 3.2 Run LoadPredictV2 pipeline only
```bash
cd training_code/loadPredictV2
python test.py
```

### 3.3 Optional folders for artifacts
Some scripts write model/forecast artifacts. Create once:

```bash
mkdir -p training_code/loadPredictV2/modelsV2
mkdir -p training_code/loadPredictV2/forecastV2
mkdir -p training_code/loadPredictV2/csvFiles
```

## 4) Run TFLite Serving API Locally
From repository root:

```bash
cd multiClusterSync/apps/tflite-infer/base
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PowerShell equivalent:

```powershell
cd multiClusterSync/apps/tflite-infer/base
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set model artifact paths:

```bash
export MODEL_PATH="/absolute/path/to/model.tflite"
export SCALERX_PATH="/absolute/path/to/scalerx.pkl"
export SCALERY_PATH="/absolute/path/to/scalery.pkl"
export FEATURE_COLUMNS="/absolute/path/to/featurecol.pkl"
export TFLITE_THREADS="2"
export API_KEY="_API_KEY_"
export ENERGYDATA_API_URL="_ENERGYDATA_API_URL_"
export MEASUREMENTS_API_URL="_MEASUREMENTS_API_URL_"
```

PowerShell equivalent:

```powershell
$env:MODEL_PATH="C:\\absolute\\path\\to\\model.tflite"
$env:SCALERX_PATH="C:\\absolute\\path\\to\\scalerx.pkl"
$env:SCALERY_PATH="C:\\absolute\\path\\to\\scalery.pkl"
$env:FEATURE_COLUMNS="C:\\absolute\\path\\to\\featurecol.pkl"
$env:TFLITE_THREADS="2"
$env:API_KEY="_API_KEY_"
$env:ENERGYDATA_API_URL="_ENERGYDATA_API_URL_"
$env:MEASUREMENTS_API_URL="_MEASUREMENTS_API_URL_"
```

Start API:

```bash
uvicorn app:app --host 0.0.0.0 --port 8050
```

Test endpoints:

```bash
curl http://localhost:8050/health
curl http://localhost:8050/predict
```

## 5) Build Docker Images Manually
Server image:

```bash
cd multiClusterSync
docker build -f apps/tflite-infer/base/Dockerfile -t _DOCKER_IMAGE_:server-local apps/tflite-infer/base
```

Model image example (building-a):

```bash
docker build -f apps/tflite-infer/overlays/building-a/Dockerfile -t _DOCKER_IMAGE_:building-a-local apps/tflite-infer/overlays/building-a
```

## 6) Deploy to Kubernetes (Kustomize)
Create registry pull secret once (namespace `default`):

```bash
kubectl create secret docker-registry regcred \
  --docker-server=https://index.docker.io/v1/ \
  --docker-username=_DOCKERHUB_USERNAME_ \
  --docker-password=_DOCKERHUB_TOKEN_ \
  --namespace default
```

Deploy overlay:

```bash
cd multiClusterSync/apps/tflite-infer
kubectl apply -k overlays/building-a
# or
kubectl apply -k overlays/building-b
```

Check rollout and service:

```bash
kubectl get pods -n default
kubectl logs deploy/tflite-infer -n default
kubectl get svc tflite-infer -n default
```

## 7) CI Workflows
- `multiClusterSync/.github/workflows/main.yml`: builds/pushes model overlay images and updates overlay tags.
- `multiClusterSync/.github/workflows/build-server.yml`: builds/pushes server image and updates server tags in overlays.

Both workflows require repository secrets:
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`
