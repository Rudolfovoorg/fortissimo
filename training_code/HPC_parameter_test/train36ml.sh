#!/bin/bash
#SBATCH --job-name=ml_training
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=10G
#SBATCH --time=00:15:00 
#SBATCH --output=ml_training_%j.out
#SBATCH --error=ml_training_%j.err


cd $SLURM_SUBMIT_DIR

echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
curl -X 'PUT' \
    '_HPC_JOB_API_URL_' \
    -H 'accept: */*' \
    -H 'X-API-Key: API_KEY' \
    -H 'Content-Type: application/json' \
    -d "{
    \"jobId\": $SLURM_JOB_ID,
    \"status\": \"Runing"
  }" || echo "Warning: Failed to update API"

# Initialize variables
START_TIME=$(date -u +"%Y-%m-%dT%H:%M:%S.%3NZ")
TRAINING_EXIT_CODE=99  # Default to failure
JOB_STATUS="Failed"
JOB_RESULT="Job terminated unexpectedly"

echo "Starting ML training..."
   
   
cleanup() {
  END_TIME=$(date -u +"%Y-%m-%dT%H:%M:%S.%3NZ")
  if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    JOB_STATUS="Success"
    JOB_RESULT="Training completed successfully"
  else
    JOB_STATUS="Failed"
    JOB_RESULT="Training failed or terminated"
  fi

  echo "Updating API with status: $JOB_STATUS"

  curl -X 'PUT' \
    '_HPC_JOB_API_URL_' \
    -H 'accept: */*' \
    -H 'X-API-Key: API_KEY' \
    -H 'Content-Type: application/json' \
    -d "{
    \"jobId\": $SLURM_JOB_ID,
    \"jobDescription\": \"ML Training Job - $SLURM_JOB_NAME\",
    \"startTime\": \"$START_TIME\",
    \"endTime\": \"$END_TIME\",
    \"status\": \"$JOB_STATUS\",
    \"result\": \"$JOB_RESULT\"
  }" || echo "Warning: Failed to update API"
}
trap cleanup EXIT




singularity exec --nv tensorflow_2.15.0-gpu.sif python3 run_train.py
TRAINING_EXIT_CODE=$?

# Update API with completion status


echo "Job completed at $(date)"
