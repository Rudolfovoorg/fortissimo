#!/bin/bash

# Submit the job and capture the job ID
JOB_SUBMIT_OUTPUT=$(sbatch train36ml.sh)
echo "$JOB_SUBMIT_OUTPUT"

# Extract job ID from sbatch output (assumes "Submitted batch job <id>")
JOB_ID=$(echo "$JOB_SUBMIT_OUTPUT" | awk '{print $4}')
START_TIME=$(date -u +"%Y-%m-%dT%H:%M:%S.%3NZ")

# Call the API to register the job in "Pending" state
curl -X 'POST' \
  '_HPC_JOB_API_URL_' \
  -H 'accept: */*' \
  -H 'X-API-Key: _API_KEY_' \
  -H 'Content-Type: application/json' \
  -d "{
  \"jobId\": $JOB_ID,
  \"jobDescription\": \"ML Training Job - ml_training\",
  \"startTime\": \"$START_TIME\",
  \"status\": \"Pending\",
  \"result\": \"Job submitted to SLURM queue\"
}" || echo "Warning: Failed to register pending job"
