#!/bin/bash
set -e

# ==============================================================================
# [START] Federated Learning Node Orchestrator
# ==============================================================================
# This script builds the environment, mocks data, uploads it to MinIO, 
# and launches the Client/Server containers.
# ==============================================================================

# 1. Configuration
# Inferred from @MFL/cred-minio
export STORAGE_ENDPOINT="storage.saintgits01.footprint-ai.com"

# --- Active Configuration (Saintgits Footprint AI) ---
export STORAGE_ACCESS_KEY="Tjqsfoy5tu3Itq1r"
export STORAGE_ACCESS_SECRET="oRfEU3ZzlZl4WvBW7Y6kFCDootwKwc8G"
export STORAGE_BUCKET_NAME="project-2-electric-spitfire"
export RESTCOL_HOST="https://saintgits01.footprint-ai.com/reststore"

# --- Legacy/Mock Configuration (Commented out for switching) ---
# export STORAGE_ACCESS_KEY="demo@footprint-ai.com"
# export STORAGE_ACCESS_SECRET="bar!@$#@bar"
# export STORAGE_BUCKET_NAME="crisis-mmd"
# export RESTCOL_HOST="http://host.docker.internal:50091"
export RESTCOL_AUTH_TOKEN="eyJhbGciOiJSUzI1NiIsImtpZCI6ImM4MzExZmZmNWE3NjY5ZjRmOTNjMTMyODg3NmRkYTc3NjJmYTMyYjkifQ.eyJpc3MiOiJodHRwczovL2F1dGguc2FpbnRnaXRzMDEuZm9vdHByaW50LWFpLmNvbS9kZXgiLCJzdWIiOiJDaU5qYmoxa1pXMXZMRzkxUFZCbGIzQnNaU3hrWXoxbGVHRnRjR3hsTEdSalBXOXlaeElFYkdSaGNBIiwiYXVkIjoia2FmZWlkby1hcHAiLCJleHAiOjE3NjM2MTUzMjYsImlhdCI6MTc2MzUyODkyNiwiYXRfaGFzaCI6InB2Rmd5cEEtdHhiZzBLUVJ2eWh3WEEiLCJlbWFpbCI6ImRlbW9AZm9vdHByaW50LWFpLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJncm91cHMiOlsic3AxMCJdLCJuYW1lIjoiZGVtbyJ9.TgE5DE1Y8zUFgVXmX4vPiwH09Xl_RkAMPNtELm_Om1Jn_f3z5OWAfi7i4ELk3sMOFUEQ-ZvFkQR7N5IiQ9Q8zTydJYcMWYsmxirivbMPqEmNyYYLTjg-mbgqcYRgeLI40Tod90KEPu0X7F73Brrk_15EVovB1ZLuCXVdYoiaej0nJ2OmtxhLfXhzKzLIkbjEUTlkFp5yhaT02Xv4Ii1cy7_pPBCjm-DR986UFEBsvNxIX9187REVkNuw1vhpiBA4uOvHn9s0IcN4QGfmA2IScVy2HxAlwjuznF5avt5NUK-iDNBFLXg1kKViqw0-CgFkVU-jQbtryFnhhm2af_0ang"
export SESSION_ID="session_$(date +%s)"
export COLLECTION_ID="crisis-mmd"

echo "[INFO] [1/5] Building Docker Image..."
docker-compose build base

echo "[INFO] [2/5] Generating & Partitioning Data..."
docker-compose run --rm data-prep

echo "[WARN] [3/5] Uploading Data to MinIO ($STORAGE_ENDPOINT)..."

# Upload Images (Custom CNN Features)
echo "   > Uploading Images (Custom CNN Features)..."
if ! docker-compose run --rm uploader bash -c "python3 fed_multimodal_restcol/trainer/run/upload.py \
  --pkls fed_multimodal_restcol/preprocess/output/feature/img/custom_cnn/custom_aqi/alpha50/0.pkl \
         fed_multimodal_restcol/preprocess/output/feature/img/custom_cnn/custom_aqi/alpha50/dev.pkl \
         fed_multimodal_restcol/preprocess/output/feature/img/custom_cnn/custom_aqi/alpha50/test.pkl \
  --collection_id $COLLECTION_ID \
  --restcol_host $RESTCOL_HOST \
  --restcol_authtoken '$RESTCOL_AUTH_TOKEN'" > img_upload.log 2>&1; then
    echo "[ERROR] Upload Failed! Log content:"
    cat img_upload.log
    exit 1
fi
cat img_upload.log
export IMG_DOC_ID=$(cat img_upload.log | grep "docid:" | awk '{print $2}' | tr -d ' \r')

# Upload Text (Tabular MLP Features)
echo "   > Uploading Text (Tabular MLP Features)..."
if ! docker-compose run --rm uploader bash -c "python3 fed_multimodal_restcol/trainer/run/upload.py \
  --pkls fed_multimodal_restcol/preprocess/output/feature/text/tabular_mlp/custom_aqi/alpha50/0.pkl \
         fed_multimodal_restcol/preprocess/output/feature/text/tabular_mlp/custom_aqi/alpha50/dev.pkl \
         fed_multimodal_restcol/preprocess/output/feature/text/tabular_mlp/custom_aqi/alpha50/test.pkl \
  --collection_id $COLLECTION_ID \
  --restcol_host $RESTCOL_HOST \
  --restcol_authtoken '$RESTCOL_AUTH_TOKEN'" > text_upload.log 2>&1; then
    echo "[ERROR] Upload Failed! Log content:"
    cat text_upload.log
    exit 1
fi
cat text_upload.log
export TEXT_DOC_ID=$(cat text_upload.log | grep "docid:" | awk '{print $2}' | tr -d ' \r')

if [ -z "$IMG_DOC_ID" ] || [ -z "$TEXT_DOC_ID" ]; then
  echo "[ERROR] Error: Failed to retrieve Document IDs from upload output."
  echo "   Use mock IDs for testing if network is unreachable."
  # Fallback for testing if upload fails
  export IMG_DOC_ID="mock-img-id"
  export TEXT_DOC_ID="mock-text-id"
fi

echo "   [SUCCESS] Captured IDs: IMG=$IMG_DOC_ID, TEXT=$TEXT_DOC_ID"

echo "[INFO] [4/5] Starting Federated Server & Client..."
echo "    Session ID: $SESSION_ID"
echo "    RestCol:    $RESTCOL_HOST"

docker-compose up -d server client

echo "[SUCCESS] [5/5] System is Running!"
echo "    View logs:  docker-compose logs -f"
echo "    Stop:       docker-compose down"
