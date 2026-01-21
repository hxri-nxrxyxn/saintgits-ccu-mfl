#!/bin/bash
set -e

# ==============================================================================
# Federated Learning Node Pipeline
# ==============================================================================

# 1. Configuration
export STORAGE_ENDPOINT="storage.saintgits01.footprint-ai.com"

# --- Active Configuration ---
export STORAGE_ACCESS_KEY="Tjqsfoy5tu3Itq1r"
export STORAGE_ACCESS_SECRET="oRfEU3ZzlZl4WvBW7Y6kFCDootwKwc8G"
export STORAGE_BUCKET_NAME="project-2-electric-spitfire"
export RESTCOL_HOST="https://saintgits01.footprint-ai.com/reststore"
export RESTCOL_AUTH_TOKEN="eyJhbGciOiJSUzI1NiIsImtpZCI6ImM4MzExZmZmNWE3NjY5ZjRmOTNjMTMyODg3NmRkYTc3NjJmYTMyYjkifQ.eyJpc3MiOiJodHRwczovL2F1dGguc2FpbnRnaXRzMDEuZm9vdHByaW50LWFpLmNvbS9kZXgiLCJzdWIiOiJDaU5qYmoxa1pXMXZMRzkxUFZCbGIzQnNaU3hrWXoxbGVHRnRjR3hsTEdSalBXOXlaeElFYkdSaGNBIiwiYXVkIjoia2FmZWlkby1hcHAiLCJleHAiOjE3NjM2MTUzMjYsImlhdCI6MTc2MzUyODkyNiwiYXRfaGFzaCI6InB2Rmd5cEEtdHhiZzBLUVJ2eWh3WEEiLCJlbWFpbCI6ImRlbW9AZm9vdHByaW50LWFpLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJncm91cHMiOlsic3AxMCJdLCJuYW1lIjoiZGVtbyJ9.TgE5DE1Y8zUFgVXmX4vPiwH09Xl_RkAMPNtELm_Om1Jn_f3z5OWAfi7i4ELk3sMOFUEQ-ZvFkQR7N5IiQ9Q8zTydJYcMWYsmxirivbMPqEmNyYYLTjg-mbgqcYRgeLI40Tod90KEPu0X7F73Brrk_15EVovB1ZLuCXVdYoiaej0nJ2OmtxhLfXhzKzLIkbjEUTlkFp5yhaT02Xv4Ii1cy7_pPBCjm-DR986UFEBsvNxIX9187REVkNuw1vhpiBA4uOvHn9s0IcN4QGfmA2IScVy2HxAlwjuznF5avt5NUK-iDNBFLXg1kKViqw0-CgFkVU-jQbtryFnhhm2af_0ang"
export SESSION_ID="session_$(date +%s)"
export COLLECTION_ID="crisis-mmd"

# Initialize .env file to suppress docker warnings
echo "RESTCOL_AUTH_TOKEN=$RESTCOL_AUTH_TOKEN" > .env
echo "STORAGE_ENDPOINT=$STORAGE_ENDPOINT" >> .env
echo "STORAGE_ACCESS_KEY=$STORAGE_ACCESS_KEY" >> .env
echo "STORAGE_ACCESS_SECRET=$STORAGE_ACCESS_SECRET" >> .env
echo "STORAGE_BUCKET_NAME=$STORAGE_BUCKET_NAME" >> .env
echo "RESTCOL_HOST=$RESTCOL_HOST" >> .env
echo "SESSION_ID=$SESSION_ID" >> .env
# Placeholders to prevent "variable not set" warnings during build
echo "IMG_DOC_ID=" >> .env
echo "TEXT_DOC_ID=" >> .env

echo "cleaning up old containers..."
docker compose down --remove-orphans 2>/dev/null || true
docker rm -f fl-server fl-client 2>/dev/null || true

echo "building docker image..."
docker compose build base

echo "generating & partitioning data..."
docker compose run --rm data-prep

echo "uploading data to minio..."

# Upload Images
echo " > uploading images..."
if ! docker compose run --rm uploader bash -c "python3 fed_multimodal_restcol/trainer/run/upload.py \
  --pkls fed_multimodal_restcol/preprocess/output/feature/img/custom_cnn/custom_aqi/alpha50/0.pkl \
         fed_multimodal_restcol/preprocess/output/feature/img/custom_cnn/custom_aqi/alpha50/dev.pkl \
         fed_multimodal_restcol/preprocess/output/feature/img/custom_cnn/custom_aqi/alpha50/test.pkl \
  --collection_id $COLLECTION_ID \
  --restcol_host $RESTCOL_HOST \
  --restcol_authtoken '$RESTCOL_AUTH_TOKEN'" > img_upload.log 2>&1; then
    echo "error: upload failed. check img_upload.log"
    cat img_upload.log
    exit 1
fi
export IMG_DOC_ID=$(cat img_upload.log | grep "docid:" | awk '{print $2}' | tr -d ' \r')

# Upload Text
echo " > uploading text..."
if ! docker compose run --rm uploader bash -c "python3 fed_multimodal_restcol/trainer/run/upload.py \
  --pkls fed_multimodal_restcol/preprocess/output/feature/text/tabular_mlp/custom_aqi/alpha50/0.pkl \
         fed_multimodal_restcol/preprocess/output/feature/text/tabular_mlp/custom_aqi/alpha50/dev.pkl \
         fed_multimodal_restcol/preprocess/output/feature/text/tabular_mlp/custom_aqi/alpha50/test.pkl \
  --collection_id $COLLECTION_ID \
  --restcol_host $RESTCOL_HOST \
  --restcol_authtoken '$RESTCOL_AUTH_TOKEN'" > text_upload.log 2>&1; then
    echo "error: upload failed. check text_upload.log"
    cat text_upload.log
    exit 1
fi
export TEXT_DOC_ID=$(cat text_upload.log | grep "docid:" | awk '{print $2}' | tr -d ' \r')

if [ -z "$IMG_DOC_ID" ] || [ -z "$TEXT_DOC_ID" ]; then
  echo "error: failed to get document IDs. using mocks for testing."
  export IMG_DOC_ID="mock-img-id"
  export TEXT_DOC_ID="mock-text-id"
fi

# Update .env with real IDs so docker logs works later
sed -i '/IMG_DOC_ID=/d' .env
sed -i '/TEXT_DOC_ID=/d' .env
echo "IMG_DOC_ID=$IMG_DOC_ID" >> .env
echo "TEXT_DOC_ID=$TEXT_DOC_ID" >> .env

echo "captured IDs: IMG=$IMG_DOC_ID, TEXT=$TEXT_DOC_ID"

echo "starting server & client..."
docker compose up -d server client

echo "done. system running."
echo "logs: docker compose logs -f"

