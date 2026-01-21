# Federated Multimodal Learning Node (Mock & Deploy)

This folder contains a complete, containerized environment for running a Federated Learning node. It is designed to work "out of the box" by mocking necessary weather/AQI data, processing it, and connecting to the central aggregation server.

## 🚀 Quick Start

To run the entire pipeline (Build -> Generate Data -> Upload -> Train), simply execute:

```bash
./start_pipeline.sh
```

This script will:
1.  **Build** the Docker environment with all dependencies (PyTorch, FFmpeg, etc.).
2.  **Generate** mock weather images and AQI data.
3.  **Process** the data:
    *   Partition it for Federated Learning.
    *   Extract features using a **Custom CNN** (for images) and **Tabular MLP** (for CSV data).
4.  **Upload** the processed features to the MinIO storage backend.
5.  **Launch** the `fl-server` and `fl-client` containers to begin training.

## 📂 Architecture

*   **`docker-compose.yml`**: Defines the services.
    *   `data-prep`: Runs generation and feature extraction.
    *   `uploader`: Helper service to push data to MinIO.
    *   `server`: The local Federated Server (Session coordinator).
    *   `client`: The local Federated Client (Trainer).
*   **`start_pipeline.sh`**: The orchestration script. Edit this file to update credentials or endpoints.
*   **`fed_multimodal_restcol/`**: The core application code.
*   **`data/`**: Location for raw images and CSVs (populated by the pipeline).

## 🛠️ Configuration

To change the target server or credentials, edit `start_pipeline.sh`. The file now supports easy switching between **Active** (Saintgits Prod) and **Legacy** (Mock/Local) configurations via comments.

```bash
# Example in start_pipeline.sh
# --- Active Configuration ---
export STORAGE_ACCESS_KEY="Tjqsfoy5tu3Itq1r"
...
# --- Legacy Configuration ---
# export STORAGE_ACCESS_KEY="demo@footprint-ai.com"
...
```

## 📊 Monitoring

Once running, view the logs to see the training progress:

```bash
docker-compose logs -f
```

To stop the system:

```bash
docker-compose down
```

---

# 🕵️ Troubleshooting & Verification Guide

**Author:** Gemini (Your AI Copilot)  
**Target:** The Proficient DevOps Guy (You)

If you are reading this, "start_pipeline.sh" probably didn't work perfectly on the first try. That's normal. Here is your survival guide to debugging this system.

## 1. Where do I start? (The "Smoke Test")

The pipeline runs in stages. If it fails, identify *which stage* died.

### Stage A: Build & Data Prep
**Command:** `docker-compose run --rm data-prep`
*   **What it does:** Generates images in `data/raw_images` and features in `fed_multimodal_restcol/preprocess/output`.
*   **Verification:** Check if `fed_multimodal_restcol/preprocess/output/feature` is empty. If it is, the feature extractor crashed (likely a PyTorch/CPU issue or missing `ffmpeg`).

### Stage B: Connectivity & Credentials (The likely failure point)
**Command:** The `uploader` service in `start_pipeline.sh`.
*   **The Issue:** We switched to external endpoints (`storage.saintgits01...`). If your network blocks them or the keys are wrong, this will hang or error out.
*   **How to Verify Keys:**
    You asked if the keys work. Use this manual test to find out without running the whole pipeline:
    ```bash
    # Run a temporary container to test connection
    docker-compose run --rm uploader python3 -c "
from minio import Minio
client = Minio('storage.saintgits01.footprint-ai.com',
               access_key='Tjqsfoy5tu3Itq1r',
               secret_key='oRfEU3ZzlZl4WvBW7Y6kFCDootwKwc8G',
               secure=True)
try:
    print('Buckets:', client.list_buckets())
    print('✅ SUCCESS: Keys work!')
except Exception as e:
    print(f'❌ FAILURE: {e}')
    "
    ```
    If this prints `✅ SUCCESS`, the keys are valid. If not, revert to the "Legacy" keys in `start_pipeline.sh`.

### Stage C: Server & Client Handshake
**Command:** `docker-compose up server client`
*   **The Issue:** The Client needs to find the Server via `RestCol`.
*   **Debug Tip:**
    1.  Tail the server logs: `docker-compose logs -f server`
    2.  Look for: `server_collection_id=...` (We added this print statement for you).
    3.  Tail the client logs: `docker-compose logs -f client`
    4.  The Client must say "got global model". If it says "waiting...", it can't find the Server's ID.

## 2. Common Errors & Fixes

| Error | Likely Cause | Fix |
| :--- | :--- | :--- |
| `S3 Connection Error` / `Timeout` | Wrong Endpoint or Firewall | Check `STORAGE_ENDPOINT` in `start_pipeline.sh`. Ensure you have internet access. |
| `Access Denied` (MinIO) | Bad Keys | Run the **Verification** script above. Swap keys in `start_pipeline.sh` if needed. |
| `RuntimeError: CUDA error` | No GPU | Ensure `device = torch.device("cpu")` is set in `client.py` and `server.py` (The code is currently set to force CPU). |
| `ffmpeg: command not found` | Docker Build Fail | `apt-get install ffmpeg` is in the Dockerfile, but sometimes rebuilds are needed: `docker-compose build --no-cache`. |

## 3. Operations Manual (How to run like a Pro)

1.  **Edit Credentials:** Open `start_pipeline.sh` and toggle the comments if you need to switch environments.
2.  **Clean Start:** Always run `docker-compose down` before starting a fresh run to clear old session containers.
3.  **Check Data:** If you want to see what data is being trained on, look at `data/airquality.csv`. It's human-readable.
