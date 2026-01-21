# MFL Project Game Plan & Architecture

**Project:** Federated Multimodal Learning Node (Air & Water Quality)  
**Date:** January 21, 2026  
**Status:** Integrated & Functional  

## 1. Executive Summary

This project implements a **Federated Learning (FL) Node** designed to train a multimodal AI model on environmental data (Air Quality & Water Quality) without exposing raw sensitive data to a central server. The system integrates a real-time dashboard for data visualization with an FL pipeline that periodically processes this data, trains a local model, and shares only the model updates (weights) with a central aggregation server.

## 2. System Architecture

The system consists of three main components:

1.  **Sensor Dashboard (Data Source):** A Flask-based web application that collects, visualizes, and exports real-time sensor data.
2.  **Federated Learning Pipeline (The Core):** A containerized pipeline that prepares data, extracts features, and runs the FL client.
3.  **Storage & Coordination:** Uses MinIO (Object Storage) and RestCol (Metadata Store) for handling model weights and training sessions.

### High-Level Data Flow

1.  **Sensing:** IoT Nodes -> MQTT Broker -> Dashboard Database (MySQL/SQLite).
2.  **Export:** Dashboard exports hourly CSVs of environmental readings.
3.  **Ingestion:** FL Pipeline reads these CSVs (`generate_real_dataset.py`).
4.  **Preprocessing:** 
    *   **Tabular Data:** Processed via `extract_text_feature.py` using a Tabular MLP.
    *   **Visual Data:** Dummy images (representing camera feeds) processed via `extract_img_feature.py` using a Custom CNN.
5.  **Training:** The `fl-client` trains on this local data.
6.  **Aggregation:** Model updates are pushed to the `fl-server` (or central cloud server) via `RestCol`.

## 3. Directory Structure & Key Files

### A. Federated Learning Node (`@MFL/working_setup/`)
This is the "Brain" of the operation.

*   **`start_pipeline.sh`**: **[Entry Point]** Orchestrates the entire process. Builds Docker containers, generates data, uploads features, and starts training.
*   **`docker-compose.yml`**: Defines the microservices (`data-prep`, `uploader`, `server`, `client`).
*   **`generate_real_dataset.py`**: **[New]** Transforming Dashboard CSV exports into the specific format required by the MFL model.
*   **`fed_multimodal_restcol/`**: Python package containing the FL logic.
    *   `preprocess/`: Scripts to partition data and extract features (CNN for images, MLP for text/tabular).
    *   `trainer/`: Contains `client.py` (local training loop) and `server.py` (aggregation logic).
    *   `restcol/`: Client library for communicating with the RestCol metadata server.

### B. Real-Time Dashboard (`@rt dashboard v2/`)
This is the "Face" of the operation.

*   **`app.py`**: Flask web server serving the UI and API endpoints (`/get_latest_data`, `/get_history`).
*   **`exports/`**: Directory where hourly environmental data is saved as CSV files.
*   **`templates/`**: HTML files for the dashboard UI (`index.html`, `air.html`, `water.html`).
*   **`sensor_db.sqlite`**: Local database storing all raw sensor readings.

## 4. API & Integration Points

| Component | Endpoint / Method | Purpose |
| :--- | :--- | :--- |
| **Dashboard** | `GET /export/csv` | Exports raw DB data to CSV for the FL pipeline to consume. |
| **RestCol** | `POST /collections` | Used by FL Client to register datasets and training sessions. |
| **MinIO** | `S3 Protocol` | Stores heavy artifacts like model checkpoints (`.pkl` files) and dataset features. |
| **FL Server** | `RestCol Document` | The Server writes its global model to a specific Document ID; Clients poll this to get the latest model. |

## 5. How to Run & Verify

### Step 1: Start the Dashboard
(If not already running)
```bash
cd "rt dashboard v2"
python3 app.py
```
*   **Verify:** Open `http://localhost:5000` to see live sensor data.

### Step 2: Run the FL Pipeline
This consumes data from the dashboard and starts training.
```bash
cd "MFL/working_setup"
./start_pipeline.sh
```
*   **Verify:** 
    *   Watch terminal output for "Training... Epoch 1".
    *   Check `MFL/working_setup/data/airquality.csv` to see the integrated dataset.

## 6. Future Improvements (Professor's Note)

*   **Grafana Integration:** The current Flask dashboard is a prototype. In production, this should be replaced by Grafana linked to the SQL database for more robust visualization.
*   **Real Camera Integration:** Currently, the pipeline generates noise images (`mock_img_X.jpg`). Integrating real camera feeds from the IoT nodes will allow the Multimodal model to learn from visual environmental cues (e.g., smog, cloud cover).
*   **Cloud Aggregation:** Move the `fl-server` container to a cloud instance to aggregate updates from multiple physical locations (Edges).
