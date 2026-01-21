FROM python:3.10.15

ENV PYTHONPATH=/app

WORKDIR /app

# Install system dependencies (ffmpeg is required by feature_manager.py)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY . .
