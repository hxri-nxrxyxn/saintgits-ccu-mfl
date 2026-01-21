import os
import pandas as pd
import numpy as np
from PIL import Image

# Setup Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMG_DIR = os.path.join(DATA_DIR, "raw_images")
CSV_PATH = os.path.join(DATA_DIR, "airquality.csv")

os.makedirs(IMG_DIR, exist_ok=True)

# Generate 20 Mock Images and Data Rows
num_samples = 20
data = []
aqi_classes = ['Good', 'Moderate', 'Unhealthy', 'Hazardous']

print(f"Generating {num_samples} mock samples...")

for i in range(num_samples):
    filename = f"mock_img_{i}.jpg"
    filepath = os.path.join(IMG_DIR, filename)
    
    # 1. Create a random noise image (128x128)
    # Using random noise to simulate 'complex' image data for testing
    img_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save(filepath)
    
    # 2. Generate random weather/AQI data
    row = {
        'Filename': filename,
        'AQI_Class': np.random.choice(aqi_classes),
        'aqi': np.random.uniform(0, 500),
        'pm2_5': np.random.uniform(0, 300),
        'pm10': np.random.uniform(0, 400),
        'RH': np.random.uniform(20, 100),         # Relative Humidity (Legacy)
        'RAINFALL': np.random.uniform(0, 50),     # (Legacy)
        'Temperature': np.random.uniform(-10, 40),# (Legacy)
        'WD_HR': np.random.uniform(0, 360),       # Wind Direction (Legacy)
        'WS_HR': np.random.uniform(0, 20)         # Wind Speed (Legacy)
    }
    data.append(row)

# Save to CSV
df = pd.DataFrame(data)
df.to_csv(CSV_PATH, index=False)

print(f"[SUCCESS] Mock data generated successfully!")
print(f"   Images: {IMG_DIR}")
print(f"   CSV: {CSV_PATH}")
