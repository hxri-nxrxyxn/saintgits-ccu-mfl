import os
import pandas as pd
import numpy as np
from PIL import Image
import glob

# Setup Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SOURCE_CSV_DIR = os.path.join(DATA_DIR, "dashboard_exports")
IMG_DIR = os.path.join(DATA_DIR, "raw_images")
OUTPUT_CSV_PATH = os.path.join(DATA_DIR, "airquality.csv")

# Ensure directories exist
os.makedirs(IMG_DIR, exist_ok=True)

# 1. Load and Combine Dashboard CSVs
print(f"Loading CSVs from {SOURCE_CSV_DIR}...")
csv_files = glob.glob(os.path.join(SOURCE_CSV_DIR, "*.csv"))
if not csv_files:
    print(f"[ERROR] No CSV files found in {SOURCE_CSV_DIR}")
    exit(1)

df_list = []
for f in csv_files:
    try:
        df = pd.read_csv(f)
        df_list.append(df)
    except Exception as e:
        print(f"Skipping {f}: {e}")

if not df_list:
    print("[ERROR] No valid data found.")
    exit(1)

full_df = pd.concat(df_list, ignore_index=True)
print(f"Combined {len(full_df)} records.")

# 2. Transform Data to MFL Format
# MFL Expected Columns: Filename, AQI_Class, aqi, pm2_5, pm10, RH, RAINFALL, Temperature, WD_HR, WS_HR

# Map existing columns
mfl_df = pd.DataFrame()

# Generate Filenames (image files)
# We will generate filenames like "img_0.jpg", "img_1.jpg", etc.
mfl_df['Filename'] = [f"real_img_{i}.jpg" for i in range(len(full_df))]

# Map columns
# Source: PM2.5, PM10, RH, RAINFALL, Temperature, WIND_SPEED, AQI, AQI_Class
# Target: pm2_5, pm10, RH, RAINFALL, Temperature, WS_HR, aqi, AQI_Class
# Missing: WD_HR (Wind Direction)

column_mapping = {
    'PM2.5': 'pm2_5',
    'PM10': 'pm10',
    'RH': 'RH',
    'RAINFALL': 'RAINFALL',
    'Temperature': 'Temperature',
    'WIND_SPEED': 'WS_HR',
    'AQI': 'aqi',
    'AQI_Class': 'AQI_Class'
}

for source_col, target_col in column_mapping.items():
    if source_col in full_df.columns:
        mfl_df[target_col] = full_df[source_col]
    else:
        print(f"[WARN] Missing column {source_col}, filling with 0.")
        mfl_df[target_col] = 0

# Mock Missing Data (Wind Direction)
# Dashboard data has WIND_SPEED but often lacks specific Direction in hourly exports
mfl_df['WD_HR'] = np.random.uniform(0, 360, len(mfl_df))

# Fill NaNs
mfl_df = mfl_df.fillna(0)

# 3. Generate Dummy Images
# The MFL model uses a Custom CNN that expects 128x128 images.
# Since we only have tabular data, we will generate noise images.
# In a real scenario, you might want to generate a chart or use a static placeholder.
print(f"Generating {len(mfl_df)} dummy images in {IMG_DIR}...")

# Create a single base image to copy/save to save time? 
# No, for 'realism' in file handling lets create them.
# To speed up, we'll create one random array and modify it slightly or just save it.
# Actually, let's create a gradient or something visually distinct per class if possible,
# but for now random noise is sufficient for the pipeline to run.

# Pre-generate a few patterns to cycle through to save generation time
patterns = [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(5)]

for idx, row in mfl_df.iterrows():
    filename = row['Filename']
    filepath = os.path.join(IMG_DIR, filename)
    
    # Skip if exists to save time on re-runs
    if os.path.exists(filepath):
        continue
        
    # Pick a pattern based on index
    img_array = patterns[idx % len(patterns)]
    img = Image.fromarray(img_array)
    img.save(filepath)
    
    if idx % 100 == 0:
        print(f"Generated {idx}/{len(mfl_df)} images...", end='\r')

print(f"\n[SUCCESS] Images generated.")

# 4. Save Final CSV
mfl_df.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"[SUCCESS] Dataset saved to {OUTPUT_CSV_PATH}")
print(mfl_df.head())
