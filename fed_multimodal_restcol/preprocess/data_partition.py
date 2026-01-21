import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import argparse
from argparse import Namespace
import logging

from partition_manager import PartitionManager

# 設定 logging 格式
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)

def data_partition(args: Namespace):
    
    num_clients, alpha = args.num_clients, args.alpha

    pm = PartitionManager(args)
    pm.fetch_label_dict()  

    # 讀取 CSV 資料
    data_path = Path(args.raw_data_dir)
    try:
        csv_data = pd.read_csv(data_path)
    except FileNotFoundError:
        logging.error(f"CSV file not found at {data_path}.")
        exit(1)
    except pd.errors.EmptyDataError:
        logging.error(f"CSV file is empty or invalid: {data_path}.")
        exit(1)

    img_dir = Path(args.img_dir)  
    if not img_dir.exists():
        logging.error(f"Image directory not found: {img_dir}")
        exit(1)

    # 分割資料 (80/10/10)
    logging.info("Partitioning AQI data")
    total_data = csv_data.sample(frac=1, random_state=42) 
    train_size = int(0.8 * len(total_data))
    dev_size = int(0.1 * len(total_data))

    train_data = total_data.iloc[:train_size]
    dev_data = total_data.iloc[train_size:train_size + dev_size]
    test_data = total_data.iloc[train_size + dev_size:]

    def create_data_dict(data: pd.DataFrame):
        data_dict = {}
        # 使用 tqdm 提供進度條
        for _, row in tqdm(data.iterrows(), total=data.shape[0]):
            img_name = str(row.get('Filename', ''))
            if not img_name:
                logging.warning("Missing Filename in row, skipping.")
                continue

            img_path = img_dir / img_name

            if not img_path.exists():
                logging.warning(f"Image not found: {img_path}")
                continue

            # 直接讀取數值特徵，並轉成 float 型態
            try:
                features = row[['aqi', 'pm2_5', 'pm10', 'RH', 'Temperature']].values.astype(float)
                # 如果需要，轉成 list (例如後續處理較方便)
                features = features.tolist()
            except Exception as e:
                logging.error(f"Error converting features for {img_name}: {e}")
                continue

            try:
                label = pm.label_dict[row['AQI_Class']]
            except KeyError:
                logging.error(f"Label not found for AQI_Class: {row['AQI_Class']} in {img_name}")
                continue

            img_n = str(row['Filename']).rsplit('.', 1)[0] 
            # 將圖片 ID、路徑、標籤、數值特徵存入字典中
            data_dict[img_name] = [img_n, str(img_path), label, features]
            print(data_dict)

        return data_dict

    # 建立訓練、驗證與測試資料字典
    train_data_dict = create_data_dict(train_data)
    dev_data_dict = create_data_dict(dev_data)
    test_data_dict = create_data_dict(test_data)

    if not train_data_dict:
        logging.error("No valid training data found. Exiting.")
        exit(1)

    # 取得訓練資料的文件 ID 並排序
    train_file_ids = sorted(train_data_dict.keys())

    # 建立標籤清單
    file_label_list = np.array([train_data_dict[file_id][2] for file_id in train_file_ids])

    # 使用 Dirichlet 分割訓練資料（注意方法名稱是否正確）
    try:
        # 假設正確的方法名稱為 dirichlet_partition
        file_idx_clients = pm.dirichlet_partition(file_label_list, min_sample_size=1)
    except AttributeError:
        logging.error("The PartitionManager does not have method 'dirichlet_partition'. Please check the method name.")
        exit(1)

    # 建立輸出資料夾
    output_data_path = Path(args.output_partition_path) / 'partition' / args.dataset
    output_data_path.mkdir(parents=True, exist_ok=True)

    # 建立每個 client 的資料分割對應
    client_data_dict = {
        client_idx: [
            train_data_dict[train_file_ids[idx]]
            for idx in file_idx_clients[client_idx]
            if train_file_ids[idx] in train_data_dict  # 確保索引存在
        ]
        for client_idx in range(args.num_clients)
    }

    # 加入驗證和測試資料
    client_data_dict["dev"] = list(dev_data_dict.values())
    client_data_dict["test"] = list(test_data_dict.values())

    # 儲存分割結果為 JSON
    alpha_str = str(args.alpha).replace('.', '')
    json_file_path = output_data_path / f'partition_alpha{alpha_str}.json'
    try:
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(client_data_dict, json_file, indent=4, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Error saving JSON file: {e}")
        exit(1)

    logging.info(f"Partitioning completed and saved to {json_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default='data/airquality.csv',
        help="Path to the raw CSV data file."
    )

    parser.add_argument(
        "--img_dir",
        type=str,
        default='data/raw_images',
        help="Path to the image directory."
    )

    parser.add_argument(
        "--output_partition_path",
        type=str,
        default='fed_multimodal_restcol/preprocess/output',
        help="Output path for partitioned data."
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=5.0,
        help="Alpha value for Dirichlet distribution."
    )

    parser.add_argument(
        '--num_clients', 
        type=int, 
        default=1,  
        help='Number of clients for data partitioning.'
    )

    parser.add_argument(
        "--dataset",
        type=str, 
        default="custom_aqi",  
        help='Dataset name.'
    )

    args = parser.parse_args()
    data_partition(args)
