import os
import pdb
import pickle
import logging
import warnings
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

warnings.filterwarnings('ignore')

from feature_manager import FeatureManager

# define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)

def parse_args():

    parser = argparse.ArgumentParser(description='Extract text features')
    parser.add_argument(
        '--raw_data_dir',
        default='/home/user/Downloads/sample_airquality_data.csv',
        type=str,
        help='source data directory'
    )
    parser.add_argument(
        '--output_dir', 
        default="./output",
        type=str, 
        help='output feature directory'
    )
    parser.add_argument(
        '--feature_type', 
        default='tabular_mlp',
        type=str, 
        help='output feature name'
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=5.0,
        help="alpha in direchlet distribution",
    )
    parser.add_argument("--dataset", default="custom_aqi")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    # read args
    args = parse_args()
    alpha_str = str(args.alpha).replace('.', '')
    output_data_path = Path(args.output_dir).joinpath(
        'feature', 
        'text', 
        args.feature_type, 
        args.dataset, 
        f'alpha{alpha_str}'
    )
    Path.mkdir(output_data_path, parents=True, exist_ok=True)
    
    # initialize feature processer
    fm = FeatureManager(args)

    # fetch all files for processing
    partition_dict = fm.fetch_partition(alpha=args.alpha)
    logging.info(f'Total number of clients found (including dev and test): {len(partition_dict.keys())}')
    
    # extract data, read base case first, alpha=1.0
    # If the base folder is empty, we extract for the base case
    base_data_path = Path(args.output_dir).joinpath(
        'feature', 
        'text', 
        args.feature_type, 
        args.dataset, 
        f'alpha{alpha_str}'
    )
    client_file_paths = os.listdir(base_data_path)
    client_file_paths.sort()
    
    # extract based feature
    if len(client_file_paths) != len(partition_dict) and args.alpha == 5.0:
        # iterate over client, including keys = dev/test
        for client in tqdm(list(partition_dict.keys())):
            data_dict = partition_dict[client].copy()
            if Path.exists(output_data_path.joinpath(f'{client}.pkl')) == True: continue
            # extract feature
            for idx in range(len(partition_dict[client])):
                text_str = partition_dict[client][idx][-1]
                if args.feature_type in ['bert', 'mobilebert']:
                    features = fm.extract_text_feature(text_str)
                elif args.feature_type == 'tabular_mlp':
                    features = fm.extract_tabular_feature(np.array(text_str, dtype=np.float32))
                else:
                    logging.warning(f"Unsupported feature_type for text: {args.feature_type}")
                    features = None

                data_dict[idx][-1] = features
            # saving features
            with open(str(output_data_path.joinpath(f'{client}.pkl')), 'wb') as handle:
                pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # base feature all extracted, and we want to explore other alpha cases
    if len(client_file_paths) == len(partition_dict):
        train_dict = dict()
        logging.info('Read alpha=5.0 data')
        for client_file_path in tqdm(client_file_paths[:-2]):
            with open(str(base_data_path.joinpath(client_file_path)), "rb") as f: 
                client_data = pickle.load(f)
            for idx in range(len(client_data)):
                key = client_data[idx][0]
                train_dict[key] = client_data[idx]
        # reading data
        logging.info(f'Save alpha={args.alpha} data')
        client_ids = [client_id for client_id in list(partition_dict.keys()) if client_id not in ['dev', 'test']]
        for client_id in tqdm(client_ids):
            save_data = list()
            for idx in range(len(partition_dict[client_id])):
                key = partition_dict[client_id][idx][0]
                save_data.append(train_dict[key])
            with open(str(output_data_path.joinpath(f'{client_id}.pkl')), 'wb') as handle:
                pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # dev and test should be the same, as fix the seed when spliting the data
        with open(str(base_data_path.joinpath('dev.pkl')), "rb") as f: dev_data = pickle.load(f)
        with open(str(base_data_path.joinpath('test.pkl')), "rb") as f: test_data = pickle.load(f)
        # saving data
        with open(str(output_data_path.joinpath(f'dev.pkl')), 'wb') as handle:
            pickle.dump(dev_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(str(output_data_path.joinpath(f'test.pkl')), 'wb') as handle:
            pickle.dump(test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

