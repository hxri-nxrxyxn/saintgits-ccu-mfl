import os
import pdb
import glob
import random
import collections
import numpy as np
import os.path as osp
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple


class PartitionManager():
    def __init__(self, args: dict):
        self.args = args
    
    def fetch_label_dict(self):
        # fetch unique labels
        if self.args.dataset == "custom_aqi":
            self.label_dict = {
                'Good':                             0,
                'Moderate':                         1,
                'Unhealthy for Sensitive Groups':   2,
                'Unhealthy':                        3,
                'Very Unhealthy':                   4,
                'Hazardous':                        5      
            }        

        elif self.args.dataset == "crisis-mmd":
            self.label_dict = {
                'not_humanitarian':                         0,
                'infrastructure_and_utility_damage':        1,
                'vehicle_damage':                           2,
                'rescue_volunteering_or_donation_effort':   3,
                'other_relevant_information':               4,
                'affected_individuals':                     5,
                'injured_or_dead_people':                   6,
                'missing_or_found_people':                  7
            }   
        
    def split_train_dev(
        self, 
        train_val_file_id: list,
        seed: int=8
    ) -> Tuple[List, List]:
        # shuffle train idx, and select 20% for dev
        train_arr = np.arange(len(train_val_file_id))
        np.random.seed(seed)
        np.random.shuffle(train_arr)
        val_len = int(len(train_arr)/5)
        # read the keys
        train_file_id = [train_val_file_id[idx] for idx in train_arr[val_len:]]
        val_file_id = [train_val_file_id[idx] for idx in train_arr[:val_len]]
        return train_file_id, val_file_id
    
    def dirichlet_partition(
        self, 
        file_label_list: list,
        seed: int=8,
        min_sample_size: int=5
    ) -> (list):
        
        # cut the data using dirichlet
        min_size = 0
        K, N = len(np.unique(file_label_list)), len(file_label_list)
        # seed
        np.random.seed(seed)
        while min_size < min_sample_size:
            file_idx_clients = [[] for _ in range(self.args.num_clients)]
            for k in range(K):
                idx_k = np.where(np.array(file_label_list) == k)[0]
                np.random.shuffle(idx_k)
                # if self.args.dataset == "hateful_memes" and k == 0:
                #    proportions = np.random.dirichlet(np.repeat(1.0, self.args.num_clients))
                # else:
                proportions = np.random.dirichlet(np.repeat(self.args.alpha, self.args.num_clients))
                # Balance
                proportions = np.array([p*(len(idx_j)<N/self.args.num_clients) for p, idx_j in zip(proportions, file_idx_clients)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                file_idx_clients = [idx_j + idx.tolist() for idx_j,idx in zip(file_idx_clients,np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in file_idx_clients])
        return file_idx_clients