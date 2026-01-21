#D:\fed-multimodal-restcol\fed_multimodal_restcol\dataloader\dataloader.py
import json
import glob
import torch
import pickle
import random
import pdb, os
import torchaudio
import numpy as np
import os.path as osp
# import pickle5 as pickle

from tqdm import tqdm
from pathlib import Path
from collections import Counter
from torch.utils.data import DataLoader, Dataset

class MMDatasetGenerator(Dataset):
    def __init__(
        self, 
        modalityA, 
        modalityB, 
        default_feat_shape_a,
        default_feat_shape_b,
        data_len: int, 
    ):
        self.data_len = data_len
        self.modalityA = modalityA
        self.modalityB = modalityB
        self.default_feat_shape_a = default_feat_shape_a
        self.default_feat_shape_b = default_feat_shape_b
    def __len__(self):
        #return 2
        return self.data_len
    

    def __getitem__(self, item):
        # read modality
        data_a = self.modalityA[item][-1]
        data_b = self.modalityB[item][-1]
        label = torch.tensor(self.modalityA[item][-2])
        
        # modality A, if missing replace with 0s, and mask
        if data_a is not None: 
            if len(data_a.shape) == 3: data_a = data_a[0]
            data_a = torch.tensor(data_a)
            len_a = len(data_a)
        else: 
            data_a = torch.tensor(np.zeros(self.default_feat_shape_a))
            len_a = 0

        # modality B, if missing replace with 0s
        if data_b is not None:
            data_b = torch.tensor(data_b)
            len_b = data_b.shape[0]
        else:
            data_b = torch.tensor(np.zeros(self.default_feat_shape_b))
            len_b = 0
        return data_a, data_b, len_a, len_b, label

def collate_mm_fn_padd(batch):
    # find longest sequence
    if batch[0][0] is not None: max_a_len = max(map(lambda x: x[0].shape[0], batch))
    if batch[0][1] is not None: max_b_len = max(map(lambda x: x[1].shape[0], batch))

    # pad according to max_len
    x_a, x_b, len_a, len_b, ys = list(), list(), list(), list(), list()
    for idx in range(len(batch)):
        x_a.append(pad_tensor(batch[idx][0], pad=max_a_len))
        x_b.append(pad_tensor(batch[idx][1], pad=max_b_len))

        len_a.append(torch.tensor(batch[idx][2]))
        len_b.append(torch.tensor(batch[idx][3]))

        ys.append(batch[idx][-1])

    # stack all
    x_a = torch.stack(x_a, dim=0)
    x_b = torch.stack(x_b, dim=0)
    len_a = torch.stack(len_a, dim=0)
    len_b = torch.stack(len_b, dim=0)
    ys = torch.stack(ys, dim=0)
    return x_a, x_b, len_a, len_b, ys

def pad_tensor(vec, pad):
    pad_size = list(vec.shape)
    pad_size[0] = pad - vec.size(0)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=0)

def new_dataloader(
        data_a: dict,
        data_b: dict,
        default_feat_shape_a: np.array=np.array([0, 0]),
        default_feat_shape_b: np.array=np.array([0, 0]),
        shuffle: bool=False,
        batch_size: int = 32,
    ) -> (DataLoader):
    """
    Set dataloader for training/dev/test.
    :param data_a: modality A data
    :param data_b: modality B data
    :param default_feat_shape_a: default input shape for modality A, fill 0 in missing modality case
    :param default_feat_shape_b: default input shape for modality B, fill 0 in missing modality case
    :param shuffle: shuffle flag for dataloader, True for training; False for dev and test
    :return: dataloader: torch dataloader
    """
    # modify data based on simulation
    labeled_data_idx, unlabeled_data_idx = list(), list()

    if len(data_a) == 0: return None
    data_ab = MMDatasetGenerator(
        data_a, 
        data_b,
        default_feat_shape_a,
        default_feat_shape_b,
        len(data_a),
    )
    if shuffle:
        # we use args input batch size for train, typically set as 16 in FL setup
        dataloader = DataLoader(
            data_ab, 
            batch_size=batch_size,
            num_workers=0, 
            shuffle=shuffle, 
            collate_fn=collate_mm_fn_padd
        )
    else:
        # we use a larger batch size for validation and testing
        dataloader = DataLoader(
            data_ab, 
            batch_size=batch_size, 
            num_workers=0, 
            shuffle=shuffle, 
            collate_fn=collate_mm_fn_padd
        )
    return dataloader
