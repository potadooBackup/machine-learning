import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from src.hyper import DATA_ROOT_DIR, NUM_WORKERS

class COVID19Dataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

def load_dataset(filename: str):
    return pd.read_csv(os.path.join(DATA_ROOT_DIR, filename)).values

def train_valid_split(dataset: Dataset, valid_ratio: float, seed:np.uint):
    '''Split provided training data into training set and validation set'''
    valid_set_size = int(valid_ratio * len(dataset)) 
    train_set_size = len(dataset) - valid_set_size
    train_set, valid_set = random_split(COVID19Dataset(dataset[:,:-1],dataset[:,-1]), [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return train_set, valid_set

def get_data_loaders(train_dataset = None, valid_dataset = None, test_dataset = None,
                     batch_size = 8, num_workers = 0):
    # Training data loader.
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False
    ) if not (train_dataset is None) else None

    # Validation data loader.
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    ) if not (valid_dataset is None) else None

    # Test data loader.
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    ) if not (test_dataset is None) else None

    return train_loader, valid_loader, test_loader