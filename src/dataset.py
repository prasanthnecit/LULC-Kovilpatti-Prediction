"""
Dataset class for real satellite LULC data
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os


class RealLULCDataset(Dataset):
    """Dataset for real satellite LULC data"""
    
    def __init__(self, data_dir, split='train'):
        """
        Args:
            data_dir: Path to data directory
            split: 'train', 'val', or 'test'
        """
        self.inputs = np.load(os.path.join(data_dir, f'{split}_inputs.npy'))
        self.targets = np.load(os.path.join(data_dir, f'{split}_targets.npy'))
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        inputs = torch.LongTensor(self.inputs[idx])  # (seq_len, H, W)
        targets = torch.LongTensor(self.targets[idx])  # (H, W)
        return inputs, targets
