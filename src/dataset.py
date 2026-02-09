"""
PyTorch Dataset for Temporal LULC Data
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple


class TemporalLULCDataset(Dataset):
    """
    PyTorch Dataset for loading temporal LULC sequences.
    
    Loads pre-generated temporal sequences and provides them in format
    suitable for training the Causal Spatiotemporal Transformer.
    """
    
    def __init__(
        self, 
        root_dir: str, 
        split: str = 'train', 
        img_size: int = 256, 
        num_classes: int = 7
    ):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory containing the data
            split: One of 'train', 'val', or 'test'
            img_size: Size of LULC maps
            num_classes: Number of LULC classes
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.num_classes = num_classes
        
        # Build list of sample paths
        self.split_dir = self.root_dir / split
        self.samples = sorted(list(self.split_dir.glob('sequence_*.npy')))
        
        if len(self.samples) == 0:
            raise ValueError(
                f"No samples found in {self.split_dir}. "
                "Please run generate_data.py first."
            )
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)
    
    def _load_lulc(self, path: Path) -> np.ndarray:
        """
        Load LULC sequence from numpy file.
        
        Args:
            path: Path to .npy file
            
        Returns:
            LULC sequence of shape (num_timesteps, H, W)
        """
        sequence = np.load(path)
        return sequence
    
    def _lulc_to_onehot(self, lulc: np.ndarray) -> np.ndarray:
        """
        Convert LULC map to one-hot encoding.
        
        Args:
            lulc: LULC map of shape (H, W) with class indices
            
        Returns:
            One-hot encoded map of shape (num_classes, H, W)
        """
        onehot = np.zeros((self.num_classes, self.img_size, self.img_size), dtype=np.float32)
        
        for class_id in range(self.num_classes):
            onehot[class_id] = (lulc == class_id).astype(np.float32)
        
        return onehot
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (inputs, target, prev_lulc):
                - inputs: Input sequence of shape (T-1, num_classes, H, W)
                - target: Target LULC map of shape (num_classes, H, W)
                - prev_lulc: Previous LULC map of shape (H, W) for physics constraints
        """
        # Load sequence
        sequence = self._load_lulc(self.samples[idx])
        
        # Split into inputs (all but last) and target (last)
        input_sequence = sequence[:-1]  # Shape: (T-1, H, W)
        target_lulc = sequence[-1]      # Shape: (H, W)
        prev_lulc = sequence[-2]        # Shape: (H, W) - for physics constraints
        
        # Convert to one-hot encoding
        inputs = []
        for lulc in input_sequence:
            onehot = self._lulc_to_onehot(lulc)
            inputs.append(onehot)
        
        inputs = np.stack(inputs, axis=0)  # Shape: (T-1, num_classes, H, W)
        target = self._lulc_to_onehot(target_lulc)  # Shape: (num_classes, H, W)
        
        # Convert to tensors
        inputs = torch.from_numpy(inputs).float()
        target = torch.from_numpy(target).float()
        prev_lulc = torch.from_numpy(prev_lulc).long()
        
        return inputs, target, prev_lulc
