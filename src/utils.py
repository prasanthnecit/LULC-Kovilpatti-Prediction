"""
Utility functions for training
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, precision_score, recall_score
import random
from pathlib import Path


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_metrics(predictions, targets):
    """Calculate classification metrics"""
    preds_flat = predictions.flatten()
    targets_flat = targets.flatten()
    
    return {
        'accuracy': accuracy_score(targets_flat, preds_flat),
        'f1': f1_score(targets_flat, preds_flat, average='weighted', zero_division=0),
        'kappa': cohen_kappa_score(targets_flat, preds_flat),
        'precision': precision_score(targets_flat, preds_flat, average='weighted', zero_division=0),
        'recall': recall_score(targets_flat, preds_flat, average='weighted', zero_division=0)
    }


def save_checkpoint(model, optimizer, epoch, metrics, path):
    """Save model checkpoint"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, path)
    print(f"  💾 Checkpoint saved: {path}")
