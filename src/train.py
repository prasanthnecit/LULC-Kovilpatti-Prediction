"""
Training utilities for LULC prediction model
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score, precision_score, recall_score


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Calculate evaluation metrics for LULC prediction.
    
    Args:
        pred: Predicted logits of shape (B, num_classes, H, W)
        target: Target one-hot of shape (B, num_classes, H, W)
        
    Returns:
        Dictionary of metrics (accuracy, F1, kappa, precision, recall)
    """
    # Convert to class indices
    pred_classes = torch.argmax(pred, dim=1).cpu().numpy().flatten()
    target_classes = torch.argmax(target, dim=1).cpu().numpy().flatten()
    
    # Calculate metrics
    accuracy = (pred_classes == target_classes).mean()
    f1 = f1_score(target_classes, pred_classes, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(target_classes, pred_classes)
    precision = precision_score(target_classes, pred_classes, average='weighted', zero_division=0)
    recall = recall_score(target_classes, pred_classes, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'kappa': kappa,
        'precision': precision,
        'recall': recall
    }


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    physics_loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        dataloader: Training dataloader
        criterion: Main loss criterion (CrossEntropy)
        physics_loss_fn: Physics-informed loss
        optimizer: Optimizer
        device: Device to train on
        config: Configuration dictionary
        
    Returns:
        Dictionary with epoch metrics
    """
    model.train()
    
    total_loss = 0.0
    total_ce_loss = 0.0
    total_physics_loss = 0.0
    all_preds = []
    all_targets = []
    
    # Get lambda values from config
    lambda_physics = config.get('lambda_physics', 0.1)
    lambda_continuity = config.get('lambda_continuity', 0.05)
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, (inputs, target, prev_lulc) in enumerate(pbar):
        # Move to device
        inputs = inputs.to(device)      # (B, T, C, H, W)
        target = target.to(device)      # (B, C, H, W)
        prev_lulc = prev_lulc.to(device)  # (B, H, W)
        
        # Forward pass
        optimizer.zero_grad()
        pred, _ = model(inputs)  # (B, C, H, W)
        
        # Calculate losses
        ce_loss = criterion(pred, target)
        
        # Physics constraints
        transition_loss = physics_loss_fn.transition_constraint(pred, prev_lulc)
        continuity_loss = physics_loss_fn.spatial_continuity(pred)
        
        physics_loss = lambda_physics * transition_loss + lambda_continuity * continuity_loss
        
        # Combined loss
        loss = ce_loss + physics_loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_physics_loss += physics_loss.item()
        
        all_preds.append(pred.detach())
        all_targets.append(target.detach())
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ce': f'{ce_loss.item():.4f}',
            'phys': f'{physics_loss.item():.4f}'
        })
    
    # Calculate epoch metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = calculate_metrics(all_preds, all_targets)
    
    metrics.update({
        'loss': total_loss / len(dataloader),
        'ce_loss': total_ce_loss / len(dataloader),
        'physics_loss': total_physics_loss / len(dataloader)
    })
    
    return metrics


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    physics_loss_fn: nn.Module,
    device: torch.device,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """
    Validate the model.
    
    Args:
        model: The model to validate
        dataloader: Validation dataloader
        criterion: Main loss criterion
        physics_loss_fn: Physics-informed loss
        device: Device to validate on
        config: Configuration dictionary
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    
    total_loss = 0.0
    total_ce_loss = 0.0
    total_physics_loss = 0.0
    all_preds = []
    all_targets = []
    
    # Get lambda values from config
    lambda_physics = config.get('lambda_physics', 0.1)
    lambda_continuity = config.get('lambda_continuity', 0.05)
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for inputs, target, prev_lulc in pbar:
            # Move to device
            inputs = inputs.to(device)
            target = target.to(device)
            prev_lulc = prev_lulc.to(device)
            
            # Forward pass
            pred, _ = model(inputs)
            
            # Calculate losses
            ce_loss = criterion(pred, target)
            
            # Physics constraints
            transition_loss = physics_loss_fn.transition_constraint(pred, prev_lulc)
            continuity_loss = physics_loss_fn.spatial_continuity(pred)
            
            physics_loss = lambda_physics * transition_loss + lambda_continuity * continuity_loss
            
            # Combined loss
            loss = ce_loss + physics_loss
            
            # Track metrics
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_physics_loss += physics_loss.item()
            
            all_preds.append(pred.detach())
            all_targets.append(target.detach())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = calculate_metrics(all_preds, all_targets)
    
    metrics.update({
        'loss': total_loss / len(dataloader),
        'ce_loss': total_ce_loss / len(dataloader),
        'physics_loss': total_physics_loss / len(dataloader)
    })
    
    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: str
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch number
        metrics: Metrics dictionary
        path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")
