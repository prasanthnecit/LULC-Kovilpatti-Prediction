"""
Utility functions for LULC prediction
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from sklearn.metrics import confusion_matrix


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def visualize_prediction(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    target: torch.Tensor,
    generator,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize model prediction compared to ground truth.
    
    Args:
        model: Trained model
        inputs: Input sequence
        target: Ground truth target
        generator: LULC generator for RGB conversion
        save_path: Optional path to save figure
    """
    model.eval()
    
    with torch.no_grad():
        pred, _ = model(inputs.unsqueeze(0))
        pred = pred.squeeze(0)
    
    # Convert to class maps
    pred_classes = torch.argmax(pred, dim=0).cpu().numpy()
    target_classes = torch.argmax(target, dim=0).cpu().numpy()
    
    # Convert to RGB
    pred_rgb = generator.lulc_to_rgb(pred_classes)
    target_rgb = generator.lulc_to_rgb(target_classes)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(target_rgb)
    axes[0].set_title('Ground Truth', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(pred_rgb)
    axes[1].set_title('Prediction', fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_temporal_sequence(
    sequence: List[np.ndarray],
    generator,
    years: List[int],
    save_path: Optional[str] = None
) -> None:
    """
    Visualize temporal evolution of LULC.
    
    Args:
        sequence: List of LULC maps
        generator: LULC generator for RGB conversion
        years: List of year labels
        save_path: Optional path to save figure
    """
    n_timesteps = len(sequence)
    fig, axes = plt.subplots(1, n_timesteps, figsize=(4 * n_timesteps, 4))
    
    if n_timesteps == 1:
        axes = [axes]
    
    for idx, (lulc_map, year) in enumerate(zip(sequence, years)):
        rgb = generator.lulc_to_rgb(lulc_map)
        axes[idx].imshow(rgb)
        axes[idx].set_title(f'Year {year}', fontsize=12)
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Temporal sequence saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def predict_future_multistep(
    model: torch.nn.Module,
    initial_sequence: torch.Tensor,
    num_steps: int = 3,
    device: torch.device = torch.device('cpu')
) -> List[torch.Tensor]:
    """
    Perform multi-step future prediction.
    
    Args:
        model: Trained model
        initial_sequence: Initial input sequence of shape (T, C, H, W)
        num_steps: Number of future steps to predict
        device: Device to run on
        
    Returns:
        List of predicted LULC maps
    """
    model.eval()
    predictions = []
    
    current_sequence = initial_sequence.clone().to(device)
    
    with torch.no_grad():
        for step in range(num_steps):
            # Predict next timestep
            pred, _ = model(current_sequence.unsqueeze(0))
            pred = pred.squeeze(0)  # (C, H, W)
            
            predictions.append(pred.cpu())
            
            # Convert prediction to one-hot for next input
            pred_onehot = torch.zeros_like(pred)
            pred_classes = torch.argmax(pred, dim=0)
            for c in range(pred.size(0)):
                pred_onehot[c] = (pred_classes == c).float()
            
            # Update sequence (remove oldest, add newest)
            current_sequence = torch.cat([
                current_sequence[1:],
                pred_onehot.unsqueeze(0)
            ], dim=0)
    
    return predictions


def create_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_names: List[str],
    save_path: Optional[str] = None
) -> None:
    """
    Create and visualize confusion matrix.
    
    Args:
        predictions: Predicted logits of shape (B, C, H, W)
        targets: Target one-hot of shape (B, C, H, W)
        class_names: List of class names
        save_path: Optional path to save figure
    """
    # Convert to class indices
    pred_classes = torch.argmax(predictions, dim=1).cpu().numpy().flatten()
    target_classes = torch.argmax(targets, dim=1).cpu().numpy().flatten()
    
    # Compute confusion matrix
    cm = confusion_matrix(target_classes, pred_classes)
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Ground Truth', fontsize=12)
    ax.set_title('Confusion Matrix (Normalized)', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_results(
    history: Dict[str, List[float]],
    test_metrics: Dict[str, float],
    config: Dict[str, Any],
    save_dir: str
) -> None:
    """
    Save training results to JSON.
    
    Args:
        history: Training history dictionary
        test_metrics: Test metrics dictionary
        config: Configuration dictionary
        save_dir: Directory to save results
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'config': config,
        'history': history,
        'test_metrics': test_metrics
    }
    
    results_path = save_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")
    
    # Also plot training curves
    plot_training_curves(history, save_dir / 'training_curves.png')


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str
) -> None:
    """
    Plot training and validation curves.
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[0, 1].plot(history['train_accuracy'], label='Train')
    axes[0, 1].plot(history['val_accuracy'], label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy Curves')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 curves
    axes[1, 0].plot(history['train_f1'], label='Train')
    axes[1, 0].plot(history['val_f1'], label='Validation')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 Score Curves')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Kappa curves
    axes[1, 1].plot(history['train_kappa'], label='Train')
    axes[1, 1].plot(history['val_kappa'], label='Validation')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Cohen\'s Kappa')
    axes[1, 1].set_title('Kappa Curves')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()
