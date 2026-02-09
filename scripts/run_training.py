#!/usr/bin/env python3
"""
Main training script for LULC prediction model.
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import TemporalLULCDataset
from src.model import CausalSpatiotemporalTransformer, PhysicsInformedLoss
from src.train import train_epoch, validate, save_checkpoint
from src.utils import set_seed, save_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train LULC prediction model'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='./configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu, auto-detect if not specified)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.epochs is not None:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    
    print("=" * 70)
    print("LULC Prediction Training - Kovilpatti Region")
    print("=" * 70)
    
    # Set random seed
    set_seed(42)
    
    # Set up device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nDevice: {device}")
    
    # Create output directories
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path(config['paths']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = TemporalLULCDataset(
        root_dir=config['paths']['data_dir'],
        split='train',
        img_size=config['data']['img_size'],
        num_classes=config['data']['num_classes']
    )
    
    val_dataset = TemporalLULCDataset(
        root_dir=config['paths']['data_dir'],
        split='val',
        img_size=config['data']['img_size'],
        num_classes=config['data']['num_classes']
    )
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = CausalSpatiotemporalTransformer(
        num_classes=config['data']['num_classes'],
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        n_layers=config['model']['n_layers'],
        dropout=config['model']['dropout'],
        img_size=config['data']['img_size']
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {num_params:,}")
    
    # Initialize loss functions
    criterion = nn.CrossEntropyLoss()
    physics_loss_fn = PhysicsInformedLoss()
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'],
        eta_min=1e-6
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['metrics']['loss']
        print(f"  Resuming from epoch {start_epoch}")
    
    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'train_f1': [],
        'train_kappa': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'val_kappa': []
    }
    
    patience_counter = 0
    patience = config['training']['early_stopping_patience']
    
    # Pass training config to epoch functions
    training_config = {
        'lambda_physics': config['training']['lambda_physics'],
        'lambda_continuity': config['training']['lambda_continuity']
    }
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        print("-" * 70)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, physics_loss_fn,
            optimizer, device, training_config
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, physics_loss_fn,
            device, training_config
        )
        
        # Update learning rate
        scheduler.step()
        
        # Print metrics
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}, "
              f"Kappa: {train_metrics['kappa']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, "
              f"Kappa: {val_metrics['kappa']:.4f}")
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1'])
        history['train_kappa'].append(train_metrics['kappa'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_kappa'].append(val_metrics['kappa'])
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            best_path = checkpoint_dir / 'best_model.pth'
            save_checkpoint(model, optimizer, epoch, val_metrics, str(best_path))
            print(f"  ✓ Best model saved (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
        
        # Save latest checkpoint
        latest_path = checkpoint_dir / 'latest_model.pth'
        save_checkpoint(model, optimizer, epoch, val_metrics, str(latest_path))
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    # Load best model for final evaluation
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    
    best_checkpoint = torch.load(checkpoint_dir / 'best_model.pth', map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Final test evaluation
    print("\nEvaluating on test set...")
    test_dataset = TemporalLULCDataset(
        root_dir=config['paths']['data_dir'],
        split='test',
        img_size=config['data']['img_size'],
        num_classes=config['data']['num_classes']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    test_metrics = validate(
        model, test_loader, criterion, physics_loss_fn,
        device, training_config
    )
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    print(f"  Kappa: {test_metrics['kappa']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    
    # Save results
    save_results(history, test_metrics, config, log_dir)
    
    print("\n" + "=" * 70)
    print("All done! 🎉")
    print("=" * 70)


if __name__ == '__main__':
    main()
