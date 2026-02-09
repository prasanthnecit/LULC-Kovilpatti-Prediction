#!/usr/bin/env python
"""
Complete training script for real LULC data
Trains Spatiotemporal Transformer on real Kovilpatti satellite data
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import SpatiotemporalTransformer
from src.dataset import RealLULCDataset
from src.utils import set_seed, calculate_metrics, save_checkpoint

# Constants
BEST_MODEL_FILENAME = 'best_model_real.pth'
HISTORY_FILENAME = 'training_history_real.json'
CURVES_FILENAME = 'training_curves.png'


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for inputs, targets in pbar:
        inputs = inputs.to(device)  # (batch, seq_len, H, W)
        targets = targets.to(device)  # (batch, H, W)
        
        # Forward
        outputs, _ = model(inputs)  # (batch, num_classes, H, W)
        
        # Loss
        loss = criterion(outputs, targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Metrics
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(targets.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate metrics
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    metrics = calculate_metrics(all_preds, all_targets)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validation", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    metrics = calculate_metrics(all_preds, all_targets)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train LULC model on real data')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers (default 0 for cross-platform compatibility)')
    args = parser.parse_args()
    
    # Setup
    set_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Print header
    print("=" * 80)
    print(" 🛰️  LULC PREDICTION - REAL DATA TRAINING")
    print("=" * 80)
    print(f"📍 Region: Kovilpatti, Tamil Nadu")
    print(f"🎮 Device: {device}")
    if torch.cuda.is_available():
        print(f"💻 GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 80)
    
    # Create datasets
    print(f"\n📂 Loading data from: {args.data_dir}")
    train_dataset = RealLULCDataset(args.data_dir, split='train')
    val_dataset = RealLULCDataset(args.data_dir, split='val')
    
    print(f"  ✅ Train: {len(train_dataset)} samples")
    print(f"  ✅ Val: {len(val_dataset)} samples")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Model
    print("\n🧠 Creating model...")
    model = SpatiotemporalTransformer(
        num_classes=7,
        d_model=256,
        n_heads=8,
        n_layers=4,
        dropout=0.1
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Model parameters: {total_params:,}")
    
    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Create output directories
    Path(args.output_dir, 'checkpoints').mkdir(parents=True, exist_ok=True)
    Path(args.output_dir, 'logs').mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("\n" + "=" * 80)
    print(" 🚂 TRAINING START")
    print("=" * 80)
    
    best_val_acc = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': []
    }
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n📅 Epoch {epoch}/{args.epochs}")
        print("-" * 80)
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step()
        
        # Print results
        print(f"\n📊 Results:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}, Kappa: {train_metrics['kappa']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, Kappa: {val_metrics['kappa']:.4f}")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            checkpoint_path = f"{args.output_dir}/checkpoints/{BEST_MODEL_FILENAME}"
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                checkpoint_path
            )
            print(f"  ✨ Best model saved! Accuracy: {best_val_acc:.4f}")
        
        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])
    
    # Save results
    with open(f"{args.output_dir}/logs/{HISTORY_FILENAME}", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(history['train_f1'], label='Train')
    plt.plot(history['val_f1'], label='Val')
    plt.title('F1-Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    curves_path = f"{args.output_dir}/logs/{CURVES_FILENAME}"
    plt.savefig(curves_path, dpi=150, bbox_inches='tight')
    print(f"\n📈 Training curves saved to: {curves_path}")
    
    # Final summary
    print("\n" + "=" * 80)
    print(" ✨ TRAINING COMPLETE ✨")
    print("=" * 80)
    print(f"📊 Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"💾 Model saved: {args.output_dir}/checkpoints/{BEST_MODEL_FILENAME}")
    print(f"📈 Logs saved: {args.output_dir}/logs/")
    print("=" * 80)


if __name__ == '__main__':
    main()
