#!/usr/bin/env python
"""
Training script for real satellite data
Supports both real and synthetic LULC data

Usage:
    python scripts/run_training_real.py --data_dir data/Kovilpatti_LULC_Real/ --real_data
    python scripts/run_training_real.py --data_dir data/synthetic/ 
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train LULC prediction model on real or synthetic data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing training data'
    )
    
    parser.add_argument(
        '--real_data',
        action='store_true',
        help='Flag to indicate using real satellite data'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs',
        help='Output directory for models and logs'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for training'
    )
    
    return parser.parse_args()


def load_data(data_dir: str):
    """
    Load training data
    
    Args:
        data_dir: Directory containing .npy files
        
    Returns:
        Dictionary with train/val/test splits
    """
    data_path = Path(data_dir)
    
    print(f"\n📂 Loading data from: {data_dir}")
    
    data = {}
    for split in ['train', 'val', 'test']:
        input_file = data_path / f"{split}_inputs.npy"
        target_file = data_path / f"{split}_targets.npy"
        
        if not input_file.exists() or not target_file.exists():
            print(f"  ⚠️  {split} data not found, skipping...")
            continue
        
        data[split] = {
            'inputs': np.load(input_file),
            'targets': np.load(target_file)
        }
        
        print(f"  ✅ {split}: {len(data[split]['inputs'])} samples")
    
    # Load metadata if available
    metadata_file = data_path / "metadata.txt"
    if metadata_file.exists():
        print(f"\n📋 Metadata:")
        with open(metadata_file, 'r') as f:
            print(f.read())
    
    return data


def detect_data_source(data_dir: str):
    """
    Detect if data is real or synthetic
    
    Args:
        data_dir: Data directory
        
    Returns:
        Tuple of (is_real, info_dict)
    """
    data_path = Path(data_dir)
    
    # Check for real data indicators
    is_real = 'real' in data_dir.lower() or 'kovilpatti' in data_dir.lower()
    
    info = {
        'source': 'Real Satellite Data' if is_real else 'Synthetic Data',
        'region': 'Kovilpatti, Tamil Nadu' if is_real else 'Synthetic',
        'location': '9.17°N, 77.87°E' if is_real else 'N/A'
    }
    
    # Try to detect years from cached data
    cache_dir = Path('data/cache')
    if cache_dir.exists():
        cached_files = list(cache_dir.glob('*.npy'))
        if cached_files:
            years = set()
            for f in cached_files:
                # Extract year from filename like "Kovilpatti_2020_lulc.npy"
                parts = f.stem.split('_')
                for part in parts:
                    if part.isdigit() and len(part) == 4:
                        years.add(int(part))
            if years:
                info['years'] = sorted(years)
    
    return is_real, info


def train_model(data: dict, args):
    """
    Train the model
    
    Args:
        data: Dictionary with train/val/test data
        args: Training arguments
    """
    print("\n" + "=" * 80)
    print(" TRAINING")
    print("=" * 80)
    
    # TODO: Implement actual training loop
    # This is a placeholder that demonstrates the interface
    
    print("\n⚠️  Note: Full training implementation requires model definition")
    print("This script provides the data loading interface for real satellite data")
    
    print(f"\n📊 Training configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {args.device}")
    
    # Print data shapes
    if 'train' in data:
        print(f"\n📐 Data shapes:")
        print(f"  Input: {data['train']['inputs'].shape}")
        print(f"  Target: {data['train']['targets'].shape}")
    
    # Placeholder for training loop
    print(f"\n💡 To complete training, you need to:")
    print(f"  1. Define your model architecture")
    print(f"  2. Create DataLoaders from the loaded numpy arrays")
    print(f"  3. Implement training loop with your model")
    print(f"  4. Save trained model to {args.output_dir}")


def main():
    """Main execution function"""
    args = parse_args()
    
    print("=" * 80)
    print(" 🧠 LULC PREDICTION MODEL TRAINING")
    print("=" * 80)
    
    # Detect data source
    is_real, info = detect_data_source(args.data_dir)
    
    if args.real_data or is_real:
        print("\n🛰️  Training Mode: REAL SATELLITE DATA")
        print(f"📍 Region: {info['region']}")
        print(f"🌍 Location: {info['location']}")
        if 'years' in info:
            print(f"📅 Data Years: {info['years']}")
        print(f"📡 Source: Microsoft Planetary Computer / ESA WorldCover")
        print(f"🎯 Resolution: 10m")
    else:
        print("\n🔧 Training Mode: SYNTHETIC DATA")
    
    # Load data
    data = load_data(args.data_dir)
    
    if not data:
        print("\n❌ No data loaded! Check your data directory.")
        return 1
    
    # Train model
    try:
        train_model(data, args)
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "=" * 80)
    print(" ✨ TRAINING COMPLETE ✨")
    print("=" * 80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
