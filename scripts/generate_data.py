#!/usr/bin/env python3
"""
Script to generate synthetic LULC dataset for Kovilpatti region.
"""

import argparse
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generator import KovilpattiLULCGenerator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic LULC dataset for Kovilpatti'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/Kovilpatti_LULC',
        help='Output directory for generated data'
    )
    parser.add_argument(
        '--img-size',
        type=int,
        default=256,
        help='Size of LULC maps'
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=7,
        help='Number of LULC classes'
    )
    parser.add_argument(
        '--train-samples',
        type=int,
        default=200,
        help='Number of training samples'
    )
    parser.add_argument(
        '--val-samples',
        type=int,
        default=40,
        help='Number of validation samples'
    )
    parser.add_argument(
        '--test-samples',
        type=int,
        default=40,
        help='Number of test samples'
    )
    parser.add_argument(
        '--num-timesteps',
        type=int,
        default=5,
        help='Number of timesteps per sequence (default: 5 for 2018-2022)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()


def generate_dataset(
    generator: KovilpattiLULCGenerator,
    num_samples: int,
    num_timesteps: int,
    output_dir: Path
) -> None:
    """
    Generate dataset for a split.
    
    Args:
        generator: LULC generator
        num_samples: Number of samples to generate
        num_timesteps: Number of timesteps per sequence
        output_dir: Output directory for this split
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_samples):
        # Generate temporal sequence
        sequence = generator.generate_temporal_sequence(num_timesteps)
        sequence = np.stack(sequence, axis=0)  # (T, H, W)
        
        # Save as numpy file
        save_path = output_dir / f'sequence_{i:04d}.npy'
        np.save(save_path, sequence)
        
        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{num_samples} sequences")
    
    print(f"  ✓ Saved {num_samples} sequences to {output_dir}")


def main():
    """Main function."""
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("Kovilpatti LULC Dataset Generation")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Image size: {args.img_size}x{args.img_size}")
    print(f"  Number of classes: {args.num_classes}")
    print(f"  Timesteps per sequence: {args.num_timesteps}")
    print(f"  Train samples: {args.train_samples}")
    print(f"  Validation samples: {args.val_samples}")
    print(f"  Test samples: {args.test_samples}")
    print(f"  Random seed: {args.seed}")
    print()
    
    # Initialize generator
    generator = KovilpattiLULCGenerator(
        img_size=args.img_size,
        num_classes=args.num_classes
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    
    # Generate train set
    print("Generating training set...")
    train_dir = output_dir / 'train'
    generate_dataset(generator, args.train_samples, args.num_timesteps, train_dir)
    
    # Generate validation set
    print("\nGenerating validation set...")
    val_dir = output_dir / 'val'
    generate_dataset(generator, args.val_samples, args.num_timesteps, val_dir)
    
    # Generate test set
    print("\nGenerating test set...")
    test_dir = output_dir / 'test'
    generate_dataset(generator, args.test_samples, args.num_timesteps, test_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Dataset Generation Complete!")
    print("=" * 60)
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {args.train_samples + args.val_samples + args.test_samples}")
    print(f"  Train: {args.train_samples} sequences")
    print(f"  Validation: {args.val_samples} sequences")
    print(f"  Test: {args.test_samples} sequences")
    print(f"\nEach sequence contains {args.num_timesteps} timesteps")
    print(f"Each LULC map is {args.img_size}x{args.img_size} pixels")
    print(f"\nLULC Classes ({args.num_classes}):")
    for i, name in enumerate(generator.class_names):
        print(f"  {i}: {name}")
    print()


if __name__ == '__main__':
    main()
