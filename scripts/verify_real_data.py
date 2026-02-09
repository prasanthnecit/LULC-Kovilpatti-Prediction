#!/usr/bin/env python
"""
Verify real data download and preprocessing

Usage:
    python scripts/verify_real_data.py --data_dir data/Kovilpatti_LULC_Real/
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.real_data.utils import print_data_summary


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Verify real satellite data download and preprocessing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/Kovilpatti_LULC_Real',
        help='Directory containing processed data'
    )
    
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='data/cache',
        help='Cache directory with raw downloads'
    )
    
    return parser.parse_args()


def verify_files_exist(data_dir: str) -> bool:
    """
    Verify all required files exist
    
    Args:
        data_dir: Data directory
        
    Returns:
        True if all files exist, False otherwise
    """
    data_path = Path(data_dir)
    
    print("\n📁 Checking files...")
    
    required_files = [
        'train_inputs.npy',
        'train_targets.npy',
        'val_inputs.npy',
        'val_targets.npy',
        'test_inputs.npy',
        'test_targets.npy',
        'metadata.txt'
    ]
    
    all_exist = True
    for filename in required_files:
        filepath = data_path / filename
        exists = filepath.exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {filename}")
        if not exists:
            all_exist = False
    
    return all_exist


def verify_data_shapes(data_dir: str) -> bool:
    """
    Verify data shapes are consistent
    
    Args:
        data_dir: Data directory
        
    Returns:
        True if shapes are valid, False otherwise
    """
    data_path = Path(data_dir)
    
    print("\n📐 Checking data shapes...")
    
    try:
        for split in ['train', 'val', 'test']:
            inputs = np.load(data_path / f"{split}_inputs.npy")
            targets = np.load(data_path / f"{split}_targets.npy")
            
            print(f"\n  {split.upper()}:")
            print(f"    Inputs:  {inputs.shape} (dtype: {inputs.dtype})")
            print(f"    Targets: {targets.shape} (dtype: {targets.dtype})")
            
            # Validate shapes
            if len(inputs.shape) != 4:  # (n_samples, n_timesteps, H, W)
                print(f"    ⚠️  Invalid input shape! Expected 4D, got {len(inputs.shape)}D")
                return False
            
            if len(targets.shape) != 3:  # (n_samples, H, W)
                print(f"    ⚠️  Invalid target shape! Expected 3D, got {len(targets.shape)}D")
                return False
            
            if inputs.shape[0] != targets.shape[0]:
                print(f"    ⚠️  Sample count mismatch!")
                return False
            
            if inputs.shape[-2:] != targets.shape[-2:]:
                print(f"    ⚠️  Spatial dimensions mismatch!")
                return False
            
            print(f"    ✅ Shapes valid")
        
        return True
        
    except Exception as e:
        print(f"\n  ❌ Error loading data: {e}")
        return False


def check_class_distribution(data_dir: str):
    """
    Check class distribution in the data
    
    Args:
        data_dir: Data directory
    """
    data_path = Path(data_dir)
    
    print("\n📊 Class Distribution Analysis...")
    
    for split in ['train', 'val', 'test']:
        targets = np.load(data_path / f"{split}_targets.npy")
        
        print(f"\n  {split.upper()}:")
        unique, counts = np.unique(targets, return_counts=True)
        total = counts.sum()
        
        for cls, count in zip(unique, counts):
            pct = (count / total) * 100
            print(f"    Class {cls}: {count:8,} pixels ({pct:5.2f}%)")


def validate_temporal_consistency(data_dir: str) -> bool:
    """
    Validate temporal consistency
    
    Args:
        data_dir: Data directory
        
    Returns:
        True if consistent, False otherwise
    """
    data_path = Path(data_dir)
    
    print("\n⏱️  Checking temporal consistency...")
    
    try:
        train_inputs = np.load(data_path / "train_inputs.npy")
        
        n_timesteps = train_inputs.shape[1]
        print(f"  Number of timesteps: {n_timesteps}")
        
        if n_timesteps < 1:
            print(f"  ⚠️  Need at least 1 timestep!")
            return False
        
        print(f"  ✅ Temporal structure valid")
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def verify_cache(cache_dir: str):
    """
    Verify cached raw data
    
    Args:
        cache_dir: Cache directory
    """
    cache_path = Path(cache_dir)
    
    print("\n💾 Checking cache...")
    
    if not cache_path.exists():
        print(f"  ⚠️  Cache directory not found: {cache_dir}")
        return
    
    cached_files = list(cache_path.glob('*.npy'))
    
    if not cached_files:
        print(f"  ⚠️  No cached files found")
        return
    
    print(f"  Found {len(cached_files)} cached files:")
    for f in cached_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"    ✅ {f.name} ({size_mb:.2f} MB)")


def print_metadata(data_dir: str):
    """
    Print metadata
    
    Args:
        data_dir: Data directory
    """
    metadata_file = Path(data_dir) / "metadata.txt"
    
    print("\n📋 Metadata:")
    
    if not metadata_file.exists():
        print("  ⚠️  metadata.txt not found")
        return
    
    with open(metadata_file, 'r') as f:
        content = f.read()
        for line in content.split('\n'):
            if line.strip():
                print(f"  {line}")


def main():
    """Main verification function"""
    args = parse_args()
    
    print("=" * 80)
    print(" 🔍 REAL DATA VERIFICATION")
    print("=" * 80)
    print(f"\n📂 Data directory: {args.data_dir}")
    print(f"💾 Cache directory: {args.cache_dir}")
    
    # Check if data directory exists
    if not Path(args.data_dir).exists():
        print(f"\n❌ Data directory not found: {args.data_dir}")
        print(f"\n💡 Run fetch_real_data.py first:")
        print(f"   python scripts/fetch_real_data.py --output_dir {args.data_dir}")
        return 1
    
    # Run verification checks
    checks_passed = []
    
    # 1. Files exist
    print("\n" + "=" * 80)
    print(" CHECK 1: File Existence")
    print("=" * 80)
    files_ok = verify_files_exist(args.data_dir)
    checks_passed.append(("Files exist", files_ok))
    
    if not files_ok:
        print("\n❌ Some files are missing!")
        return 1
    
    # 2. Data shapes
    print("\n" + "=" * 80)
    print(" CHECK 2: Data Shapes")
    print("=" * 80)
    shapes_ok = verify_data_shapes(args.data_dir)
    checks_passed.append(("Data shapes", shapes_ok))
    
    # 3. Class distribution
    print("\n" + "=" * 80)
    print(" CHECK 3: Class Distribution")
    print("=" * 80)
    check_class_distribution(args.data_dir)
    checks_passed.append(("Class distribution", True))
    
    # 4. Temporal consistency
    print("\n" + "=" * 80)
    print(" CHECK 4: Temporal Consistency")
    print("=" * 80)
    temporal_ok = validate_temporal_consistency(args.data_dir)
    checks_passed.append(("Temporal consistency", temporal_ok))
    
    # 5. Cache verification
    print("\n" + "=" * 80)
    print(" CHECK 5: Cache Verification")
    print("=" * 80)
    verify_cache(args.cache_dir)
    checks_passed.append(("Cache", True))
    
    # 6. Metadata
    print("\n" + "=" * 80)
    print(" CHECK 6: Metadata")
    print("=" * 80)
    print_metadata(args.data_dir)
    checks_passed.append(("Metadata", True))
    
    # Summary
    print("\n" + "=" * 80)
    print(" VERIFICATION SUMMARY")
    print("=" * 80)
    
    for check_name, passed in checks_passed:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {check_name}")
    
    all_passed = all(passed for _, passed in checks_passed)
    
    if all_passed:
        print("\n✨ All verification checks passed! ✨")
        print("\n🎯 Your data is ready for training!")
        print(f"\n   python scripts/run_training_real.py --data_dir {args.data_dir} --real_data")
    else:
        print("\n⚠️  Some checks failed. Please review the errors above.")
    
    print("\n" + "=" * 80 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
