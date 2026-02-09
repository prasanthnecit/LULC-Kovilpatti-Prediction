#!/usr/bin/env python
"""
Test script to validate the model architecture and training components
Run this after installing PyTorch to ensure everything works correctly
"""

import sys
import os
import numpy as np
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_model_architecture():
    """Test model can be instantiated and forward pass works"""
    print("=" * 80)
    print("Testing Model Architecture")
    print("=" * 80)
    
    try:
        import torch
        from src.model import SpatiotemporalTransformer
        
        # Create model
        model = SpatiotemporalTransformer(
            num_classes=7,
            d_model=256,
            n_heads=8,
            n_layers=4,
            dropout=0.1
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created successfully")
        print(f"   Total parameters: {total_params:,}")
        
        # Test forward pass with dummy data
        batch_size = 2
        seq_len = 2
        H, W = 256, 256
        
        # Create dummy input (batch, seq_len, H, W)
        x = torch.randint(0, 7, (batch_size, seq_len, H, W))
        
        # Forward pass
        output, attn_weights = model(x)
        
        print(f"✅ Forward pass successful")
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Expected: ({batch_size}, 7, {H}, {W})")
        print(f"   Attention weights: {len(attn_weights)} layers")
        
        assert output.shape == (batch_size, 7, H, W), "Output shape mismatch!"
        print("✅ Output shape correct!")
        
        return True
        
    except ImportError:
        print("❌ PyTorch not installed. Install with: pip install torch>=2.0.0")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset():
    """Test dataset class"""
    print("\n" + "=" * 80)
    print("Testing Dataset")
    print("=" * 80)
    
    test_dir = None
    try:
        import torch
        from src.dataset import RealLULCDataset
        
        # Create temporary directory for test data
        test_dir = tempfile.mkdtemp()
        
        # Create dummy numpy arrays
        num_samples = 10
        seq_len = 2
        H, W = 256, 256
        
        train_inputs = np.random.randint(0, 7, (num_samples, seq_len, H, W), dtype=np.int32)
        train_targets = np.random.randint(0, 7, (num_samples, H, W), dtype=np.int32)
        
        np.save(os.path.join(test_dir, "train_inputs.npy"), train_inputs)
        np.save(os.path.join(test_dir, "train_targets.npy"), train_targets)
        
        # Create dataset
        dataset = RealLULCDataset(test_dir, split='train')
        
        print(f"✅ Dataset created successfully")
        print(f"   Number of samples: {len(dataset)}")
        
        # Test getting item
        inputs, targets = dataset[0]
        print(f"✅ Dataset __getitem__ works")
        print(f"   Input shape: {inputs.shape}")
        print(f"   Target shape: {targets.shape}")
        
        assert inputs.shape == (seq_len, H, W), "Input shape mismatch!"
        assert targets.shape == (H, W), "Target shape mismatch!"
        print("✅ Data shapes correct!")
        
        return True
        
    except ImportError:
        print("❌ PyTorch not installed. Install with: pip install torch>=2.0.0")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if test_dir and os.path.exists(test_dir):
            shutil.rmtree(test_dir)


def test_utils():
    """Test utility functions"""
    print("\n" + "=" * 80)
    print("Testing Utilities")
    print("=" * 80)
    
    try:
        from src.utils import set_seed, calculate_metrics
        
        # Test set_seed
        set_seed(42)
        print("✅ set_seed() works")
        
        # Test calculate_metrics
        predictions = np.random.randint(0, 7, (100, 100))
        targets = np.random.randint(0, 7, (100, 100))
        
        metrics = calculate_metrics(predictions, targets)
        
        print("✅ calculate_metrics() works")
        print(f"   Metrics: {list(metrics.keys())}")
        
        assert 'accuracy' in metrics, "Missing accuracy metric"
        assert 'f1' in metrics, "Missing f1 metric"
        assert 'kappa' in metrics, "Missing kappa metric"
        print("✅ All expected metrics present!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader():
    """Test DataLoader integration"""
    print("\n" + "=" * 80)
    print("Testing DataLoader Integration")
    print("=" * 80)
    
    test_dir = None
    try:
        import torch
        from torch.utils.data import DataLoader
        from src.dataset import RealLULCDataset
        
        # Create temporary directory for test data
        test_dir = tempfile.mkdtemp()
        
        num_samples = 10
        seq_len = 2
        H, W = 256, 256
        
        train_inputs = np.random.randint(0, 7, (num_samples, seq_len, H, W), dtype=np.int32)
        train_targets = np.random.randint(0, 7, (num_samples, H, W), dtype=np.int32)
        
        np.save(os.path.join(test_dir, "train_inputs.npy"), train_inputs)
        np.save(os.path.join(test_dir, "train_targets.npy"), train_targets)
        
        # Create dataset and dataloader
        dataset = RealLULCDataset(test_dir, split='train')
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        print(f"✅ DataLoader created successfully")
        
        # Test iteration
        for batch_inputs, batch_targets in dataloader:
            print(f"✅ DataLoader iteration works")
            print(f"   Batch input shape: {batch_inputs.shape}")
            print(f"   Batch target shape: {batch_targets.shape}")
            break
        
        return True
        
    except ImportError:
        print("❌ PyTorch not installed. Install with: pip install torch>=2.0.0")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if test_dir and os.path.exists(test_dir):
            shutil.rmtree(test_dir)


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print(" 🧪 TESTING LULC TRAINING IMPLEMENTATION")
    print("=" * 80 + "\n")
    
    results = []
    
    # Test each component
    results.append(("Model Architecture", test_model_architecture()))
    results.append(("Dataset", test_dataset()))
    results.append(("Utilities", test_utils()))
    results.append(("DataLoader Integration", test_dataloader()))
    
    # Print summary
    print("\n" + "=" * 80)
    print(" 📊 TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print("=" * 80)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 80 + "\n")
    
    if passed == total:
        print("🎉 All tests passed! The implementation is ready for training.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
