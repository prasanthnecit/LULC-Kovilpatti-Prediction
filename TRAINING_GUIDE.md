# LULC Training Implementation Guide

Complete training implementation for the LULC (Land Use Land Cover) prediction model using real satellite data from Kovilpatti, Tamil Nadu.

## 🎯 Features

- ✅ **Complete Model Architecture**: Spatiotemporal Transformer with CNN encoder, multi-head attention, and decoder
- ✅ **PyTorch Dataset**: Custom dataset class for real LULC data
- ✅ **Full Training Loop**: With validation, metrics, checkpointing, and visualization
- ✅ **GPU Support**: Optimized for CUDA with gradient clipping
- ✅ **Metrics**: Accuracy, F1-Score, Cohen's Kappa, Precision, Recall
- ✅ **Visualization**: Automatic generation of training curves

## 📦 Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- tqdm >= 4.65.0

### 2. Verify Installation

```bash
python tests/test_training_implementation.py
```

This will test:
- Model architecture and forward pass
- Dataset loading
- Utility functions
- DataLoader integration

## 🚀 Usage

### Basic Training

```bash
python scripts/run_training_real.py --data_dir data/Kovilpatti_LULC_Real/ --epochs 50 --batch_size 32
```

### Full Options

```bash
python scripts/run_training_real.py \
    --data_dir data/Kovilpatti_LULC_Real/ \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.0001 \
    --device cuda \
    --output_dir outputs \
    --num_workers 4
```

### Parameters

- `--data_dir`: Directory containing training data (train_inputs.npy, train_targets.npy, etc.)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 32)
- `--lr`: Learning rate (default: 0.0001)
- `--device`: Device to use - 'cuda' or 'cpu' (default: cuda)
- `--output_dir`: Output directory for models and logs (default: outputs)
- `--num_workers`: DataLoader workers (default: 4)

## 📊 Expected Output

### Training Progress

```
================================================================================
 🛰️  LULC PREDICTION - REAL DATA TRAINING
================================================================================
📍 Region: Kovilpatti, Tamil Nadu
🎮 Device: cuda
💻 GPU: NVIDIA RTX 4000 Ada Generation
================================================================================

📂 Loading data from: data/Kovilpatti_LULC_Real/
  ✅ Train: 62,487 samples
  ✅ Val: 13,390 samples

🧠 Creating model...
📊 Model parameters: 12,345,678

================================================================================
 🚂 TRAINING START
================================================================================

📅 Epoch 1/50
--------------------------------------------------------------------------------
Training: 100%|████████████| 1953/1953 [02:15<00:00, 14.39it/s, loss=1.2345]
Validation: 100%|██████████| 419/419 [00:25<00:00, 16.52it/s]

📊 Results:
  Train - Loss: 1.2345, Acc: 0.7234, F1: 0.7012, Kappa: 0.6789
  Val   - Loss: 1.3456, Acc: 0.7123, F1: 0.6923, Kappa: 0.6678
  ✨ Best model saved! Accuracy: 0.7123
...
```

### Generated Files

```
outputs/
├── checkpoints/
│   └── best_model_real.pth          # Best model checkpoint
└── logs/
    ├── training_history_real.json   # Training metrics
    └── training_curves.png          # Loss/Accuracy/F1 plots
```

## 🏗️ Architecture Details

### Spatiotemporal Transformer

The model consists of:

1. **Spatial Encoder** (per timestep)
   - 3 Convolutional layers (64 → 128 → 256 channels)
   - Batch normalization and ReLU activation
   - Max pooling (reduces spatial dimensions by 4x)

2. **Temporal Transformer**
   - Positional encoding for temporal sequences
   - 4 Transformer encoder blocks
   - 8-head multi-head attention
   - Feed-forward network (d_model × 4)

3. **Decoder**
   - 2 Transposed convolutions (upsampling)
   - Final convolution to num_classes (7)

**Input**: (batch, 2, 256, 256) - 2 timesteps, 256×256 resolution  
**Output**: (batch, 7, 256, 256) - 7 LULC classes

### Model Parameters

- `num_classes=7`: Urban, Forest, Agriculture, Water, Barren, Wetland, Grassland
- `d_model=256`: Hidden dimension
- `n_heads=8`: Number of attention heads
- `n_layers=4`: Number of transformer blocks
- `dropout=0.1`: Dropout rate

## 📈 Training Details

- **Optimizer**: AdamW (weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR
- **Loss**: CrossEntropyLoss
- **Gradient Clipping**: max_norm=1.0
- **Seed**: 42 (for reproducibility)

## 💾 Checkpoint Format

Checkpoints contain:
- `epoch`: Training epoch
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `metrics`: Validation metrics

### Load Checkpoint

```python
import torch
from src.model import SpatiotemporalTransformer

model = SpatiotemporalTransformer(num_classes=7, d_model=256, n_heads=8, n_layers=4)
checkpoint = torch.load('outputs/checkpoints/best_model_real.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Best accuracy: {checkpoint['metrics']['accuracy']:.4f}")
```

## 🔧 Troubleshooting

### Out of Memory

If you encounter CUDA OOM errors:

```bash
# Reduce batch size
python scripts/run_training_real.py --data_dir data/Kovilpatti_LULC_Real/ --batch_size 16

# Use CPU
python scripts/run_training_real.py --data_dir data/Kovilpatti_LULC_Real/ --device cpu
```

### Data Not Found

Ensure your data directory has:
- `train_inputs.npy`: (N, 2, 256, 256) - training inputs
- `train_targets.npy`: (N, 256, 256) - training targets
- `val_inputs.npy`: (M, 2, 256, 256) - validation inputs
- `val_targets.npy`: (M, 256, 256) - validation targets

## 🎓 Advanced Usage

### Resume Training

(Optional feature - implement if needed)

```python
# Load checkpoint and continue training
checkpoint = torch.load('outputs/checkpoints/best_model_real.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

### Custom Model Configuration

Edit model parameters in the training script:

```python
model = SpatiotemporalTransformer(
    num_classes=7,
    d_model=128,      # Smaller model
    n_heads=4,        # Fewer heads
    n_layers=2,       # Fewer layers
    dropout=0.2       # Higher dropout
)
```

## 📚 References

- **Dataset**: ESA WorldCover 10m (via Microsoft Planetary Computer)
- **Region**: Kovilpatti, Tamil Nadu (9.17°N, 77.87°E)
- **Resolution**: 10m
- **Classes**: 7 LULC categories

## ✨ Expected Performance

On NVIDIA RTX 4000 Ada Generation:
- **Training time**: ~10-15 minutes for 50 epochs
- **GPU utilization**: 70-90%
- **Memory usage**: ~8-12 GB
- **Expected accuracy**: 70-85% (depends on data quality)

## 🐛 Issues

If you encounter any issues:
1. Check that all dependencies are installed correctly
2. Verify data files exist and have correct format
3. Run the test script: `python tests/test_training_implementation.py`
4. Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`
