# 🚀 Quick Start - Training Your Model

This guide gets you training in 5 minutes!

## Prerequisites

✅ Real data downloaded (62,487 train samples, 13,390 val samples)  
✅ GPU ready (NVIDIA RTX 4000 Ada Generation or similar)  
✅ Python 3.8+ installed

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- PyTorch 2.0+
- NumPy, scikit-learn
- matplotlib, tqdm
- All other required packages

## Step 2: Verify Installation

```bash
python tests/test_training_implementation.py
```

Expected output:
```
🧪 TESTING LULC TRAINING IMPLEMENTATION
================================================================================
Testing Model Architecture
================================================================================
✅ Model created successfully
   Total parameters: 12,345,678
...
Total: 4/4 tests passed
🎉 All tests passed! The implementation is ready for training.
```

## Step 3: Start Training

```bash
python scripts/run_training_real.py \
    --data_dir data/Kovilpatti_LULC_Real/ \
    --epochs 50 \
    --batch_size 32
```

### What Happens:
1. ✅ Loads 62,487 training samples
2. ✅ Loads 13,390 validation samples  
3. ✅ Creates model with ~12M parameters
4. ✅ Trains for 50 epochs with progress bars
5. ✅ Saves best model automatically
6. ✅ Generates training curves

### Expected Output:
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
```

## Step 4: Check Results

After training completes, you'll have:

```
outputs/
├── checkpoints/
│   └── best_model_real.pth          # Your trained model!
└── logs/
    ├── training_history_real.json   # Training metrics
    └── training_curves.png          # Visualization
```

## Customization

### Use CPU Instead of GPU
```bash
python scripts/run_training_real.py \
    --data_dir data/Kovilpatti_LULC_Real/ \
    --device cpu \
    --batch_size 16
```

### More Epochs
```bash
python scripts/run_training_real.py \
    --data_dir data/Kovilpatti_LULC_Real/ \
    --epochs 100
```

### Lower Learning Rate
```bash
python scripts/run_training_real.py \
    --data_dir data/Kovilpatti_LULC_Real/ \
    --lr 0.00005
```

### All Options
```bash
python scripts/run_training_real.py --help
```

## Troubleshooting

### Out of Memory?
```bash
# Reduce batch size
python scripts/run_training_real.py --data_dir data/Kovilpatti_LULC_Real/ --batch_size 16

# Or use CPU
python scripts/run_training_real.py --data_dir data/Kovilpatti_LULC_Real/ --device cpu
```

### Data Not Found?
Ensure your data directory has:
- `train_inputs.npy`
- `train_targets.npy`
- `val_inputs.npy`
- `val_targets.npy`

### Import Errors?
```bash
pip install -r requirements.txt --upgrade
```

## What's Next?

After training:
1. Load your model: See `TRAINING_GUIDE.md`
2. Make predictions on new data
3. Visualize attention maps
4. Fine-tune hyperparameters

## Need Help?

📖 Full documentation: `TRAINING_GUIDE.md`  
🔧 Technical details: `IMPLEMENTATION_COMPLETE.md`  
💻 Code: `src/` directory

---

**Happy Training! 🎉**
