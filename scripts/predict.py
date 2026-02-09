#!/usr/bin/env python3
"""
Prediction script for LULC forecasting.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import CausalSpatiotemporalTransformer
from src.data_generator import KovilpattiLULCGenerator
from src.utils import visualize_prediction, predict_future_multistep


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Make LULC predictions using trained model'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input sequence (.npy file with shape (T, H, W))'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./outputs/predictions',
        help='Output directory for predictions'
    )
    parser.add_argument(
        '--num-future-steps',
        type=int,
        default=0,
        help='Number of future steps to predict (0 = single step)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu, auto-detect if not specified)'
    )
    parser.add_argument(
        '--img-size',
        type=int,
        default=256,
        help='Image size'
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=7,
        help='Number of LULC classes'
    )
    
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device, img_size: int, num_classes: int):
    """Load trained model from checkpoint."""
    # Initialize model (using default architecture)
    model = CausalSpatiotemporalTransformer(
        num_classes=num_classes,
        d_model=256,
        n_heads=8,
        n_layers=4,
        dropout=0.1,
        img_size=img_size
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def lulc_to_onehot(lulc: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert LULC map to one-hot encoding."""
    H, W = lulc.shape
    onehot = np.zeros((num_classes, H, W), dtype=np.float32)
    
    for class_id in range(num_classes):
        onehot[class_id] = (lulc == class_id).astype(np.float32)
    
    return onehot


def main():
    """Main prediction function."""
    args = parse_args()
    
    print("=" * 70)
    print("LULC Prediction - Kovilpatti Region")
    print("=" * 70)
    
    # Set up device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nDevice: {device}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, device, args.img_size, args.num_classes)
    print("  ✓ Model loaded successfully")
    
    # Load input sequence
    print(f"\nLoading input sequence from: {args.input}")
    sequence = np.load(args.input)
    print(f"  Input shape: {sequence.shape}")
    
    # Ensure we have at least 4 timesteps for input
    if sequence.shape[0] < 4:
        raise ValueError(f"Input sequence must have at least 4 timesteps, got {sequence.shape[0]}")
    
    # Use first 4 timesteps as input
    input_sequence = sequence[:4]
    
    # Convert to one-hot
    input_onehot = []
    for lulc in input_sequence:
        onehot = lulc_to_onehot(lulc, args.num_classes)
        input_onehot.append(onehot)
    
    input_tensor = torch.from_numpy(np.stack(input_onehot, axis=0)).float()  # (T, C, H, W)
    
    # Initialize generator for visualization
    generator = KovilpattiLULCGenerator(
        img_size=args.img_size,
        num_classes=args.num_classes
    )
    
    # Make prediction
    print("\nMaking prediction...")
    
    if args.num_future_steps > 0:
        # Multi-step prediction
        predictions = predict_future_multistep(
            model, input_tensor, args.num_future_steps, device
        )
        
        print(f"  ✓ Generated {len(predictions)} future predictions")
        
        # Save and visualize each prediction
        for i, pred in enumerate(predictions):
            pred_class = torch.argmax(pred, dim=0).cpu().numpy()
            
            # Save as numpy
            pred_path = output_dir / f'prediction_step_{i+1}.npy'
            np.save(pred_path, pred_class)
            
            # Visualize
            pred_rgb = generator.lulc_to_rgb(pred_class)
            
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(pred_rgb)
            ax.set_title(f'Prediction - Future Step {i+1}', fontsize=14)
            ax.axis('off')
            
            img_path = output_dir / f'prediction_step_{i+1}.png'
            plt.savefig(img_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"    Saved: {pred_path}")
            print(f"    Saved: {img_path}")
        
        # Create combined visualization
        fig, axes = plt.subplots(1, len(predictions), figsize=(6 * len(predictions), 6))
        if len(predictions) == 1:
            axes = [axes]
        
        for i, pred in enumerate(predictions):
            pred_class = torch.argmax(pred, dim=0).cpu().numpy()
            pred_rgb = generator.lulc_to_rgb(pred_class)
            axes[i].imshow(pred_rgb)
            axes[i].set_title(f'Future Step {i+1}', fontsize=12)
            axes[i].axis('off')
        
        combined_path = output_dir / 'predictions_combined.png'
        plt.tight_layout()
        plt.savefig(combined_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n  ✓ Combined visualization: {combined_path}")
        
    else:
        # Single-step prediction
        with torch.no_grad():
            pred, attn = model(input_tensor.unsqueeze(0).to(device))
            pred = pred.squeeze(0).cpu()
        
        pred_class = torch.argmax(pred, dim=0).numpy()
        
        # Save prediction
        pred_path = output_dir / 'prediction.npy'
        np.save(pred_path, pred_class)
        print(f"  ✓ Saved prediction: {pred_path}")
        
        # Visualize
        pred_rgb = generator.lulc_to_rgb(pred_class)
        
        # Create comparison with last input if available
        if sequence.shape[0] >= 5:
            target = sequence[4]
            target_rgb = generator.lulc_to_rgb(target)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            axes[0].imshow(target_rgb)
            axes[0].set_title('Ground Truth (if available)', fontsize=14)
            axes[0].axis('off')
            
            axes[1].imshow(pred_rgb)
            axes[1].set_title('Prediction', fontsize=14)
            axes[1].axis('off')
        else:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(pred_rgb)
            ax.set_title('Prediction', fontsize=14)
            ax.axis('off')
        
        img_path = output_dir / 'prediction.png'
        plt.tight_layout()
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved visualization: {img_path}")
    
    print("\n" + "=" * 70)
    print("Prediction Complete! 🎉")
    print("=" * 70)
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
