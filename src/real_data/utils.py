"""
Utility functions for real satellite data handling
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from pathlib import Path


# Class names and colors for visualization
CLASS_NAMES = [
    "Urban",
    "Forest",
    "Agriculture",
    "Water",
    "Barren",
    "Wetland",
    "Grassland"
]

CLASS_COLORS = [
    '#FF0000',  # Urban - Red
    '#00FF00',  # Forest - Green
    '#FFFF00',  # Agriculture - Yellow
    '#0000FF',  # Water - Blue
    '#A0522D',  # Barren - Brown
    '#00FFFF',  # Wetland - Cyan
    '#90EE90',  # Grassland - Light Green
]


def get_kovilpatti_bbox() -> list:
    """
    Get bounding box for Kovilpatti region
    
    Returns:
        [min_lon, min_lat, max_lon, max_lat]
    """
    # Kovilpatti: 9.17°N, 77.87°E, ~10km radius
    return [77.77, 9.07, 77.97, 9.27]


def visualize_real_data(lulc_array: np.ndarray, 
                       year: int,
                       title: str = None,
                       save_path: str = None,
                       cmap: str = 'tab10'):
    """
    Visualize real LULC data
    
    Args:
        lulc_array: LULC array
        year: Year of data
        title: Plot title
        save_path: Path to save figure
        cmap: Colormap to use
    """
    plt.figure(figsize=(12, 10))
    
    if title is None:
        title = f"LULC Map - {year}"
    
    plt.imshow(lulc_array, cmap=cmap, interpolation='nearest')
    plt.colorbar(label='LULC Class', shrink=0.8)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Add statistics
    unique, counts = np.unique(lulc_array, return_counts=True)
    stats_text = f"Shape: {lulc_array.shape}\n"
    stats_text += f"Classes: {len(unique)}\n"
    stats_text += f"Range: {lulc_array.min()}-{lulc_array.max()}"
    
    plt.text(0.02, 0.98, stats_text,
            transform=plt.gca().transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            verticalalignment='top',
            fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  💾 Saved visualization: {save_path}")
    
    plt.show()


def visualize_class_distribution(lulc_array: np.ndarray,
                                 year: int,
                                 save_path: str = None):
    """
    Visualize class distribution as bar chart
    
    Args:
        lulc_array: LULC array
        year: Year of data
        save_path: Path to save figure
    """
    unique, counts = np.unique(lulc_array, return_counts=True)
    percentages = (counts / counts.sum()) * 100
    
    plt.figure(figsize=(12, 6))
    
    # Bar chart
    plt.subplot(1, 2, 1)
    bars = plt.bar(range(len(unique)), percentages)
    plt.xlabel('LULC Class')
    plt.ylabel('Percentage (%)')
    plt.title(f'Class Distribution - {year}')
    plt.xticks(range(len(unique)), unique)
    plt.grid(axis='y', alpha=0.3)
    
    # Pie chart
    plt.subplot(1, 2, 2)
    plt.pie(percentages, labels=[f'Class {c}' for c in unique],
           autopct='%1.1f%%', startangle=90)
    plt.title(f'Class Proportions - {year}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  💾 Saved distribution plot: {save_path}")
    
    plt.show()


def compare_real_vs_synthetic(real_data: np.ndarray,
                              synthetic_data: np.ndarray,
                              year: int):
    """
    Compare real and synthetic LULC data
    
    Args:
        real_data: Real LULC array
        synthetic_data: Synthetic LULC array
        year: Year
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Real data
    axes[0].imshow(real_data, cmap='tab10', interpolation='nearest')
    axes[0].set_title(f'Real Satellite Data - {year}', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    
    # Synthetic data
    axes[1].imshow(synthetic_data, cmap='tab10', interpolation='nearest')
    axes[1].set_title(f'Synthetic Data - {year}', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    
    plt.tight_layout()
    plt.show()
    
    # Compare distributions
    print("\n📊 Distribution Comparison:")
    print("\nReal Data:")
    real_unique, real_counts = np.unique(real_data, return_counts=True)
    for cls, count in zip(real_unique, real_counts):
        pct = (count / real_counts.sum()) * 100
        print(f"  Class {cls}: {pct:.2f}%")
    
    print("\nSynthetic Data:")
    syn_unique, syn_counts = np.unique(synthetic_data, return_counts=True)
    for cls, count in zip(syn_unique, syn_counts):
        pct = (count / syn_counts.sum()) * 100
        print(f"  Class {cls}: {pct:.2f}%")


def calculate_real_transition_matrix(years_data: Dict[int, np.ndarray],
                                     n_classes: int = 7) -> np.ndarray:
    """
    Calculate actual transition probabilities from real data
    
    Args:
        years_data: Dictionary of year -> LULC array
        n_classes: Number of classes
        
    Returns:
        Transition matrix (n_classes x n_classes)
    """
    print(f"\n🔄 Calculating transition matrix from real data...")
    
    years = sorted(years_data.keys())
    if len(years) < 2:
        print("  ⚠️  Need at least 2 years")
        return np.eye(n_classes)
    
    # Initialize transition count matrix
    transition_counts = np.zeros((n_classes, n_classes), dtype=np.int64)
    
    # Count transitions between consecutive years
    for i in range(len(years) - 1):
        year1 = years[i]
        year2 = years[i + 1]
        
        data1 = years_data[year1]
        data2 = years_data[year2]
        
        # Ensure same shape
        min_h = min(data1.shape[0], data2.shape[0])
        min_w = min(data1.shape[1], data2.shape[1])
        data1 = data1[:min_h, :min_w]
        data2 = data2[:min_h, :min_w]
        
        # Count transitions
        for from_class in range(n_classes):
            mask = (data1 == from_class)
            if mask.any():
                to_classes = data2[mask]
                for to_class in range(n_classes):
                    transition_counts[from_class, to_class] += np.sum(to_classes == to_class)
        
        print(f"  Processed {year1} → {year2}")
    
    # Normalize to get probabilities
    transition_matrix = np.zeros((n_classes, n_classes), dtype=np.float32)
    for i in range(n_classes):
        row_sum = transition_counts[i].sum()
        if row_sum > 0:
            transition_matrix[i] = transition_counts[i] / row_sum
        else:
            # No transitions from this class, set diagonal to 1
            transition_matrix[i, i] = 1.0
    
    print(f"  ✅ Transition matrix calculated")
    print(f"\nTransition Matrix:")
    print(transition_matrix)
    
    return transition_matrix


def export_to_geotiff(array: np.ndarray,
                      output_path: str,
                      metadata: Dict = None):
    """
    Export array as GeoTIFF
    
    Args:
        array: LULC array
        output_path: Output file path
        metadata: Metadata dictionary (bounds, crs, etc.)
    """
    try:
        import rasterio
        from rasterio.transform import from_bounds
        
        if metadata is None:
            # Use Kovilpatti default bounds
            bbox = get_kovilpatti_bbox()
            metadata = {
                'bounds': bbox,
                'crs': 'EPSG:4326'
            }
        
        # Create transform
        height, width = array.shape
        bounds = metadata.get('bounds', get_kovilpatti_bbox())
        transform = from_bounds(*bounds, width, height)
        
        # Write GeoTIFF
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=array.dtype,
            crs=metadata.get('crs', 'EPSG:4326'),
            transform=transform,
            compress='lzw'
        ) as dst:
            dst.write(array, 1)
        
        print(f"  ✅ Exported to GeoTIFF: {output_path}")
        
    except ImportError:
        print("  ⚠️  rasterio not available, cannot export to GeoTIFF")
    except Exception as e:
        print(f"  ❌ Error exporting GeoTIFF: {e}")


def print_data_summary(data: np.ndarray, title: str = "Data Summary"):
    """
    Print summary statistics for LULC data
    
    Args:
        data: LULC array
        title: Title for summary
    """
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    print(f"  Shape: {data.shape}")
    print(f"  Dtype: {data.dtype}")
    print(f"  Range: {data.min()} - {data.max()}")
    print(f"  Total pixels: {data.size:,}")
    
    unique, counts = np.unique(data, return_counts=True)
    print(f"\n  Class Distribution:")
    for cls, count in zip(unique, counts):
        pct = (count / data.size) * 100
        class_name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"Class {cls}"
        print(f"    {class_name:12s}: {count:8,} pixels ({pct:5.2f}%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test utilities
    print("🧪 Testing utilities...")
    
    # Test bbox
    bbox = get_kovilpatti_bbox()
    print(f"Kovilpatti BBox: {bbox}")
    
    # Test visualization with dummy data
    dummy_data = np.random.randint(0, 7, (500, 500))
    print_data_summary(dummy_data, "Dummy LULC Data")
