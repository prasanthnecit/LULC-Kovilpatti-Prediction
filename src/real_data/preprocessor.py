"""
Preprocessor to convert real satellite data to model format
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


class RealDataPreprocessor:
    """
    Preprocesses real LULC data for model training
    """
    
    # Class mapping from Impact Observatory LULC to our 7 classes
    # IO Classes: 1=Water, 2=Trees, 4=Flooded veg, 5=Crops, 7=Built, 8=Bare, 9=Snow, 10=Clouds, 11=Rangeland
    # Our classes: 0=Urban, 1=Forest, 2=Agriculture, 3=Water, 4=Barren, 5=Wetland, 6=Grassland
    CLASS_MAPPING_IO = {
        1: 3,   # Water → Water
        2: 1,   # Trees → Forest
        4: 5,   # Flooded vegetation → Wetland
        5: 2,   # Crops → Agriculture
        7: 0,   # Built Area → Urban
        8: 4,   # Bare ground → Barren
        9: 4,   # Snow/Ice → Barren
        10: -1, # Clouds → NoData (will be filtered)
        11: 6,  # Rangeland → Grassland
    }
    
    # ESA WorldCover class mapping
    # ESA: 10=Tree, 20=Shrubland, 30=Grassland, 40=Cropland, 50=Built, 60=Bare, 70=Snow, 80=Water, 90=Herbaceous wetland, 95=Mangroves, 100=Moss
    CLASS_MAPPING_ESA = {
        10: 1,   # Tree cover → Forest
        20: 6,   # Shrubland → Grassland
        30: 6,   # Grassland → Grassland
        40: 2,   # Cropland → Agriculture
        50: 0,   # Built-up → Urban
        60: 4,   # Bare/sparse vegetation → Barren
        70: 4,   # Snow and ice → Barren
        80: 3,   # Permanent water bodies → Water
        90: 5,   # Herbaceous wetland → Wetland
        95: 5,   # Mangroves → Wetland
        100: 6,  # Moss and lichen → Grassland
    }
    
    def __init__(self, n_classes: int = 7):
        """
        Initialize preprocessor
        
        Args:
            n_classes: Number of output classes (default: 7)
        """
        self.n_classes = n_classes
        print(f"🔧 Preprocessor initialized for {n_classes} classes")
    
    def remap_classes(self, lulc_array: np.ndarray, 
                     source: str = "io") -> np.ndarray:
        """
        Remap LULC classes to our standard 7 classes
        
        Args:
            lulc_array: Original LULC array
            source: Data source ("io" for Impact Observatory, "esa" for WorldCover)
            
        Returns:
            Remapped array
        """
        print(f"  🗺️  Remapping classes from {source.upper()} format...")
        
        # Select mapping
        if source.lower() == "io":
            mapping = self.CLASS_MAPPING_IO
        elif source.lower() == "esa":
            mapping = self.CLASS_MAPPING_ESA
        else:
            print(f"  ⚠️  Unknown source '{source}', using IO mapping")
            mapping = self.CLASS_MAPPING_IO
        
        # Create output array
        remapped = np.full_like(lulc_array, -1, dtype=np.int16)
        
        # Apply mapping
        unique_vals = np.unique(lulc_array)
        print(f"  Original classes: {unique_vals[:20]}")
        
        for orig_class, new_class in mapping.items():
            mask = lulc_array == orig_class
            if mask.any():
                remapped[mask] = new_class
        
        # Handle unmapped values
        unmapped_mask = remapped == -1
        if unmapped_mask.any():
            unmapped_vals = np.unique(lulc_array[unmapped_mask])
            print(f"  ⚠️  Unmapped classes: {unmapped_vals} → assigning to Barren (4)")
            remapped[unmapped_mask] = 4  # Default to Barren
        
        print(f"  Remapped classes: {np.unique(remapped)}")
        print(f"  ✅ Remapping complete")
        
        return remapped
    
    def crop_to_patches(self, large_array: np.ndarray,
                       patch_size: int = 256,
                       overlap: int = 0) -> List[np.ndarray]:
        """
        Crop large array into smaller patches
        
        Args:
            large_array: Large LULC array
            patch_size: Size of patches (default: 256x256)
            overlap: Overlap between patches (default: 0)
            
        Returns:
            List of patches
        """
        print(f"  ✂️  Cropping to {patch_size}x{patch_size} patches...")
        
        height, width = large_array.shape
        stride = patch_size - overlap
        
        patches = []
        for i in range(0, height - patch_size + 1, stride):
            for j in range(0, width - patch_size + 1, stride):
                patch = large_array[i:i+patch_size, j:j+patch_size]
                patches.append(patch)
        
        print(f"  ✅ Created {len(patches)} patches from {height}x{width} image")
        return patches
    
    def create_temporal_samples(self, years_data: Dict[int, np.ndarray],
                               patch_size: int = 256,
                               overlap: int = 0,
                               source: str = "io") -> Tuple[np.ndarray, np.ndarray]:
        """
        Create temporal training samples
        
        Args:
            years_data: Dictionary of year -> LULC array
            patch_size: Patch size
            overlap: Overlap between patches
            source: Data source for class mapping
            
        Returns:
            Tuple of (input_sequences, target_sequences)
        """
        print(f"\n📦 Creating temporal samples...")
        
        years = sorted(years_data.keys())
        print(f"  Years: {years}")
        
        if len(years) < 2:
            print("  ⚠️  Need at least 2 years for temporal sequences")
            return np.array([]), np.array([])
        
        # Remap all years
        remapped_data = {}
        for year in years:
            print(f"\n  Processing {year}:")
            remapped_data[year] = self.remap_classes(years_data[year], source=source)
        
        # Ensure all years have same shape
        shapes = [arr.shape for arr in remapped_data.values()]
        if len(set(shapes)) > 1:
            print(f"  ⚠️  Different shapes: {shapes}")
            # Find minimum shape
            min_h = min(s[0] for s in shapes)
            min_w = min(s[1] for s in shapes)
            print(f"  Cropping all to {min_h}x{min_w}")
            for year in years:
                remapped_data[year] = remapped_data[year][:min_h, :min_w]
        
        # Create patches for each year
        patches_by_year = {}
        for year in years:
            patches = self.crop_to_patches(remapped_data[year], patch_size, overlap)
            patches_by_year[year] = patches
            print(f"  {year}: {len(patches)} patches")
        
        # Create temporal sequences: [t, t+1, ...] → [t+n]
        # For 3 years: [2020, 2021] → [2022]
        n_patches = len(patches_by_year[years[0]])
        input_sequences = []
        target_sequences = []
        
        # Use first n-1 years as input, last year as target
        for patch_idx in range(n_patches):
            # Input: all years except last
            input_seq = [patches_by_year[year][patch_idx] for year in years[:-1]]
            # Target: last year
            target_seq = patches_by_year[years[-1]][patch_idx]
            
            input_sequences.append(np.stack(input_seq, axis=0))  # Shape: (n_years-1, H, W)
            target_sequences.append(target_seq)  # Shape: (H, W)
        
        input_sequences = np.array(input_sequences)
        target_sequences = np.array(target_sequences)
        
        print(f"\n  ✅ Created {len(input_sequences)} temporal samples")
        print(f"  Input shape: {input_sequences.shape}")
        print(f"  Target shape: {target_sequences.shape}")
        
        return input_sequences, target_sequences
    
    def split_train_val_test(self, inputs: np.ndarray, targets: np.ndarray,
                            ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                            random_seed: int = 42) -> Dict:
        """
        Split data into train/val/test sets
        
        Args:
            inputs: Input sequences
            targets: Target sequences
            ratios: Train/val/test split ratios
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with train/val/test splits
        """
        print(f"\n📊 Splitting data with ratios {ratios}...")
        
        np.random.seed(random_seed)
        n_samples = len(inputs)
        indices = np.random.permutation(n_samples)
        
        train_ratio, val_ratio, test_ratio = ratios
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        splits = {
            'train': {
                'inputs': inputs[train_idx],
                'targets': targets[train_idx]
            },
            'val': {
                'inputs': inputs[val_idx],
                'targets': targets[val_idx]
            },
            'test': {
                'inputs': inputs[test_idx],
                'targets': targets[test_idx]
            }
        }
        
        print(f"  Train: {len(train_idx)} samples")
        print(f"  Val: {len(val_idx)} samples")
        print(f"  Test: {len(test_idx)} samples")
        print(f"  ✅ Split complete")
        
        return splits
    
    def save_as_npy(self, splits: Dict, output_dir: str):
        """
        Save processed data as .npy files
        
        Args:
            splits: Dictionary with train/val/test splits
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving to {output_dir}...")
        
        for split_name, data in splits.items():
            # Save inputs and targets
            input_file = output_path / f"{split_name}_inputs.npy"
            target_file = output_path / f"{split_name}_targets.npy"
            
            np.save(input_file, data['inputs'])
            np.save(target_file, data['targets'])
            
            print(f"  ✅ {split_name}: {input_file.name}, {target_file.name}")
        
        # Save metadata
        metadata = {
            'n_classes': self.n_classes,
            'patch_size': splits['train']['inputs'].shape[-1],
            'n_timesteps': splits['train']['inputs'].shape[1],
            'train_size': len(splits['train']['inputs']),
            'val_size': len(splits['val']['inputs']),
            'test_size': len(splits['test']['inputs'])
        }
        
        metadata_file = output_path / "metadata.txt"
        with open(metadata_file, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        
        print(f"  ✅ Metadata: {metadata_file.name}")
        print(f"\n✅ All data saved successfully!")


def preprocess_for_model(years_data: Dict[int, np.ndarray],
                        output_dir: str,
                        patch_size: int = 256,
                        source: str = "io") -> Dict:
    """
    Convenience function to preprocess real data for model training
    
    Args:
        years_data: Dictionary of year -> LULC array
        output_dir: Output directory for processed data
        patch_size: Size of patches
        source: Data source ("io" or "esa")
        
    Returns:
        Dictionary with splits
    """
    print("🔄 Preprocessing real satellite data for model...")
    
    preprocessor = RealDataPreprocessor()
    
    # Create temporal samples
    inputs, targets = preprocessor.create_temporal_samples(
        years_data, patch_size=patch_size, source=source
    )
    
    if len(inputs) == 0:
        print("❌ No samples created!")
        return {}
    
    # Split data
    splits = preprocessor.split_train_val_test(inputs, targets)
    
    # Save data
    preprocessor.save_as_npy(splits, output_dir)
    
    return splits


if __name__ == "__main__":
    # Test preprocessor
    print("🧪 Testing RealDataPreprocessor...")
    
    # Create dummy data
    dummy_data = {
        2020: np.random.randint(1, 12, (1000, 1000)),
        2021: np.random.randint(1, 12, (1000, 1000)),
        2022: np.random.randint(1, 12, (1000, 1000))
    }
    
    splits = preprocess_for_model(
        dummy_data,
        output_dir="./data/test_output",
        patch_size=256
    )
    
    print(f"\n✅ Test complete! Created {len(splits)} splits")
