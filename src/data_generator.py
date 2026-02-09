"""
Data Generator for Kovilpatti LULC
Generates realistic multi-temporal Land Use/Land Cover data for Kovilpatti region.
"""

import numpy as np
from typing import Tuple, List
import cv2


class KovilpattiLULCGenerator:
    """
    Generator for realistic LULC maps specific to Kovilpatti region characteristics.
    
    Kovilpatti is a semi-arid region in Tamil Nadu with:
    - Industrial development (fireworks, match industries)
    - Agricultural activities (cotton, groundnut)
    - Water scarcity challenges
    - Urban expansion patterns
    """
    
    def __init__(self, img_size: int = 256, num_classes: int = 7):
        """
        Initialize the LULC generator.
        
        Args:
            img_size: Size of the generated LULC maps (img_size x img_size)
            num_classes: Number of LULC classes (default: 7)
        """
        self.img_size = img_size
        self.num_classes = num_classes
        
        # LULC class definitions
        self.class_names = [
            "Urban",      # 0
            "Forest",     # 1
            "Agriculture",# 2
            "Water",      # 3
            "Barren",     # 4
            "Wetland",    # 5
            "Grassland"   # 6
        ]
        
        # RGB colors for visualization
        self.class_colors = np.array([
            [255, 0, 0],      # Urban - Red
            [0, 128, 0],      # Forest - Dark Green
            [144, 238, 144],  # Agriculture - Light Green
            [0, 0, 255],      # Water - Blue
            [165, 42, 42],    # Barren - Brown
            [0, 255, 255],    # Wetland - Cyan
            [173, 255, 47]    # Grassland - Yellow-Green
        ], dtype=np.uint8)
        
        # Initialize transition matrix
        self.transition_matrix = self._create_kovilpatti_transition_matrix()
    
    def _create_kovilpatti_transition_matrix(self) -> np.ndarray:
        """
        Create transition probability matrix based on Kovilpatti patterns.
        
        Reflects realistic land cover changes:
        - Urban expansion (agriculture/barren -> urban)
        - Agricultural degradation (agriculture -> barren)
        - Water scarcity (water/wetland -> barren)
        - Limited forest regeneration
        
        Returns:
            7x7 transition probability matrix
        """
        # Initialize with high probability of staying the same
        T = np.eye(7) * 0.7
        
        # Urban (0): Expands, rarely converts back
        T[0, :] = [0.85, 0.0, 0.05, 0.0, 0.05, 0.0, 0.05]
        
        # Forest (1): Can degrade to agriculture or grassland
        T[1, :] = [0.02, 0.70, 0.10, 0.0, 0.05, 0.03, 0.10]
        
        # Agriculture (2): Can convert to urban, barren, or stay
        T[2, :] = [0.15, 0.02, 0.60, 0.02, 0.15, 0.03, 0.03]
        
        # Water (3): Stable but can dry to barren (water scarcity)
        T[3, :] = [0.0, 0.0, 0.05, 0.75, 0.15, 0.05, 0.0]
        
        # Barren (4): Can convert to urban, agriculture, or grassland
        T[4, :] = [0.10, 0.02, 0.10, 0.02, 0.65, 0.03, 0.08]
        
        # Wetland (5): Can dry to barren or convert to water
        T[5, :] = [0.02, 0.03, 0.05, 0.15, 0.20, 0.50, 0.05]
        
        # Grassland (6): Can convert to agriculture, barren, or urban
        T[6, :] = [0.08, 0.05, 0.15, 0.02, 0.10, 0.05, 0.55]
        
        # Normalize rows
        T = T / T.sum(axis=1, keepdims=True)
        
        return T
    
    def generate_initial_map(self) -> np.ndarray:
        """
        Generate initial LULC map with Kovilpatti characteristics.
        
        Creates a realistic base map with:
        - Urban center
        - Scattered water bodies
        - Agricultural zones
        - Barren semi-arid areas
        - Small forest patches
        
        Returns:
            Initial LULC map of shape (img_size, img_size)
        """
        lulc_map = np.zeros((self.img_size, self.img_size), dtype=np.int32)
        
        # Start with barren land (semi-arid region)
        lulc_map[:, :] = 4  # Barren
        
        # Create urban center (Kovilpatti town)
        center_x, center_y = self.img_size // 2, self.img_size // 2
        urban_radius = self.img_size // 10
        y, x = np.ogrid[:self.img_size, :self.img_size]
        urban_mask = ((x - center_x)**2 + (y - center_y)**2) <= urban_radius**2
        lulc_map[urban_mask] = 0  # Urban
        
        # Add agricultural zones (around urban, in patches)
        for _ in range(15):
            ag_x = np.random.randint(0, self.img_size)
            ag_y = np.random.randint(0, self.img_size)
            ag_size = np.random.randint(15, 35)
            y, x = np.ogrid[:self.img_size, :self.img_size]
            ag_mask = ((x - ag_x)**2 + (y - ag_y)**2) <= ag_size**2
            # Don't override urban
            ag_mask = ag_mask & (lulc_map != 0)
            lulc_map[ag_mask] = 2  # Agriculture
        
        # Add water bodies (sparse, small)
        for _ in range(3):
            water_x = np.random.randint(0, self.img_size)
            water_y = np.random.randint(0, self.img_size)
            water_size = np.random.randint(8, 15)
            y, x = np.ogrid[:self.img_size, :self.img_size]
            water_mask = ((x - water_x)**2 + (y - water_y)**2) <= water_size**2
            lulc_map[water_mask] = 3  # Water
        
        # Add small forest patches (limited)
        for _ in range(5):
            forest_x = np.random.randint(0, self.img_size)
            forest_y = np.random.randint(0, self.img_size)
            forest_size = np.random.randint(10, 20)
            y, x = np.ogrid[:self.img_size, :self.img_size]
            forest_mask = ((x - forest_x)**2 + (y - forest_y)**2) <= forest_size**2
            forest_mask = forest_mask & (lulc_map == 4)  # Only on barren
            lulc_map[forest_mask] = 1  # Forest
        
        # Add grassland patches
        for _ in range(10):
            grass_x = np.random.randint(0, self.img_size)
            grass_y = np.random.randint(0, self.img_size)
            grass_size = np.random.randint(12, 25)
            y, x = np.ogrid[:self.img_size, :self.img_size]
            grass_mask = ((x - grass_x)**2 + (y - grass_y)**2) <= grass_size**2
            grass_mask = grass_mask & (lulc_map == 4)  # Only on barren
            lulc_map[grass_mask] = 6  # Grassland
        
        # Add few wetlands
        for _ in range(2):
            wetland_x = np.random.randint(0, self.img_size)
            wetland_y = np.random.randint(0, self.img_size)
            wetland_size = np.random.randint(6, 12)
            y, x = np.ogrid[:self.img_size, :self.img_size]
            wetland_mask = ((x - wetland_x)**2 + (y - wetland_y)**2) <= wetland_size**2
            wetland_mask = wetland_mask & (lulc_map == 4)  # Only on barren
            lulc_map[wetland_mask] = 5  # Wetland
        
        return lulc_map
    
    def apply_transition(self, lulc_map: np.ndarray, intensity: float = 1.0) -> np.ndarray:
        """
        Apply temporal transition to LULC map.
        
        Args:
            lulc_map: Current LULC map
            intensity: Transition intensity (0.0 to 1.0), controls speed of change
            
        Returns:
            New LULC map after transition
        """
        new_map = lulc_map.copy()
        
        # Urban expansion from center
        center_x, center_y = self.img_size // 2, self.img_size // 2
        y, x = np.ogrid[:self.img_size, :self.img_size]
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Find urban boundary and expand it
        urban_mask = (lulc_map == 0)
        if urban_mask.any():
            max_urban_dist = distances[urban_mask].max()
            expansion_radius = max_urban_dist + (3 * intensity)
            
            # Expand urban into nearby agriculture/barren
            expansion_mask = (distances <= expansion_radius) & (distances > max_urban_dist)
            expansion_mask = expansion_mask & ((lulc_map == 2) | (lulc_map == 4))
            
            # Apply probabilistic expansion
            expansion_prob = np.random.random(expansion_mask.shape)
            expansion_mask = expansion_mask & (expansion_prob < 0.3 * intensity)
            new_map[expansion_mask] = 0  # Urban
        
        # Apply random transitions based on transition matrix
        for i in range(self.img_size):
            for j in range(self.img_size):
                current_class = lulc_map[i, j]
                
                # Skip if already changed by urban expansion
                if new_map[i, j] != current_class:
                    continue
                
                # Apply transition with probability based on intensity
                if np.random.random() < 0.1 * intensity:
                    probs = self.transition_matrix[current_class]
                    new_class = np.random.choice(self.num_classes, p=probs)
                    new_map[i, j] = new_class
        
        # Apply spatial smoothing to avoid isolated pixels
        new_map = self._spatial_smoothing(new_map)
        
        return new_map
    
    def _spatial_smoothing(self, lulc_map: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Apply spatial smoothing using mode filter to reduce noise.
        
        Args:
            lulc_map: LULC map to smooth
            kernel_size: Size of smoothing kernel
            
        Returns:
            Smoothed LULC map
        """
        from scipy import stats
        
        smoothed = lulc_map.copy()
        pad = kernel_size // 2
        
        for i in range(pad, self.img_size - pad):
            for j in range(pad, self.img_size - pad):
                window = lulc_map[i-pad:i+pad+1, j-pad:j+pad+1]
                mode_val = stats.mode(window, axis=None, keepdims=False)[0]
                smoothed[i, j] = mode_val
        
        return smoothed
    
    def generate_temporal_sequence(self, num_timesteps: int = 5) -> List[np.ndarray]:
        """
        Generate temporal sequence of LULC maps.
        
        Args:
            num_timesteps: Number of time steps to generate (default: 5 for 2018-2022)
            
        Returns:
            List of LULC maps, one for each timestep
        """
        sequence = []
        
        # Generate initial map
        current_map = self.generate_initial_map()
        sequence.append(current_map)
        
        # Generate subsequent maps with transitions
        for t in range(1, num_timesteps):
            # Intensity increases slightly over time (accelerating change)
            intensity = 0.8 + (0.2 * t / num_timesteps)
            current_map = self.apply_transition(current_map, intensity=intensity)
            sequence.append(current_map)
        
        return sequence
    
    def lulc_to_rgb(self, lulc_map: np.ndarray) -> np.ndarray:
        """
        Convert LULC class map to RGB image for visualization.
        
        Args:
            lulc_map: LULC map with class indices
            
        Returns:
            RGB image of shape (img_size, img_size, 3)
        """
        rgb_map = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        for class_id in range(self.num_classes):
            mask = (lulc_map == class_id)
            rgb_map[mask] = self.class_colors[class_id]
        
        return rgb_map
