"""
API-based fetcher for real satellite LULC data
Uses Microsoft Planetary Computer STAC API
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings

import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

try:
    import pystac_client
    import planetary_computer as pc
    STAC_AVAILABLE = True
except ImportError:
    STAC_AVAILABLE = False
    warnings.warn("pystac-client or planetary-computer not installed. Install with: pip install pystac-client planetary-computer")


class KovilpattiDataFetcher:
    """
    Fetches real LULC data for Kovilpatti region from Microsoft Planetary Computer
    """
    
    # Kovilpatti region coordinates
    DEFAULT_LAT = 9.17
    DEFAULT_LON = 77.87
    DEFAULT_RADIUS_KM = 10
    
    # Planetary Computer STAC endpoint
    STAC_ENDPOINT = "https://planetarycomputer.microsoft.com/api/stac/v1"
    COLLECTION = "io-lulc-annual-v02"
    
    def __init__(self, region_name: str = "Kovilpatti", 
                 lat: float = None, lon: float = None, 
                 radius_km: float = None,
                 cache_dir: str = "./data/cache"):
        """
        Initialize data fetcher
        
        Args:
            region_name: Name of the region
            lat: Latitude of region center
            lon: Longitude of region center
            radius_km: Radius in kilometers
            cache_dir: Directory for caching downloaded data
        """
        self.region_name = region_name
        self.lat = lat or self.DEFAULT_LAT
        self.lon = lon or self.DEFAULT_LON
        self.radius_km = radius_km or self.DEFAULT_RADIUS_KM
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate bounding box (approximate)
        # 1 degree latitude ≈ 111 km
        # 1 degree longitude ≈ 111 km * cos(lat)
        lat_offset = self.radius_km / 111.0
        lon_offset = self.radius_km / (111.0 * np.cos(np.radians(self.lat)))
        
        self.bbox = [
            self.lon - lon_offset,  # min_lon
            self.lat - lat_offset,   # min_lat
            self.lon + lon_offset,   # max_lon
            self.lat + lat_offset    # max_lat
        ]
        
        print(f"📍 Region: {self.region_name}")
        print(f"🌍 Center: ({self.lat:.2f}°N, {self.lon:.2f}°E)")
        print(f"📦 Bounding Box: {[f'{x:.2f}' for x in self.bbox]}")
        print(f"💾 Cache: {self.cache_dir}")
    
    def fetch_lulc_data(self, year: int, max_retries: int = 3) -> Optional[np.ndarray]:
        """
        Fetch LULC data for a single year
        
        Args:
            year: Year to fetch (2017-2022 available)
            max_retries: Maximum number of retry attempts
            
        Returns:
            LULC array or None if failed
        """
        if not STAC_AVAILABLE:
            print("❌ STAC libraries not available. Please install: pip install pystac-client planetary-computer")
            return None
        
        # Check cache first
        cache_file = self.cache_dir / f"{self.region_name}_{year}_lulc.npy"
        if cache_file.exists():
            print(f"📂 Loading from cache: {cache_file}")
            return np.load(cache_file)
        
        print(f"\n🛰️ Fetching LULC data for {year}...")
        
        for attempt in range(max_retries):
            try:
                # Search STAC catalog
                items = self._search_stac_catalog(year)
                if not items:
                    print(f"⚠️  No data found for {year}")
                    return None
                
                # Download raster data
                lulc_array = self._download_raster(items[0], year)
                
                if lulc_array is not None:
                    # Validate data
                    if self._validate_data(lulc_array):
                        # Save to cache
                        np.save(cache_file, lulc_array)
                        print(f"✅ Data fetched and cached for {year}")
                        return lulc_array
                    else:
                        print(f"⚠️  Data validation failed for {year}")
                        return None
                        
            except Exception as e:
                print(f"⚠️  Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
        
        print(f"❌ Failed to fetch data for {year} after {max_retries} attempts")
        return None
    
    def fetch_temporal_sequence(self, years: List[int]) -> Dict[int, np.ndarray]:
        """
        Fetch LULC data for multiple years
        
        Args:
            years: List of years to fetch
            
        Returns:
            Dictionary mapping year to LULC array
        """
        results = {}
        
        print(f"\n📅 Fetching temporal sequence: {years}")
        for year in tqdm(years, desc="Fetching years"):
            data = self.fetch_lulc_data(year)
            if data is not None:
                results[year] = data
        
        print(f"\n✅ Successfully fetched {len(results)}/{len(years)} years")
        return results
    
    def _search_stac_catalog(self, year: int) -> List:
        """
        Search Planetary Computer STAC catalog
        
        Args:
            year: Year to search for
            
        Returns:
            List of STAC items
        """
        try:
            catalog = pystac_client.Client.open(
                self.STAC_ENDPOINT,
                modifier=pc.sign_inplace  # Sign URLs for access
            )
            
            # Search for items
            search = catalog.search(
                collections=[self.COLLECTION],
                bbox=self.bbox,
                datetime=f"{year}-01-01/{year}-12-31"
            )
            
            items = list(search.items())
            print(f"  Found {len(items)} items for {year}")
            return items
            
        except Exception as e:
            print(f"  Error searching catalog: {e}")
            return []
    
    def _download_raster(self, item, year: int) -> Optional[np.ndarray]:
        """
        Download raster data from STAC item
        
        Args:
            item: STAC item
            year: Year of data
            
        Returns:
            LULC array or None
        """
        try:
            # Get the data asset
            asset_key = "data"  # Impact Observatory uses "data" asset
            if asset_key not in item.assets:
                # Try alternative keys
                for key in ["classification", "lulc", "map"]:
                    if key in item.assets:
                        asset_key = key
                        break
            
            if asset_key not in item.assets:
                print(f"  ⚠️  No suitable asset found. Available: {list(item.assets.keys())}")
                return None
            
            asset = item.assets[asset_key]
            href = asset.href
            
            print(f"  📥 Downloading from: {href[:100]}...")
            
            # Read raster data
            with rasterio.open(href) as src:
                # Read the data
                # Use window to crop to exact bbox if needed
                lulc = src.read(1)  # Read first band
                
                print(f"  📊 Shape: {lulc.shape}, dtype: {lulc.dtype}")
                print(f"  📈 Value range: {lulc.min()} - {lulc.max()}")
                
                return lulc
                
        except Exception as e:
            print(f"  ❌ Error downloading raster: {e}")
            return None
    
    def _validate_data(self, data: np.ndarray) -> bool:
        """
        Validate downloaded data
        
        Args:
            data: LULC array
            
        Returns:
            True if valid, False otherwise
        """
        if data is None or data.size == 0:
            return False
        
        # Check shape
        if len(data.shape) != 2:
            print(f"  ⚠️  Invalid shape: {data.shape}")
            return False
        
        # Check dimensions
        if data.shape[0] < 100 or data.shape[1] < 100:
            print(f"  ⚠️  Image too small: {data.shape}")
            return False
        
        # Check value range (LULC classes should be 0-11)
        if data.max() > 20 or data.min() < 0:
            print(f"  ⚠️  Unusual value range: {data.min()} - {data.max()}")
            # Don't reject, might be different encoding
        
        return True


def fetch_kovilpatti_data(years: List[int] = None, 
                         cache_dir: str = "./data/cache") -> Dict[int, np.ndarray]:
    """
    Convenience function to fetch Kovilpatti LULC data
    
    Args:
        years: List of years to fetch (default: [2020, 2021, 2022])
        cache_dir: Cache directory
        
    Returns:
        Dictionary mapping year to LULC array
    """
    if years is None:
        years = [2020, 2021, 2022]
    
    fetcher = KovilpattiDataFetcher(cache_dir=cache_dir)
    return fetcher.fetch_temporal_sequence(years)


if __name__ == "__main__":
    # Test the fetcher
    print("🧪 Testing KovilpattiDataFetcher...")
    data = fetch_kovilpatti_data(years=[2020])
    
    if data:
        for year, array in data.items():
            print(f"\n{year}: shape={array.shape}, dtype={array.dtype}")
            print(f"  Unique values: {np.unique(array)[:10]}")
