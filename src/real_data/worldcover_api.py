"""
ESA WorldCover API fetcher - fallback data source
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Optional
import warnings

import numpy as np
import rasterio
from tqdm import tqdm

try:
    import pystac_client
    STAC_AVAILABLE = True
except ImportError:
    STAC_AVAILABLE = False
    warnings.warn("pystac-client not installed. Install with: pip install pystac-client")


class WorldCoverFetcher:
    """
    Fetches ESA WorldCover LULC data as fallback source
    """
    
    # ESA WorldCover STAC endpoint
    STAC_ENDPOINT = "https://services.terrascope.be/stac/"
    COLLECTION = "esa-worldcover"
    
    def __init__(self, region_name: str = "Kovilpatti",
                 lat: float = 9.17, lon: float = 77.87,
                 radius_km: float = 10,
                 cache_dir: str = "./data/cache"):
        """
        Initialize WorldCover fetcher
        
        Args:
            region_name: Name of the region
            lat: Latitude of region center
            lon: Longitude of region center
            radius_km: Radius in kilometers
            cache_dir: Directory for caching downloaded data
        """
        self.region_name = region_name
        self.lat = lat
        self.lon = lon
        self.radius_km = radius_km
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate bounding box
        lat_offset = self.radius_km / 111.0
        lon_offset = self.radius_km / (111.0 * np.cos(np.radians(self.lat)))
        
        self.bbox = [
            self.lon - lon_offset,
            self.lat - lat_offset,
            self.lon + lon_offset,
            self.lat + lat_offset
        ]
        
        print(f"🌍 WorldCover Fetcher initialized for {self.region_name}")
        print(f"📦 BBox: {[f'{x:.2f}' for x in self.bbox]}")
    
    def fetch_lulc_data(self, year: int, max_retries: int = 3) -> Optional[np.ndarray]:
        """
        Fetch WorldCover data for a year
        
        Args:
            year: Year to fetch (2020, 2021 available)
            max_retries: Maximum retry attempts
            
        Returns:
            LULC array or None
        """
        if not STAC_AVAILABLE:
            print("❌ pystac-client not available")
            return None
        
        if year not in [2020, 2021]:
            print(f"⚠️  WorldCover only has data for 2020 and 2021, not {year}")
            return None
        
        # Check cache
        cache_file = self.cache_dir / f"{self.region_name}_{year}_worldcover.npy"
        if cache_file.exists():
            print(f"📂 Loading from cache: {cache_file}")
            return np.load(cache_file)
        
        print(f"\n🛰️ Fetching WorldCover data for {year}...")
        
        for attempt in range(max_retries):
            try:
                # Search catalog
                items = self._search_stac_catalog(year)
                if not items:
                    print(f"⚠️  No WorldCover data found for {year}")
                    return None
                
                # Download
                lulc_array = self._download_raster(items[0])
                
                if lulc_array is not None and lulc_array.size > 0:
                    np.save(cache_file, lulc_array)
                    print(f"✅ WorldCover data cached for {year}")
                    return lulc_array
                    
            except Exception as e:
                print(f"⚠️  Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                continue
        
        print(f"❌ Failed to fetch WorldCover for {year}")
        return None
    
    def fetch_temporal_sequence(self, years: List[int]) -> Dict[int, np.ndarray]:
        """
        Fetch data for multiple years
        
        Args:
            years: List of years
            
        Returns:
            Dictionary of year -> array
        """
        results = {}
        for year in tqdm(years, desc="Fetching WorldCover"):
            data = self.fetch_lulc_data(year)
            if data is not None:
                results[year] = data
        return results
    
    def _search_stac_catalog(self, year: int) -> List:
        """Search ESA WorldCover STAC catalog"""
        try:
            catalog = pystac_client.Client.open(self.STAC_ENDPOINT)
            
            search = catalog.search(
                collections=[self.COLLECTION],
                bbox=self.bbox,
                datetime=f"{year}-01-01/{year}-12-31"
            )
            
            items = list(search.items())
            print(f"  Found {len(items)} WorldCover items for {year}")
            return items
            
        except Exception as e:
            print(f"  Error searching WorldCover: {e}")
            return []
    
    def _download_raster(self, item) -> Optional[np.ndarray]:
        """Download raster from STAC item"""
        try:
            # WorldCover typically has "map" asset
            asset_key = "map"
            if asset_key not in item.assets:
                for key in ["classification", "data", "lulc"]:
                    if key in item.assets:
                        asset_key = key
                        break
            
            if asset_key not in item.assets:
                print(f"  ⚠️  No asset found. Available: {list(item.assets.keys())}")
                return None
            
            href = item.assets[asset_key].href
            print(f"  📥 Downloading WorldCover...")
            
            with rasterio.open(href) as src:
                lulc = src.read(1)
                print(f"  📊 Shape: {lulc.shape}, range: {lulc.min()}-{lulc.max()}")
                return lulc
                
        except Exception as e:
            print(f"  ❌ Error downloading: {e}")
            return None


if __name__ == "__main__":
    # Test WorldCover fetcher
    print("🧪 Testing WorldCoverFetcher...")
    fetcher = WorldCoverFetcher()
    data = fetcher.fetch_temporal_sequence([2020, 2021])
    
    if data:
        for year, array in data.items():
            print(f"{year}: {array.shape}")
