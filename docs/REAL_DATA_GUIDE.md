# Real Satellite Data Guide

## Overview

This guide explains how to fetch and use real satellite data for Kovilpatti LULC (Land Use Land Cover) prediction. The system automatically downloads data via API—**no manual downloads required**!

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `pystac-client`: STAC API client
- `planetary-computer`: Microsoft Planetary Computer access
- `rasterio`: Geospatial raster data I/O
- `stackstac`: STAC to xarray converter
- `xarray`: Multi-dimensional arrays

### 2. Fetch Real Data

```bash
# Fetch Kovilpatti data for 2020-2022
python scripts/fetch_real_data.py --region kovilpatti --years 2020 2021 2022

# Fetch with custom location
python scripts/fetch_real_data.py --lat 9.17 --lon 77.87 --radius 10

# Use different data source
python scripts/fetch_real_data.py --source worldcover

# Try both sources (with fallback)
python scripts/fetch_real_data.py --source both
```

### 3. Verify Downloaded Data

```bash
python scripts/verify_real_data.py --data_dir data/Kovilpatti_LULC_Real/
```

### 4. Train on Real Data

```bash
python scripts/run_training_real.py --data_dir data/Kovilpatti_LULC_Real/ --real_data
```

## Data Sources

### Microsoft Planetary Computer (Primary)

- **Collection**: Impact Observatory LULC (io-lulc-annual-v02)
- **Coverage**: Global, 10m resolution
- **Years**: 2017-2022
- **Authentication**: Not required
- **Endpoint**: https://planetarycomputer.microsoft.com/api/stac/v1

**Features**:
- High-quality land cover classification
- Annual global coverage
- 10-meter resolution
- Free and open access

### ESA WorldCover (Fallback)

- **Collection**: ESA WorldCover
- **Coverage**: Global, 10m resolution
- **Years**: 2020, 2021
- **Authentication**: Not required
- **Endpoint**: https://services.terrascope.be/stac/

**Features**:
- European Space Agency product
- Validated accuracy
- Complementary to Planetary Computer
- Automatic fallback option

## Kovilpatti Region Details

- **Location**: Kovilpatti, Tamil Nadu, India
- **Coordinates**: 9.17°N, 77.87°E
- **Bounding Box**: [77.77, 9.07, 77.97, 9.27] (WGS84)
- **Area**: Approximately 20km × 20km
- **Coverage**: Urban, agricultural, and rural areas

## Data Processing Pipeline

### Step 1: API Fetch

The fetcher automatically:
1. Connects to STAC API
2. Searches for data matching region and year
3. Downloads raster data
4. Caches locally to avoid re-downloading
5. Validates data quality

### Step 2: Class Remapping

Real LULC classes are mapped to our 7 standard classes:

**Impact Observatory → Our Classes**:
- 1 (Water) → 3 (Water)
- 2 (Trees) → 1 (Forest)
- 4 (Flooded vegetation) → 5 (Wetland)
- 5 (Crops) → 2 (Agriculture)
- 7 (Built Area) → 0 (Urban)
- 8 (Bare ground) → 4 (Barren)
- 9 (Snow/Ice) → 4 (Barren)
- 10 (Clouds) → Filtered out
- 11 (Rangeland) → 6 (Grassland)

**ESA WorldCover → Our Classes**:
- 10 (Tree cover) → 1 (Forest)
- 20 (Shrubland) → 6 (Grassland)
- 30 (Grassland) → 6 (Grassland)
- 40 (Cropland) → 2 (Agriculture)
- 50 (Built-up) → 0 (Urban)
- 60 (Bare/sparse) → 4 (Barren)
- 70 (Snow/ice) → 4 (Barren)
- 80 (Water) → 3 (Water)
- 90 (Herbaceous wetland) → 5 (Wetland)
- 95 (Mangroves) → 5 (Wetland)
- 100 (Moss/lichen) → 6 (Grassland)

### Step 3: Patch Creation

Large rasters are divided into 256×256 patches for training:
- Configurable patch size
- Optional overlap between patches
- Quality filtering
- Spatial consistency preserved

### Step 4: Temporal Sequence Creation

Multi-year data creates temporal sequences:
- **Input**: Years 1 to N-1 (e.g., 2020, 2021)
- **Target**: Year N (e.g., 2022)
- Temporal alignment ensured
- Missing data handled

### Step 5: Train/Val/Test Split

Data split with configurable ratios:
- **Train**: 70% (default)
- **Validation**: 15% (default)
- **Test**: 15% (default)

## API Details

### STAC (SpatioTemporal Asset Catalog)

STAC is a specification for describing geospatial data. Our fetcher uses:
- **pystac-client**: Python client for STAC APIs
- **Search parameters**: Collection, bbox, datetime
- **Assets**: Raster data URLs

### Planetary Computer Integration

```python
import pystac_client
import planetary_computer as pc

# Open catalog
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=pc.sign_inplace  # Sign URLs for access
)

# Search for data
search = catalog.search(
    collections=["io-lulc-annual-v02"],
    bbox=[77.77, 9.07, 77.97, 9.27],
    datetime="2020-01-01/2020-12-31"
)

items = list(search.items())
```

### Rate Limiting & Retry

The fetcher handles:
- **Exponential backoff**: 2^attempt seconds
- **Max retries**: 3 attempts (configurable)
- **Timeout**: 300 seconds (configurable)
- **Error recovery**: Automatic fallback to alternative source

### Caching Strategy

Downloaded data is cached locally:
- **Location**: `./data/cache/` (configurable)
- **Format**: NumPy .npy files
- **Naming**: `{region}_{year}_{source}.npy`
- **Benefits**: 
  - Faster subsequent runs
  - Offline capability
  - Reduced API calls

## Troubleshooting

### Issue: API Connection Fails

**Symptoms**: 
```
❌ Planetary Computer failed: Connection error
```

**Solutions**:
1. Check internet connection
2. Try fallback source: `--source worldcover`
3. Verify firewall settings
4. Check API status at https://planetarycomputer.microsoft.com/

### Issue: No Data Found

**Symptoms**:
```
⚠️ No data found for {year}
```

**Solutions**:
1. Verify year availability (2017-2022 for PC, 2020-2021 for WC)
2. Check bounding box coordinates
3. Try different data source
4. Expand search radius

### Issue: Import Errors

**Symptoms**:
```
ImportError: No module named 'pystac_client'
```

**Solutions**:
```bash
pip install pystac-client planetary-computer rasterio stackstac xarray
```

### Issue: Insufficient Data Quality

**Symptoms**:
```
⚠️ Data validation failed
```

**Solutions**:
1. Try different year
2. Adjust quality threshold in config
3. Use fallback data source
4. Check for cloud cover in region

### Issue: Memory Errors

**Symptoms**:
```
MemoryError: Unable to allocate array
```

**Solutions**:
1. Reduce patch size: `--patch_size 128`
2. Process fewer years at once
3. Increase system RAM
4. Use data streaming instead of loading all at once

## Advanced Usage

### Custom Region

Fetch data for any location:

```python
from src.real_data.api_fetcher import KovilpattiDataFetcher

fetcher = KovilpattiDataFetcher(
    region_name="MyRegion",
    lat=10.0,
    lon=78.0,
    radius_km=15,
    cache_dir="./my_cache"
)

data = fetcher.fetch_temporal_sequence([2020, 2021, 2022])
```

### Custom Preprocessing

```python
from src.real_data.preprocessor import RealDataPreprocessor

preprocessor = RealDataPreprocessor(n_classes=7)

# Remap classes
remapped = preprocessor.remap_classes(raw_data, source="io")

# Create patches
patches = preprocessor.crop_to_patches(remapped, patch_size=512, overlap=64)

# Custom splits
splits = preprocessor.split_train_val_test(
    inputs, targets,
    ratios=(0.8, 0.1, 0.1)
)
```

### Export to GeoTIFF

```python
from src.real_data.utils import export_to_geotiff

export_to_geotiff(
    lulc_array,
    output_path="kovilpatti_2022.tif",
    metadata={'bounds': [77.77, 9.07, 77.97, 9.27], 'crs': 'EPSG:4326'}
)
```

### Calculate Transition Matrix

```python
from src.real_data.utils import calculate_real_transition_matrix

transition_matrix = calculate_real_transition_matrix(
    years_data={2020: data_2020, 2021: data_2021, 2022: data_2022},
    n_classes=7
)
```

## Performance Optimization

### Parallel Downloads

For multiple regions:
```python
from concurrent.futures import ThreadPoolExecutor

regions = [
    ("Region1", 9.17, 77.87),
    ("Region2", 10.0, 78.0),
]

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(fetch_region, name, lat, lon)
        for name, lat, lon in regions
    ]
    results = [f.result() for f in futures]
```

### Streaming Large Data

For very large regions:
```python
# Process data in chunks
for year in years:
    data = fetcher.fetch_lulc_data(year)
    # Process immediately
    patches = preprocessor.crop_to_patches(data)
    # Save and clear memory
    save_patches(patches, year)
    del data, patches
```

## Configuration

Edit `configs/config_real_data.yaml` to customize:

```yaml
real_data:
  source: planetary_computer
  region:
    name: Kovilpatti
    lat: 9.17
    lon: 77.87
    radius_km: 10
  years: [2020, 2021, 2022]
  preprocessing:
    patch_size: 256
    overlap: 0
```

## Best Practices

1. **Always cache**: Don't re-download unnecessarily
2. **Verify data**: Run verification script after download
3. **Handle errors**: Use try-except and fallbacks
4. **Document sources**: Track which API provided your data
5. **Version control**: Track data versions and timestamps
6. **Quality checks**: Validate class distributions and spatial patterns

## FAQ

**Q: Do I need authentication?**  
A: No, both Planetary Computer and ESA WorldCover are freely accessible.

**Q: How much data will be downloaded?**  
A: For Kovilpatti (20km × 20km, 10m resolution), expect ~10-50 MB per year.

**Q: Can I use this for other regions?**  
A: Yes! Just specify different lat/lon coordinates.

**Q: What if my region spans multiple STAC items?**  
A: The fetcher automatically mosaics multiple items.

**Q: How do I add more data sources?**  
A: Create a new fetcher class similar to `WorldCoverFetcher` in `src/real_data/`.

**Q: Can I mix real and synthetic data?**  
A: Yes, both use the same format. Load them separately and combine as needed.

## Resources

- [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/)
- [ESA WorldCover](https://esa-worldcover.org/)
- [STAC Specification](https://stacspec.org/)
- [pystac-client Documentation](https://pystac-client.readthedocs.io/)
- [Rasterio Documentation](https://rasterio.readthedocs.io/)

## Support

For issues:
1. Check troubleshooting section above
2. Run verification script
3. Enable debug logging
4. Check API status pages
5. Open issue on GitHub

## License

This project uses data from:
- **Impact Observatory**: Check their usage terms
- **ESA WorldCover**: CC BY 4.0 License

Always cite data sources in publications!
