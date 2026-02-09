# LULC-Kovilpatti-Prediction

Land Use Land Cover prediction for Kovilpatti, Tamil Nadu using Causal Spatiotemporal Transformer

## 🌟 Features

- **Real Satellite Data Integration**: Automatic API-based fetching from Microsoft Planetary Computer and ESA WorldCover
- **No Manual Downloads**: Everything is fetched automatically via API calls
- **Multi-Temporal Analysis**: Process data from multiple years (2017-2022)
- **High Resolution**: 10-meter resolution satellite imagery
- **Flexible**: Works with both real and synthetic data

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/prasanthnecit/LULC-Kovilpatti-Prediction.git
cd LULC-Kovilpatti-Prediction

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### Option 1: Use Real Satellite Data (Recommended)

```bash
# Step 1: Fetch real LULC data for Kovilpatti via API
python scripts/fetch_real_data.py --region kovilpatti --years 2020 2021 2022

# Step 2: Verify the downloaded data
python scripts/verify_real_data.py --data_dir data/Kovilpatti_LULC_Real/

# Step 3: Train on real data
python scripts/run_training_real.py --data_dir data/Kovilpatti_LULC_Real/ --real_data
```

#### Option 2: Use Synthetic Data

```bash
# Generate synthetic data (if you have a synthetic data generator)
# Then train on it
python scripts/run_training_real.py --data_dir data/synthetic/
```

## 🛰️ Real Satellite Data

### Automatic API-Based Fetching

This project supports **automatic downloading** of real satellite LULC data—no manual downloads required!

#### Data Sources

1. **Microsoft Planetary Computer** (Primary)
   - Collection: Impact Observatory LULC (io-lulc-annual-v02)
   - Resolution: 10m
   - Years: 2017-2022
   - Coverage: Global
   - Free access, no authentication

2. **ESA WorldCover** (Fallback)
   - Collection: ESA WorldCover
   - Resolution: 10m
   - Years: 2020-2021
   - Coverage: Global
   - Free access, no authentication

#### Automatic Fallback

The system automatically tries multiple sources:
```bash
python scripts/fetch_real_data.py --source both
```

If Planetary Computer is unavailable, it automatically falls back to ESA WorldCover.

### Kovilpatti Region

- **Location**: Kovilpatti, Tamil Nadu, India
- **Coordinates**: 9.17°N, 77.87°E
- **Coverage Area**: ~20km × 20km
- **Bounding Box**: [77.77, 9.07, 77.97, 9.27]

### Requirements

- Internet connection (for initial fetch)
- No authentication required
- Data is cached locally after first download

### Advanced Usage

```bash
# Custom region
python scripts/fetch_real_data.py --lat 10.0 --lon 78.0 --radius 15

# Different years
python scripts/fetch_real_data.py --years 2018 2019 2020

# Visualization
python scripts/fetch_real_data.py --visualize

# Custom patch size
python scripts/fetch_real_data.py --patch_size 512 --overlap 64
```

## 📊 Data Processing

### Class Mapping

Real LULC data is automatically mapped to our 7 standard classes:

| Class ID | Class Name  | Description                    |
|----------|-------------|--------------------------------|
| 0        | Urban       | Built-up areas, cities         |
| 1        | Forest      | Trees, dense vegetation        |
| 2        | Agriculture | Cropland, farmland             |
| 3        | Water       | Rivers, lakes, water bodies    |
| 4        | Barren      | Bare ground, rocky areas       |
| 5        | Wetland     | Marshes, flooded vegetation    |
| 6        | Grassland   | Grassland, shrubland, rangeland|

### Processing Pipeline

1. **Fetch**: Download from API
2. **Remap**: Convert to standard classes
3. **Patch**: Create 256×256 patches
4. **Temporal**: Create multi-year sequences
5. **Split**: Train/val/test (70%/15%/15%)
6. **Save**: Export as .npy files

## 📁 Project Structure

```
LULC-Kovilpatti-Prediction/
├── src/
│   └── real_data/
│       ├── __init__.py
│       ├── api_fetcher.py          # Planetary Computer fetcher
│       ├── worldcover_api.py        # ESA WorldCover fetcher
│       ├── preprocessor.py          # Data preprocessing
│       └── utils.py                 # Utility functions
├── scripts/
│   ├── fetch_real_data.py          # Main data fetching script
│   ├── run_training_real.py        # Training script
│   └── verify_real_data.py         # Data verification
├── notebooks/
│   └── Real_Data_API_Fetch.ipynb   # Interactive notebook
├── configs/
│   └── config_real_data.yaml       # Configuration file
├── docs/
│   └── REAL_DATA_GUIDE.md          # Detailed documentation
├── data/
│   ├── cache/                       # Cached raw downloads
│   └── Kovilpatti_LULC_Real/       # Processed training data
├── requirements.txt
└── README.md
```

## 🔧 Configuration

Edit `configs/config_real_data.yaml` to customize:

- Data sources
- Region coordinates
- Years to fetch
- Preprocessing parameters
- Training hyperparameters

## 📚 Documentation

- **[Real Data Guide](docs/REAL_DATA_GUIDE.md)**: Complete guide for using real satellite data
- **[Jupyter Notebook](notebooks/Real_Data_API_Fetch.ipynb)**: Interactive tutorial

## 🧪 Testing

Verify your installation and data:

```bash
# Test data fetcher (will download small test dataset)
python -m src.real_data.api_fetcher

# Verify downloaded data
python scripts/verify_real_data.py --data_dir data/Kovilpatti_LULC_Real/
```

## 📦 Dependencies

Key packages:
- `torch`: Deep learning framework
- `pystac-client`: STAC API client
- `planetary-computer`: Microsoft Planetary Computer access
- `rasterio`: Geospatial raster I/O
- `numpy`, `pandas`: Data processing
- `matplotlib`: Visualization

See `requirements.txt` for complete list.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional data sources (Sentinel-2, Landsat, etc.)
- More preprocessing options
- Advanced model architectures
- Visualization tools
- Performance optimizations

## 📄 License

This project is open source. Please cite data sources:
- Impact Observatory LULC data
- ESA WorldCover data (CC BY 4.0)

## 🙏 Acknowledgments

- **Microsoft Planetary Computer**: For free API access
- **ESA WorldCover**: For global land cover data
- **Impact Observatory**: For high-quality LULC classification

## 📞 Support

For issues or questions:
1. Check [documentation](docs/REAL_DATA_GUIDE.md)
2. Run verification script
3. Open an issue on GitHub

## 🗺️ Roadmap

- [ ] Add Google Earth Engine integration
- [ ] Support for Sentinel-2 raw imagery
- [ ] Real-time prediction API
- [ ] Web-based visualization dashboard
- [ ] Multi-region batch processing
- [ ] Cloud-based training pipeline
