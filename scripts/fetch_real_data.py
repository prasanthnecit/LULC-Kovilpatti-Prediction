#!/usr/bin/env python
"""
Fetch real satellite LULC data for Kovilpatti region via API
NO manual downloads required!

Usage:
    python scripts/fetch_real_data.py --region kovilpatti
    python scripts/fetch_real_data.py --region kovilpatti --years 2020 2021 2022
    python scripts/fetch_real_data.py --lat 9.17 --lon 77.87 --radius 10
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check for required dependencies
try:
    from src.real_data.api_fetcher import KovilpattiDataFetcher
    from src.real_data.worldcover_api import WorldCoverFetcher
    from src.real_data.preprocessor import preprocess_for_model
    from src.real_data.utils import print_data_summary, visualize_real_data
except ImportError as e:
    print("❌ Missing required dependencies!")
    print(f"\nError: {e}")
    print("\n💡 Please install required packages:")
    print("   pip install -r requirements.txt")
    print("\nOr install core packages:")
    print("   pip install pystac-client planetary-computer rasterio numpy matplotlib")
    sys.exit(1)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Fetch real satellite LULC data via API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--region',
        type=str,
        default='kovilpatti',
        help='Region name'
    )
    
    parser.add_argument(
        '--lat',
        type=float,
        default=9.17,
        help='Latitude of region center'
    )
    
    parser.add_argument(
        '--lon',
        type=float,
        default=77.87,
        help='Longitude of region center'
    )
    
    parser.add_argument(
        '--radius',
        type=float,
        default=10,
        help='Radius in kilometers'
    )
    
    parser.add_argument(
        '--years',
        type=int,
        nargs='+',
        default=[2020, 2021, 2022],
        help='Years to fetch'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/Kovilpatti_LULC_Real',
        help='Output directory for processed data'
    )
    
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='data/cache',
        help='Cache directory for raw downloads'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        choices=['planetary_computer', 'worldcover', 'both'],
        default='planetary_computer',
        help='Data source to use'
    )
    
    parser.add_argument(
        '--patch_size',
        type=int,
        default=256,
        help='Size of patches for training'
    )
    
    parser.add_argument(
        '--overlap',
        type=int,
        default=0,
        help='Overlap between patches'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize downloaded data'
    )
    
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_args()
    
    print("=" * 80)
    print(" 🛰️  REAL SATELLITE DATA FETCHER")
    print("=" * 80)
    print(f"\n📍 Region: {args.region.title()}")
    print(f"🌍 Location: ({args.lat}°N, {args.lon}°E)")
    print(f"📏 Radius: {args.radius} km")
    print(f"📅 Years: {args.years}")
    print(f"📡 Source: {args.source}")
    print(f"💾 Output: {args.output_dir}")
    print(f"🗄️  Cache: {args.cache_dir}")
    print()
    
    # Step 1: Fetch data from API
    print("\n" + "=" * 80)
    print(" STEP 1: Fetching Data from API")
    print("=" * 80)
    
    years_data = {}
    
    if args.source in ['planetary_computer', 'both']:
        print("\n🛰️ Trying Planetary Computer (Impact Observatory LULC)...")
        try:
            fetcher = KovilpattiDataFetcher(
                region_name=args.region,
                lat=args.lat,
                lon=args.lon,
                radius_km=args.radius,
                cache_dir=args.cache_dir
            )
            years_data = fetcher.fetch_temporal_sequence(args.years)
        except Exception as e:
            print(f"❌ Planetary Computer failed: {e}")
            if args.source == 'planetary_computer':
                print("💡 Hint: Try --source worldcover or install: pip install pystac-client planetary-computer")
    
    # Fallback to WorldCover if needed
    if (args.source in ['worldcover', 'both']) and len(years_data) == 0:
        print("\n🌍 Trying ESA WorldCover...")
        try:
            fetcher = WorldCoverFetcher(
                region_name=args.region,
                lat=args.lat,
                lon=args.lon,
                radius_km=args.radius,
                cache_dir=args.cache_dir
            )
            years_data = fetcher.fetch_temporal_sequence(args.years)
        except Exception as e:
            print(f"❌ WorldCover failed: {e}")
    
    if not years_data:
        print("\n❌ No data fetched! Check your internet connection and API availability.")
        print("💡 Make sure you have installed: pip install pystac-client planetary-computer rasterio")
        return 1
    
    print(f"\n✅ Successfully fetched {len(years_data)} years of data")
    
    # Step 2: Visualize (optional)
    if args.visualize:
        print("\n" + "=" * 80)
        print(" STEP 2: Visualizing Data")
        print("=" * 80)
        
        for year, data in years_data.items():
            print(f"\n📊 {year}:")
            print_data_summary(data, f"LULC Data - {year}")
            # Note: visualization requires matplotlib display capabilities
            # visualize_real_data(data, year)
    
    # Step 3: Preprocess for model
    print("\n" + "=" * 80)
    print(" STEP 3: Preprocessing for Model")
    print("=" * 80)
    
    source_type = "io" if args.source == 'planetary_computer' else "esa"
    
    try:
        splits = preprocess_for_model(
            years_data,
            output_dir=args.output_dir,
            patch_size=args.patch_size,
            source=source_type
        )
        
        if not splits:
            print("\n❌ Preprocessing failed!")
            return 1
            
    except Exception as e:
        print(f"\n❌ Preprocessing error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 4: Summary
    print("\n" + "=" * 80)
    print(" SUMMARY")
    print("=" * 80)
    print(f"\n✅ Data fetch and preprocessing complete!")
    print(f"\n📂 Output directory: {args.output_dir}")
    print(f"   - train_inputs.npy / train_targets.npy")
    print(f"   - val_inputs.npy / val_targets.npy")
    print(f"   - test_inputs.npy / test_targets.npy")
    print(f"   - metadata.txt")
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Train samples: {len(splits['train']['inputs'])}")
    print(f"   Val samples: {len(splits['val']['inputs'])}")
    print(f"   Test samples: {len(splits['test']['inputs'])}")
    print(f"   Input shape: {splits['train']['inputs'][0].shape}")
    print(f"   Target shape: {splits['train']['targets'][0].shape}")
    
    print(f"\n🎯 Next step: Train your model!")
    print(f"   python scripts/run_training_real.py --data_dir {args.output_dir}")
    
    print("\n" + "=" * 80)
    print(" ✨ ALL DONE! ✨")
    print("=" * 80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
