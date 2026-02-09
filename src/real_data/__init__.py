"""Real satellite data fetching and processing"""

# Conditional imports to handle optional dependencies
try:
    from .api_fetcher import fetch_kovilpatti_data, KovilpattiDataFetcher
    _API_AVAILABLE = True
except ImportError as e:
    _API_AVAILABLE = False
    fetch_kovilpatti_data = None
    KovilpattiDataFetcher = None

try:
    from .preprocessor import preprocess_for_model, RealDataPreprocessor
    _PREPROCESSOR_AVAILABLE = True
except ImportError:
    _PREPROCESSOR_AVAILABLE = False
    preprocess_for_model = None
    RealDataPreprocessor = None

__all__ = [
    'fetch_kovilpatti_data', 
    'preprocess_for_model',
    'KovilpattiDataFetcher',
    'RealDataPreprocessor'
]
