"""Real satellite data fetching and processing"""
from .api_fetcher import fetch_kovilpatti_data
from .preprocessor import preprocess_for_model

__all__ = ['fetch_kovilpatti_data', 'preprocess_for_model']
