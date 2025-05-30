"""
Helper package for the calorie-track project.

This package provides functionality for scraping nutrition data,
food information, and various utility functions.
"""

# Import and expose functions from scrap.py
from .scrap import (
    scrape_nutrition_data,
    scrape_portion_links,
    scrape_portion_nutrition,
)

# Import and expose data from food.py
from .food import food_list

# Import and expose utility functions
from .functions import (
    convert_weight_to_grams,
    safe_convert,
    get_image_from_url,
    get_image_from_path,
    preprocess_image
)

# Define what gets imported with "from helper import *"
__all__ = [
    # Scraping functions
    'scrape_nutrition_data',
    'scrape_portion_links',
    'scrape_portion_nutrition',
    
    # Food data
    'food_list',
    
    # Utility functions
    'convert_weight_to_grams',
    'safe_convert',
    'get_image_from_url',
    'get_image_from_path',
    'preprocess_image'
]