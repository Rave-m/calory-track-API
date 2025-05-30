import requests
import numpy as np
from PIL import Image
import io
import os
from fastapi import HTTPException

# conver to gram
def convert_weight_to_grams(weight):
    """
    Parameters:
        - weight (str): The weight string, e.g., "500mg", "1.2kg", "3t".
        
    Returns:
        - float: The converted weight in grams.
    """
    try:
        # Normalize input (lowercase and strip spaces)
        weight = weight.lower().strip()

        if "µg" in weight or "ug" in weight:
            value = float(weight.replace("µg", "").replace("ug", "").strip())
            return value / 1_000_000

        elif "mg" in weight:
            value = float(weight.replace("mg", "").strip())
            return value / 1000
        
        elif "kg" in weight:
            value = float(weight.replace("kg", "").strip())
            return value * 1000
        elif "g" in weight:
            value = float(weight.replace("g", "").strip())
            return value

        elif "t" in weight:
            value = float(weight.replace("t", "").strip())
            return value * 1_000_000

        else:
            raise ValueError("Unit not recognized. Please use mg, g, kg, or t.")

    except Exception as e:
        raise ValueError(f"Invalid weight format: {e}")
    
def safe_convert(value, unit):
    try:
        return float(value.replace(unit, "").strip().replace(",", "."))
    except (ValueError, AttributeError):
        return 0.0  

def get_image_from_url(url):
    """
    Download image from URL and return as bytes
    """
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        if not response.headers.get("Content-Type", "").startswith("image/"):
            raise ValueError("URL is not an image.")
        return response.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image download failed: {str(e)}")
    
def get_image_from_path(path):
    """
    Read image from local file path
    """
    try:
        if not os.path.exists(path):
            raise ValueError("File not found.")
        with open(path, "rb") as f:
            return f.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Reading local image failed: {str(e)}")
    
# Function to preprocess image for the model
def preprocess_image(image_bytes, target_size=(224, 224)):
    """
    Preprocess image bytes for the model
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(target_size)
    array = np.array(image) / 255.0
    return np.expand_dims(array, axis=0)