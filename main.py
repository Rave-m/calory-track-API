import tensorflow as tf
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np

from helper import (
    scrape_nutrition_data,
    scrape_portion_nutrition,
    scrape_portion_links,
    get_image_from_path,
    get_image_from_url,
    preprocess_image,
    food_list
)

# # Load the saved model
model_path = "model"
model = tf.saved_model.load(model_path)
infer = model.signatures["serving_default"]

app = FastAPI(title="Calorie Track API", 
              description="API for tracking nutrition and calories in Indonesian food",
              version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Pydantic models
class FoodNutritionRequest(BaseModel):
    name: str
    volume: Optional[str] = None

class NutritionInfo(BaseModel):
    Kalori: str
    Lemak: str
    Karbohidrat: str
    Protein: str

class FoodNutritionResponse(BaseModel):
    food_name: str
    nutrition_info: NutritionInfo
    volume: str
    volume_list: list

class ScanFoodRequest(BaseModel):
    image_url: str | None = None
    image_path: str | None = None

@app.get("/")
def index():
    return {"message": "Hayo Cari Apaaa?"}

@app.post("/scan_food")
async def scan_food(
    data: ScanFoodRequest = Body(...)
):
    """
    Scan food image and detect food type with nutrition data
    
    Parameters:
        - image: URL to an image 
        
    Returns:
        - Detected food type (or "gambar tidak dikenali" if confidence < 0.75)
        - Confidence score
        - Nutrition data of the detected food
        - Available volume options
        - Message explaining model limitations (if applicable)
        - Detected food name (if not in database)
        
    Note: Model can only recognize 10 Indonesian food types: Bakso Daging Sapi, Bebek Betutu, Gado-Gado, 
          Gudeg, Nasi Goreng, Pempek, Rawon, Rendang, Sate Ayam, Soto Ayam
    """
    try:
        if not data.image_url and not data.image_path:
            raise HTTPException(status_code=400, detail="Image URL or path required.")
        
        if data.image_url:
            image_bytes = get_image_from_url(data.image_url)
        else:
            image_bytes = get_image_from_path(data.image_path)

        image_array = preprocess_image(image_bytes)
        input_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)

        result = infer(input_tensor)
        prediction = list(result.values())[0].numpy()

        pred_idx = np.argmax(prediction)
        confidence = float(prediction[0][pred_idx])
        
        # Check confidence threshold
        if confidence < 0.75:
            return {
                "food_name": "gambar tidak dikenali",
                "confidence": round(confidence, 2),
                "nutrition_info": {
                    "Kalori": "0 kcal",
                    "Lemak": "0 g",
                    "Karbohidrat": "0 g",
                    "Protein": "0 g",
                },
                "volume": "unknown",
                "volume_list": [],
                "message": "Model ini hanya dapat mengenali 10 jenis makanan Indonesia: Bakso Daging Sapi, Bebek Betutu, Gado-Gado, Gudeg, Nasi Goreng, Pempek, Rawon, Rendang, Sate Ayam, Soto Ayam",

            }
        
        food_name = food_list[pred_idx] if pred_idx < len(food_list) else "unknown"        
        try:
            # Unpacking tuple return value properly
            nutrition_data, volume_info = scrape_nutrition_data(food_name)
            
            
            nutrition_info = {
                "Kalori": nutrition_data.get("Kalori", "0 kcal"),
                "Lemak": nutrition_data.get("Lemak", "0 g"),
                "Karbohidrat": nutrition_data.get("Karbohidrat", "0 g"),
                "Protein": nutrition_data.get("Protein", "0 g"),
            }
        except Exception:
            # Handle case when detected food is not in database
            nutrition_info = {
                "Kalori": "0 kcal",
                "Lemak": "0 g",
                "Karbohidrat": "0 g",
                "Protein": "0 g",
            }
            
        # Get all available volume options
        try:
            portion_links = scrape_portion_links(food_name)
            volume_list = [portion["text"] for portion in portion_links]
            # Ensure "100 gram" is always included if not present
            if "100 gram" not in volume_list:
                volume_list.insert(0, "100 gram")
        except Exception:
            volume_list = ["100 gram"]  # Default fallback
            
        # Return prediction and nutrition info
        return {
            "food_name": food_name,
            "confidence": round(float(confidence), 2),
            "nutrition_info": nutrition_info,
            "volume": volume_info if 'volume_info' in locals() else "unknown",
            "volume_list": volume_list
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/food_nutrition", response_model=FoodNutritionResponse)
def food_nutrition(data: FoodNutritionRequest):
    """
    Get nutrition information for a food item with optional volume
    
    Parameters:
        - name: Food name
    Returns:
        - food_name: Name of the food (or "makanan tidak terdaftar" if not found)
        - nutrition_info: Object containing nutrition values
        - volume: Volume used for calculation
        - volume_list: List of all available volume options
    """
    food_name = data.name
    
    if not food_name:
        raise HTTPException(status_code=400, detail="'name' must be provided.")
    
    try:
        # Use scrape_nutrition_data for standard portions
        nutrition_data, volume = scrape_nutrition_data(food_name)
        
        # Check if nutrition data is empty or contains no valid nutritional information
        if not nutrition_data or all(nutrition_data.get(key, "0") == "0" for key in ["Kalori", "Lemak", "Karbohidrat", "Protein"]):
            return {
                "food_name": "makanan tidak terdaftar",
                "nutrition_info": {
                    "Kalori": "0 kcal",
                    "Lemak": "0 g",
                    "Karbohidrat": "0 g",
                    "Protein": "0 g"
                },
                "volume": "unknown",
                "volume_list": []
            }
        
        # example output: nutrition_data={'Kalori': '260 kcal', 'Lemak': '14.55 g', 'Karbohidrat': '10.76 g', 'Protein': '21.93 g'}, volume='100 gram(g)'
        
        # Map the keys to the expected output format
        nutrition_info = {
            "Kalori" : nutrition_data.get("Kalori", "0 kcal"),
            "Lemak": nutrition_data.get("Lemak", "0 g"),
            "Karbohidrat": nutrition_data.get("Karbohidrat", "0 g"),
            "Protein": nutrition_data.get("Protein", "0 g")
        }
        
        # Get all available volume options
        try:
            portion_links = scrape_portion_links(food_name)
            volume_list = [portion["text"] for portion in portion_links]
            # Ensure "100 gram" is always included if not present
            if "100 gram" not in volume_list:
                volume_list.insert(0, "100 gram")
        except Exception:
            volume_list = ["100 gram"]  # Default fallback
    
        return {
            "food_name": food_name,
            "nutrition_info": nutrition_info,
            "volume": volume,
            "volume_list": volume_list
        }
    
    except Exception as e:
        # If there's an error in scraping, treat it as food not found
        return {
            "food_name": "makanan tidak terdaftar",
            "nutrition_info": {
                "Kalori": "0 kcal",
                "Lemak": "0 g",
                "Karbohidrat": "0 g",
                "Protein": "0 g"
            },
            "volume": "unknown",
            "volume_list": []
        }

# Add a new endpoint to get all available portion options
@app.post("/food_portions")
def food_portions(data: FoodNutritionRequest):
    """
    Get all available portion options and nutrition for a food
    
    Parameters:
        - name: Food name
        - volume: Optional specific portion to get (e.g., "100 gram", "1 porsi", "1 mangkok")
    
    Returns:
        - If volume specified: Single nutrition object for that portion
        - If volume not specified: Array of all available portions with nutrition
        - If food not found: Empty array or error message
    """
    food_name = data.name
    volume = data.volume
    
    if not food_name:
        raise HTTPException(status_code=400, detail="'name' must be provided.")
    
    try:
        # Get all portion nutrition data
        all_portions = scrape_portion_nutrition(food_name)
        
        # Check if no portions were found (food not registered)
        if not all_portions:
            return {
                "food_name": "makanan tidak terdaftar",
                "message": "Makanan tidak ditemukan dalam database",
                "available_portions": []
            }
        
        # If specific portion requested, filter and return only that portion
        if volume:
            filtered_portions = [
                portion for portion in all_portions 
                if portion.get("porsi", "").lower() == volume.lower()
            ]
            
            if not filtered_portions:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Portion '{volume}' not found for food '{food_name}'. Available portions: {[p.get('porsi') for p in all_portions]}"
                )
            
            return filtered_portions[0]  # Return single object
        
        # If no specific portion requested, return all available portions
        return all_portions
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        # If there's an error in scraping, treat it as food not found
        return {
            "food_name": "makanan tidak terdaftar",
            "message": "Makanan tidak ditemukan dalam database",
            "available_portions": []
        }