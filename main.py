import tensorflow as tf
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np

from helper import (
    scrape_nutrition_data,
    scrape_portion_nutrition,
    convert_weight_to_grams,
    safe_convert,
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
        - Detected food type
        - Nutrition data of the detected food
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
            nutrition_info = {
                "Kalori": "0 kcal",
                "Lemak": "0 g",
                "Karbohidrat": "0 g",
                "Protein": "0 g",
            }
            volume_info = "unknown"
            
        # Return prediction and nutrition info
        return {
            "food_name": food_name,
            "confidence": round(float(confidence), 2),
            "nutrition_info": nutrition_info,
            "volume": volume_info if 'volume_info' in locals() else "unknown"
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
        - food_name: Name of the food
        - nutrition_info: Object containing nutrition values
        - volume: Volume used for calculation
    """
    food_name = data.name
    
    if not food_name:
        raise HTTPException(status_code=400, detail="'name' must be provided.")
    
    try:
        nutrition_info = {}
        # Use scrape_nutrition_data for standard portions
        nutrition_data, volume = scrape_nutrition_data(food_name)
        # example output: nutrition_data={'Kalori': '260 kcal', 'Lemak': '14.55 g', 'Karbohidrat': '10.76 g', 'Protein': '21.93 g'}, volume='100 gram(g)'
        
        # Map the keys to the expected output format
        nutrition_info = {
            "Kalori" : nutrition_data.get("Kalori", "0 kcal"),
            "Lemak": nutrition_data.get("Lemak", "0 g"),
            "Karbohidrat": nutrition_data.get("Karbohidrat", "0 g"),
            "Protein": nutrition_data.get("Protein", "0 g")
        }
    
        return {
            "food_name": food_name,
            "nutrition_info": nutrition_info,
            "volume": volume
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add a new endpoint to get all available portion options
@app.post("/food_portions")
def food_portions(data: FoodNutritionRequest):
    """
    Get all available portion options and nutrition for a food
    
    Parameters:
        - name: Food name
    
    Returns:
        - Detailed nutrition information for each available portion
    """
    food_name = data.name
    
    if not food_name:
        raise HTTPException(status_code=400, detail="'name' must be provided.")
    
    try:
        result = scrape_portion_nutrition(food_name)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))