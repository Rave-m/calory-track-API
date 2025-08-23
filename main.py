import tensorflow as tf
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np

from helper import (
    scrape_nutrition_data,
    scrape_portion_nutrition,
    scrape_search_list,
    preprocess_image,
    food_list
)

# Load the saved model
model_path = "model"
model = tf.saved_model.load(model_path)
infer = model.signatures["serving_default"]

app = FastAPI()

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
    
@app.post("/scan_food")
async def scan_food(
    image: UploadFile = File(...)
):
    """
    Scan food image and detect food type with nutrition data
    
    Parameters:
        - image: Multipart form-data file field named 'image'
    """
    try:
        if image is None:
            raise HTTPException(status_code=400, detail="Image file required.")

        if image.content_type and not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

        image_bytes = await image.read()

        image_array = preprocess_image(image_bytes)
        input_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)

        result = infer(input_tensor)
        prediction = list(result.values())[0].numpy()

        pred_idx = np.argmax(prediction)
        confidence = float(prediction[0][pred_idx])
        
        # Low confidence -> unknown image
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
                "message": "Model ini hanya dapat mengenali 10 jenis makanan Indonesia: " + ", ".join(food_list)
            }
        
        detected_food = food_list[pred_idx] if pred_idx < len(food_list) else "unknown"
        
        # Try to get base nutrition and default volume info (usually 100 gram)
        try:
            nutrition_data, volume_info = scrape_nutrition_data(detected_food)
            nutrition_info = {
                "Kalori": nutrition_data.get("Kalori", "0 kcal"),
                "Lemak": nutrition_data.get("Lemak", "0 g"),
                "Karbohidrat": nutrition_data.get("Karbohidrat", "0 g"),
                "Protein": nutrition_data.get("Protein", "0 g"),
            }
            food_registered = True
        except Exception:
            nutrition_info = {
                "Kalori": "0 kcal",
                "Lemak": "0 g",
                "Karbohidrat": "0 g",
                "Protein": "0 g",
            }
            food_registered = False
            volume_info = "unknown"
        
        # Build volume_list as objects containing nutrition_info per portion
        volume_list = []
        if food_registered:
            try:
                portions = scrape_portion_nutrition(detected_food) or []
                for p in portions:
                    p_nut = {
                        "Kalori": p.get("Kalori", "0 kcal"),
                        "Lemak": p.get("Lemak", "0 g"),
                        "Karbohidrat": p.get("Karbohidrat", "0 g"),
                        "Protein": p.get("Protein", "0 g"),
                    }
                    volume_list.append({
                        "nutrition_info": p_nut,
                        "volume": p.get("volume", p.get("text", ""))
                    })
                    
                # Ensure 100 gram present (use base nutrition_data if missing)
                if not any(("100 gram" in item.get("volume","").lower() or "100 gr" in item.get("volume","").lower()) for item in volume_list):
                    volume_list.insert(0, {
                        "nutrition_info": nutrition_info,
                        "volume": volume_info if volume_info else "100 gram"
                    })
            except Exception:
                # fallback to single 100 gram entry using base nutrition
                volume_list = [{
                    "nutrition_info": nutrition_info,
                    "volume": volume_info if volume_info else "100 gram"
                }]
        else:
            # not registered -> empty list
            volume_list = []

        response = {
            "food_name": detected_food if food_registered else "makanan tidak terdaftar",
            "confidence": round(float(confidence), 2),
            "nutrition_info": nutrition_info,
            "volume": volume_info if food_registered else "unknown",
            "volume_list": volume_list
        }

        if not food_registered:
            response["message"] = (
                f"Makanan terdeteksi: '{detected_food}' tetapi tidak ada di database. "
                "Model hanya dapat mengenali: " + ", ".join(food_list)
            )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/food_nutrition", response_model=FoodNutritionResponse)
async def food_nutrition(data: FoodNutritionRequest):
    """
    Get nutrition information for a food item.
    Behaviour:
      - If food found: return same structure as /scan_food (nutrition_info + volume_list objects).
      - If no specific volume provided: 'volume' will be the top (first) item from volume_list.
      - If food not found: return "makanan tidak terdaftar" response.
    """
    food_name = data.name

    if not food_name:
        raise HTTPException(status_code=400, detail="'name' must be provided.")

    not_registered_resp = {
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

    try:
        nutrition_data, base_volume = scrape_nutrition_data(food_name)

        # treat as not-registered if scraping returned empty/zero data
        if not nutrition_data or all(nutrition_data.get(k, "0") == "0" for k in ["Kalori", "Lemak", "Karbohidrat", "Protein"]):
            return not_registered_resp

        base_nut = {
            "Kalori": nutrition_data.get("Kalori", "0 kcal"),
            "Lemak": nutrition_data.get("Lemak", "0 g"),
            "Karbohidrat": nutrition_data.get("Karbohidrat", "0 g"),
            "Protein": nutrition_data.get("Protein", "0 g"),
        }

        # build volume_list where each entry contains nutrition_info
        try:
            portions = scrape_portion_nutrition(food_name) or []
            volume_list = []
            for p in portions:
                p_nut = {
                    "Kalori": p.get("Kalori", "0 kcal"),
                    "Lemak": p.get("Lemak", "0 g"),
                    "Karbohidrat": p.get("Karbohidrat", "0 g"),
                    "Protein": p.get("Protein", "0 g"),
                }
                volume_list.append({
                    "nutrition_info": p_nut,
                    "volume": p.get("volume", p.get("text", ""))
                })

            # ensure 100 gram present at top as fallback
            if not any(("100 gram" in item.get("volume","").lower() or "100 gr" in item.get("volume","").lower()) for item in volume_list):
                volume_list.insert(0, {
                    "nutrition_info": base_nut,
                    "volume": base_volume if base_volume else "100 gram"
                })
                
        except Exception:
            volume_list = [{
                "nutrition_info": base_nut,
                "volume": base_volume if base_volume else "100 gram"
            }]

        # choose top-most volume as the selected volume
        selected_volume = volume_list[0]["volume"] if volume_list else (base_volume if base_volume else "unknown")

        return {
            "food_name": food_name,
            "nutrition_info": base_nut,
            "volume": selected_volume,
            "volume_list": volume_list
        }

    except Exception:
        return not_registered_resp
    
@app.get("/food_search")
async def search_food(query: Optional[str] = None):
    """
    Search food by query string (GET /food_search?query=...)
    """
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="'query' parameter is required.")
    try:
        results = scrape_search_list(query.strip())
        return {"query": query.strip(), "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching for '{query}': {str(e)}")