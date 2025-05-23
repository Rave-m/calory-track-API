import requests
from bs4 import BeautifulSoup
from flask_cors import CORS
from flask import Flask, request, jsonify

from helper.food import food_list
# from helper.functions import 
from helper.scrap import scrape_nutrition_data, scrape_portion_links, scrape_portion_nutrition

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/", methods=['GET'])
def index():
    return "Hayo Cari Apaaa?"

@app.route("/scan_food", methods=["POST"])
def scan_food():
    """
    Parameters:
        - image : image url

    Returns:
        - Detected objects in the image
        - Nutrition data of the detected objects
    """
    
    # data = scrape_nutrition_data()
    return jsonify({"nutrition": 'hallo'})

# @app.route("/food_nutrition", methods=["POST"])
def food_portion():
    """
    Parameters:
        - features : [food_name, volume (optional)]
i
    Returns:
        - food name
        - [Calories, Carbohydrates, Fat, Proteins]
        - alert (note recommendation for diabetes or not diabetes)
        - volume (gram)
    """
    
    data = request.json
    food_name = data.get('name')
    volume = data.get('volume')

    if not food_name:
        return jsonify({"error": "'food_name' must be provided."}), 400
    
    # if volume is not None:
    #     volume_convert = convert_weight_to_grams(volume)

    try:
        data = scrape_nutrition_data(food_name)

        # Convert values to floats to avoid type mismatch
        proteins = safe_convert(proteins, "g")
        calories = safe_convert(calories, "kcal")
        carbohydrates = safe_convert(carbohydrates, "g")
        fat = safe_convert(fat, "g")
        sugar = safe_convert(sugar, "g")
        
        # Ensure volume is a valid number
        volume_convert = float(volume_convert / 100) if volume is not None else 1

        proteins *= volume_convert
        calories *= volume_convert
        carbohydrates *= volume_convert
        fat *= volume_convert
        sugar *= volume_convert

        nutrition_info = {
            "proteins": "{:.2f} g".format(proteins),
            "calories": "{:.2f} kcal".format(calories),
            "carbohydrates": "{:.2f} g".format(carbohydrates),
            "fat": "{:.2f} g".format(fat),
            "sugar": "{:.2f} g".format(sugar)
        }

        if carbohydrates == 0 and calories == 0 and proteins == 0 and fat == 0 and sugar == 0:
            alert = "Food not found"
        elif (carbohydrates < max_carbs and 
            calories < max_calories and 
            proteins < max_protein and 
            fat < max_fat):
            alert = "Suitable for diabetes"
        else:
            alert = "Not recommended for diabetes"
        
        return jsonify({
            "food_name": food_name,
            "nutrition_info": nutrition_info,
            "alert": alert,
            "volume": "100 g" if volume is None else f"{volume}"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)