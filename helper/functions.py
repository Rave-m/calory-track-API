import numpy as np
import cv2

# klasifikasi gambar
def classify_image(image_path, model):
    """
    Mengklasifikasikan gambar menggunakan model yang diberikan.

    Args:
        image_path (str): Path ke gambar yang akan diklasifikasikan.
        model (keras.Model): Model yang digunakan untuk klasifikasi.

    Returns:
        str: Kelas yang diprediksi oleh model.
    """
    # Load dan preprocess gambar
    img = load_and_preprocess_image(image_path)
    
    # Melakukan prediksi
    predictions = model.predict(img)
    
    # Mendapatkan kelas dengan probabilitas tertinggi
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    return predicted_class

def food_clasification():
    """
    Parameters:
        - features : [food_name, volume (optional)]

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
    
    if volume is not None:
        volume_convert = convert_weight_to_grams(volume)

    try:
        proteins, calories, carbohydrates, fat, sugar = fetch_nutritions(food_name)

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