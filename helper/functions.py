import numpy as np
from PIL import Image
import io
    
# Function to preprocess image for the model
def preprocess_image(image_bytes, target_size=(224, 224)):
    """
    Preprocess image bytes for the model
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(target_size)
    array = np.array(image) / 255.0
    return np.expand_dims(array, axis=0)