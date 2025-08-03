import uvicorn
from main import model_path, food_list

# When running as main, start the server
if __name__ == "__main__":
    print("Loading TensorFlow model from:", model_path)
    print(f"Model loaded successfully. Available food classes: {food_list}")
    uvicorn.run("main:app", host="0.0.0.0", port=9099, reload=False)