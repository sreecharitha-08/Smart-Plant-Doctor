import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

IMG_SIZE = 224

# 🔥 Load model safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "plant_model.h5")

try:
    model = load_model(MODEL_PATH, compile=False)
except Exception as e:
    print(f"⚠️ Warning: Could not load model from {MODEL_PATH}")
    print(f"Error: {e}")
    model = None

# IMPORTANT: Hardcode EXACT class order based on training print
class_names = [
    "Potato_Early_blight",
    "Potato_Late_blight",
    "Potato_healthy",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_healthy"
]

def predict_image(img_path):
    if model is None:
        return "Model not loaded", 0.0
    
    try:
        img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        predictions = model.predict(img_array)

        predicted_index = np.argmax(predictions)

        # SAFETY CHECK
        if predicted_index >= len(class_names):
            return "Unknown", 0.0

        predicted_class = class_names[predicted_index]
        confidence = float(np.max(predictions) * 100)

        return predicted_class, round(confidence, 2)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error", 0.0
