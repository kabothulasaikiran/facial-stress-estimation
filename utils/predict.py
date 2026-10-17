# utils/predict.py
# This file is used by the Flask web app (Phase 6)
# It provides a reusable prediction function

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load model once when this file is imported
# This avoids reloading the model on every request
MODEL_PATH  = os.path.join("model", "stress_model.h5")
CLASS_NAMES = ["Low Stress", "Moderate Stress", "High Stress"]
IMG_SIZE    = 224

# Load model into memory
model = load_model(MODEL_PATH)

def predict_stress_from_array(face_img_array):
    """
    Predicts stress level from a face image numpy array.

    Parameters:
        face_img_array: numpy array of the face image (any size, BGR)

    Returns:
        label      (str)  : 'Low Stress', 'Moderate Stress', or 'High Stress'
        confidence (float): confidence percentage 0-100
        all_probs  (list) : probabilities for all 3 classes
    """

    # Resize to model input size
    img = cv2.resize(face_img_array, (IMG_SIZE, IMG_SIZE))

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize pixels to 0.0-1.0
    img = img.astype(np.float32) / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Predict
    predictions = model.predict(img, verbose=0)
    all_probs   = predictions[0]

    # Get top prediction
    predicted_index = np.argmax(all_probs)
    label           = CLASS_NAMES[predicted_index]
    confidence      = float(all_probs[predicted_index]) * 100

    return label, confidence, all_probs.tolist()