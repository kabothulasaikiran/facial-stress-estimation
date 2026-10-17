# predict_single.py
# Phase 4 - Predict stress level from a single image
#
# What this script does:
# 1. Loads your saved trained model
# 2. Takes one face image as input
# 3. Preprocesses it exactly like training data
# 4. Predicts the stress level
# 5. Shows the result with confidence percentage
# ============================================================

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ── 1. Configuration ──────────────────────────────────────────
# Image size must match what we used during training
IMG_SIZE   = 224

# Path to your saved model
MODEL_PATH = os.path.join("model", "stress_model.h5")

# Class names matching our 3 stress levels
# Order must match training: 0=Low, 1=Moderate, 2=High
CLASS_NAMES = ["Low Stress", "Moderate Stress", "High Stress"]

# Emoji for each stress level to make output friendly
CLASS_EMOJI = ["😊", "😐", "😰"]

# Color for each level (used when showing image with OpenCV)
# BGR format: (Blue, Green, Red)
CLASS_COLORS = [
    (0, 200, 0),      # Green  for Low Stress
    (0, 165, 255),    # Orange for Moderate Stress
    (0, 0, 220)       # Red    for High Stress
]

# ── 2. Load the trained model ─────────────────────────────────
print("=" * 55)
print("PHASE 4 — SINGLE IMAGE STRESS PREDICTION")
print("=" * 55)

print("\nLoading trained model...")
model = load_model(MODEL_PATH)
print("  Model loaded successfully!")
print(f"  Input shape expected: {model.input_shape}")

# ── 3. Define prediction function ─────────────────────────────
def predict_stress(image_path):
    """
    Takes a path to an image file and returns the predicted stress level.

    Parameters:
        image_path (str): Full path to the image file

    Returns:
        label (str)      : Stress level name e.g. 'High Stress'
        confidence (float): How confident the model is (0.0 to 1.0)
        all_probs (array): Probabilities for all 3 classes
    """

    # Check if the image file actually exists
    if not os.path.exists(image_path):
        print(f"  ERROR: Image not found at {image_path}")
        return None, None, None

    # Read the image from disk using OpenCV
    # cv2.imread returns a numpy array of pixel values
    img = cv2.imread(image_path)

    if img is None:
        print(f"  ERROR: Could not read image at {image_path}")
        return None, None, None

    # Save original image for display later
    original_img = img.copy()

    # Resize image to 224x224 — same size used during training
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Convert from BGR (OpenCV default) to RGB
    # Model was trained on RGB images so we must match that
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to float and normalize pixels from 0-255 to 0.0-1.0
    # This must exactly match the preprocessing done in Phase 2
    img = img.astype(np.float32) / 255.0

    # Add batch dimension
    # Model expects input shape: (batch_size, 224, 224, 3)
    # Our single image shape is: (224, 224, 3)
    # np.expand_dims adds the batch dimension: (1, 224, 224, 3)
    img = np.expand_dims(img, axis=0)

    # Run prediction
    # model.predict returns array of probabilities for each class
    # Example: [[0.05, 0.15, 0.80]] means 80% chance of High Stress
    predictions = model.predict(img, verbose=0)

    # Get the probabilities for all 3 classes
    all_probs = predictions[0]

    # np.argmax finds the index with highest probability
    # Example: [0.05, 0.15, 0.80] → index 2 → High Stress
    predicted_index = np.argmax(all_probs)

    # Get the label name and confidence score
    label      = CLASS_NAMES[predicted_index]
    confidence = all_probs[predicted_index]

    return label, confidence, all_probs, original_img, predicted_index


# ── 4. Pick a test image from the dataset ─────────────────────
# We use one image from the test folder to verify prediction works
# You can change this path to any face image on your computer

# Try to find a test image automatically
test_image_path = None

test_dir = os.path.join("dataset", "raw", "test")

# Loop through emotion folders to find any image
for emotion in os.listdir(test_dir):
    emotion_folder = os.path.join(test_dir, emotion)
    if os.path.isdir(emotion_folder):
        images = os.listdir(emotion_folder)
        if images:
            # Pick the first image from this emotion folder
            test_image_path = os.path.join(emotion_folder, images[0])
            test_emotion    = emotion
            break

if test_image_path is None:
    print("No test image found. Please set test_image_path manually.")
else:
    print(f"\nTest image       : {test_image_path}")
    print(f"Actual emotion   : {test_emotion}")

    # ── 5. Run prediction ──────────────────────────────────────
    print("\nRunning prediction...")
    result = predict_stress(test_image_path)

    label, confidence, all_probs, original_img, pred_index = result

    if label is not None:

        # ── 6. Print results ───────────────────────────────────
        print("\n" + "=" * 55)
        print("PREDICTION RESULT")
        print("=" * 55)
        print(f"\n  {CLASS_EMOJI[pred_index]}  Predicted Stress Level : {label}")
        print(f"  Confidence             : {confidence * 100:.1f}%")
        print(f"\n  All probabilities:")
        for i, (name, prob) in enumerate(zip(CLASS_NAMES, all_probs)):
            bar = "█" * int(prob * 30)
            print(f"    {name:<18} : {prob * 100:5.1f}%  {bar}")

        print("\n" + "=" * 55)
        print("DISCLAIMER")
        print("=" * 55)
        print("  ⚠️  This is an estimated stress level based on")
        print("      facial expressions only.")
        print("      This is NOT a medical diagnosis.")
        print("      Please consult a healthcare professional")
        print("      for any health concerns.")
        print("=" * 55)

        # ── 7. Show the image with prediction label ────────────
        # Resize original image for display
        display_img = cv2.resize(original_img, (400, 400))

        # Get color for this stress level
        color = CLASS_COLORS[pred_index]

        # Draw a colored rectangle at the top of the image
        cv2.rectangle(display_img, (0, 0), (400, 60), color, -1)

        # Put the prediction text on the image
        cv2.putText(
            display_img,
            f"{label}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # Put the confidence text below
        cv2.putText(
            display_img,
            f"Confidence: {confidence * 100:.1f}%",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        # Show the image in a popup window
        cv2.imshow("Stress Prediction Result", display_img)
        print("\n  Image displayed. Press any key to close the window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()