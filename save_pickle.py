# save_pickle.py
# Creates pickle files needed by the Flask web application
#
# What this script saves:
# 1. label_encoder.pkl  - converts numbers to stress label names
# 2. model_config.pkl   - stores all model configuration settings
# ============================================================

import pickle
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

print("=" * 55)
print("CREATING PICKLE FILES")
print("=" * 55)

# ── 1. Create Label Encoder ───────────────────────────────────
# LabelEncoder converts between text labels and numbers
# 0 = Low Stress
# 1 = Moderate Stress
# 2 = High Stress

print("\nCreating label encoder...")

# Create encoder object
label_encoder = LabelEncoder()

# Fit it with our 3 class names in correct order
# Order must match what we used during training
class_names = ["Low Stress", "Moderate Stress", "High Stress"]
label_encoder.fit(class_names)

# Test the encoder works correctly
print("\nTesting label encoder:")
print(f"  Text → Number:")
for name in class_names:
    number = label_encoder.transform([name])[0]
    print(f"    {name} → {number}")

print(f"\n  Number → Text:")
for number in [0, 1, 2]:
    name = label_encoder.inverse_transform([number])[0]
    print(f"    {number} → {name}")

# Save label encoder as pickle file
encoder_path = os.path.join("model", "label_encoder.pkl")
with open(encoder_path, 'wb') as f:
    # pickle.dump saves the object to the file
    # 'wb' means write in binary mode
    pickle.dump(label_encoder, f)

print(f"\n  ✓ label_encoder.pkl saved → model/label_encoder.pkl")

# ── 2. Create Model Config ────────────────────────────────────
# This dictionary stores all settings used during training
# So every part of the app uses exactly the same settings

print("\nCreating model config...")

model_config = {
    # Image size used during training
    "img_size"      : 224,

    # Number of stress classes
    "num_classes"   : 3,

    # Class names in order (index matches label number)
    "class_names"   : ["Low Stress", "Moderate Stress", "High Stress"],

    # Emoji for each class
    "class_emojis"  : ["😊", "😐", "😰"],

    # Colors for each class (hex codes for web)
    "class_colors"  : ["#28a745", "#ffc107", "#dc3545"],

    # Emotion to stress mapping used during preprocessing
    "emotion_map"   : {
        "happy"   : 0,
        "surprise": 0,
        "neutral" : 1,
        "disgust" : 1,
        "angry"   : 2,
        "fear"    : 2,
        "sad"     : 2
    },

    # Model file path
    "model_path"    : os.path.join("model", "stress_model.h5"),

    # Dataset used
    "dataset"       : "FER-2013",

    # Base model used
    "base_model"    : "MobileNetV2",

    # Project version
    "version"       : "1.0.0"
}

# Save model config as pickle file
config_path = os.path.join("model", "model_config.pkl")
with open(config_path, 'wb') as f:
    pickle.dump(model_config, f)

print(f"  ✓ model_config.pkl saved → model/model_config.pkl")

# ── 3. Verify both pickle files ───────────────────────────────
print("\n" + "=" * 55)
print("VERIFYING PICKLE FILES")
print("=" * 55)

# Load and verify label encoder
print("\nLoading label_encoder.pkl...")
with open(encoder_path, 'rb') as f:
    # pickle.load reads the object back from file
    # 'rb' means read in binary mode
    loaded_encoder = pickle.load(f)

print(f"  Classes in encoder: {list(loaded_encoder.classes_)}")
print(f"  ✓ Label encoder verified!")

# Load and verify model config
print("\nLoading model_config.pkl...")
with open(config_path, 'rb') as f:
    loaded_config = pickle.load(f)

print(f"  Image size    : {loaded_config['img_size']}")
print(f"  Num classes   : {loaded_config['num_classes']}")
print(f"  Class names   : {loaded_config['class_names']}")
print(f"  Base model    : {loaded_config['base_model']}")
print(f"  Version       : {loaded_config['version']}")
print(f"  ✓ Model config verified!")

# ── 4. Show final model folder contents ──────────────────────
print("\n" + "=" * 55)
print("MODEL FOLDER CONTENTS")
print("=" * 55)

model_files = os.listdir("model")
for f in sorted(model_files):
    size = os.path.getsize(os.path.join("model", f))
    # Convert bytes to MB
    size_mb = size / (1024 * 1024)
    print(f"  {f:<25} {size_mb:.2f} MB")

print("\n" + "=" * 55)
print("ALL PICKLE FILES CREATED SUCCESSFULLY!")
print("=" * 55)