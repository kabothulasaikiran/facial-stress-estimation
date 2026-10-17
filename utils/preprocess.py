# utils/preprocess.py
# This script:
# 1. Reads every image from train and test folders
# 2. Resizes each image to 224x224 pixels
# 3. Converts emotion labels to stress labels (0, 1, 2)
# 4. Normalizes pixel values from 0-255 to 0.0-1.0
# 5. Saves everything as 4 files in dataset/processed/

import os
import cv2
import numpy as np

# Image size required by MobileNetV2
IMG_SIZE = 224

# Folder paths
TRAIN_DIR  = os.path.join("dataset", "raw", "train")
TEST_DIR   = os.path.join("dataset", "raw", "test")
OUTPUT_DIR = os.path.join("dataset", "processed")

# Emotion to stress number mapping
# 0 = Low   1 = Moderate   2 = High
STRESS_MAP = {
    "happy"   : 0,
    "surprise": 0,
    "neutral" : 1,
    "disgust" : 1,
    "angry"   : 2,
    "fear"    : 2,
    "sad"     : 2
}

CLASS_NAMES = ["Low", "Moderate", "High"]


def load_images(data_dir):
    images = []
    labels = []

    for emotion in sorted(os.listdir(data_dir)):
        folder = os.path.join(data_dir, emotion)

        # Skip if not a folder
        if not os.path.isdir(folder):
            continue

        # Skip unknown emotions
        if emotion not in STRESS_MAP:
            continue

        label = STRESS_MAP[emotion]
        print(f"  Loading {emotion} → {CLASS_NAMES[label]} Stress...")

        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)

            # Read image from disk
            img = cv2.imread(img_path)

            # Skip broken images
            if img is None:
                continue

            # Resize to 224x224
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            images.append(img)
            labels.append(label)

    # Convert lists to numpy arrays
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    # Normalize pixels from 0-255 to 0.0-1.0
    images = images / 255.0

    return images, labels


# ── Main pipeline ─────────────────────────────────────────────
print("=" * 50)
print("PREPROCESSING STARTED")
print("This will take 10 to 20 minutes. Please wait.")
print("=" * 50)

# Load training images
print("\nLoading training images...")
X_train, y_train = load_images(TRAIN_DIR)
print(f"\n  Training shape : {X_train.shape}")

# Load test images
print("\nLoading test images...")
X_test, y_test = load_images(TEST_DIR)
print(f"\n  Test shape     : {X_test.shape}")

# Show class counts
print("\nClass distribution in training set:")
for i, name in enumerate(CLASS_NAMES):
    count = int(np.sum(y_train == i))
    print(f"  {name} Stress : {count} images")

# Save all 4 files
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("\nSaving files to dataset/processed/ ...")
np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "X_test.npy"),  X_test)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"),  y_test)

print("  ✓ X_train.npy saved")
print("  ✓ y_train.npy saved")
print("  ✓ X_test.npy  saved")
print("  ✓ y_test.npy  saved")

print("\n" + "=" * 50)
print("PREPROCESSING COMPLETE!")
print("=" * 50)