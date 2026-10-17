# explore_dataset.py
# This script counts how many images are in each emotion folder
# and shows us how emotions map to stress levels

import os

# Path to training images
TRAIN_DIR = os.path.join("dataset", "raw", "train")
TEST_DIR  = os.path.join("dataset", "raw", "test")

# This maps each emotion to a stress level
STRESS_MAP = {
    "happy"   : "Low Stress",
    "surprise": "Low Stress",
    "neutral" : "Moderate Stress",
    "disgust" : "Moderate Stress",
    "angry"   : "High Stress",
    "fear"    : "High Stress",
    "sad"     : "High Stress"
}

print("=" * 50)
print("DATASET EXPLORATION REPORT")
print("=" * 50)

# Count images in training set
print("\nTRAINING SET:")
total_train = 0
for emotion in sorted(os.listdir(TRAIN_DIR)):
    folder = os.path.join(TRAIN_DIR, emotion)
    if os.path.isdir(folder):
        count = len(os.listdir(folder))
        stress = STRESS_MAP.get(emotion, "Unknown")
        print(f"  {emotion:<12} : {count} images  →  {stress}")
        total_train += count
print(f"\n  Total training images: {total_train}")

# Count images in test set
print("\nTEST SET:")
total_test = 0
for emotion in sorted(os.listdir(TEST_DIR)):
    folder = os.path.join(TEST_DIR, emotion)
    if os.path.isdir(folder):
        count = len(os.listdir(folder))
        print(f"  {emotion:<12} : {count} images")
        total_test += count
print(f"\n  Total test images: {total_test}")
print(f"  Grand total     : {total_train + total_test}")

print("\n" + "=" * 50)
print("Exploration complete!")
print("=" * 50)