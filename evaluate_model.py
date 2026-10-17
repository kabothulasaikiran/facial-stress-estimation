# evaluate_model.py
# Phase 3 - Evaluate the trained model
#
# What this script does:
# 1. Loads the saved model (stress_model.h5)
# 2. Draws accuracy and loss graphs
# 3. Draws a confusion matrix
# 4. Shows classification report

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report

# ── 1. Configuration ──────────────────────────────────────────
CLASS_NAMES = ["Low Stress", "Moderate Stress", "High Stress"]
MODEL_PATH  = os.path.join("model", "stress_model.h5")
DATA_DIR    = os.path.join("dataset", "processed")
NUM_CLASSES = 3

# ── 2. Load test data ─────────────────────────────────────────
print("=" * 55)
print("PHASE 3 — MODEL EVALUATION")
print("=" * 55)

print("\nLoading test data...")
X_test     = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test     = np.load(os.path.join(DATA_DIR, "y_test.npy"))
y_test_cat = to_categorical(y_test, NUM_CLASSES)
print(f"  X_test shape : {X_test.shape}")

# ── 3. Load the saved model ───────────────────────────────────
print("\nLoading saved model...")
model = load_model(MODEL_PATH)
print("  Model loaded successfully!")

# ── 4. Load training history ──────────────────────────────────
print("\nLoading training history...")
history = np.load(
    os.path.join("model", "history.npy"),
    allow_pickle=True
).item()

# ── 5. Plot accuracy and loss graphs ─────────────────────────
print("\nDrawing accuracy and loss graphs...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy graph
axes[0].plot(history['accuracy'],
             label='Training Accuracy',
             color='#2ecc71', linewidth=2)
axes[0].plot(history['val_accuracy'],
             label='Validation Accuracy',
             color='#3498db', linewidth=2)
axes[0].set_title('Model Accuracy over Epochs',
                  fontsize=13, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss graph
axes[1].plot(history['loss'],
             label='Training Loss',
             color='#e74c3c', linewidth=2)
axes[1].plot(history['val_loss'],
             label='Validation Loss',
             color='#e67e22', linewidth=2)
axes[1].set_title('Model Loss over Epochs',
                  fontsize=13, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join("static", "images", "accuracy_loss.png"), dpi=150)
print("  Accuracy graph saved → static/images/accuracy_loss.png")
plt.show()

# ── 6. Make predictions ───────────────────────────────────────
print("\nMaking predictions on test data...")
y_pred_probs = model.predict(X_test, verbose=1)
y_pred       = np.argmax(y_pred_probs, axis=1)

# ── 7. Draw confusion matrix ──────────────────────────────────
print("\nDrawing confusion matrix...")

cm = confusion_matrix(y_test, y_pred)

fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    linewidths=0.5
)
ax2.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
ax2.set_ylabel('Actual Stress Level',    fontsize=11)
ax2.set_xlabel('Predicted Stress Level', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join("static", "images", "confusion_matrix.png"), dpi=150)
print("  Confusion matrix saved → static/images/confusion_matrix.png")
plt.show()

# ── 8. Print classification report ───────────────────────────
print("\n" + "=" * 55)
print("CLASSIFICATION REPORT")
print("=" * 55)
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

# ── 9. Final accuracy ─────────────────────────────────────────
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print("=" * 55)
print(f"  Final Test Accuracy : {test_acc * 100:.2f}%")
print(f"  Final Test Loss     : {test_loss:.4f}")
print("=" * 55)
print("\nEvaluation complete!")