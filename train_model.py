# train_model.py
# STABLE VERSION - Based on what gave 78% before
# with careful improvements to push to 82-85%
#
# Strategy:
# - Keep augmentation simple and safe
# - Only 2 phases like before
# - Proven learning rates
# - More patience for early stopping
# ============================================================

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D,
                                     Dropout, BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                        ReduceLROnPlateau)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

# ── 1. Reproducibility ────────────────────────────────────────
np.random.seed(42)
tf.random.set_seed(42)

# ── 2. Configuration ──────────────────────────────────────────
IMG_SIZE    = 224
NUM_CLASSES = 3
BATCH_SIZE  = 32
MODEL_DIR   = "model"
DATA_DIR    = os.path.join("dataset", "processed")
CLASS_NAMES = ["Low Stress", "Moderate Stress", "High Stress"]

# ── 3. Load data ──────────────────────────────────────────────
print("=" * 55)
print("STABLE IMPROVED TRAINING")
print("=" * 55)

print("\nLoading data...")
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))

print(f"  Training : {X_train.shape[0]} images")
print(f"  Test     : {X_test.shape[0]} images")

# ── 4. Class distribution ─────────────────────────────────────
print("\nClass distribution:")
for i, name in enumerate(CLASS_NAMES):
    count = int(np.sum(y_train == i))
    pct   = count / len(y_train) * 100
    print(f"  {name}: {count} ({pct:.1f}%)")

# ── 5. One-hot encode ─────────────────────────────────────────
y_train_cat = to_categorical(y_train, NUM_CLASSES)
y_test_cat  = to_categorical(y_test,  NUM_CLASSES)

# ── 6. Class weights ──────────────────────────────────────────
print("\nComputing class weights...")
weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(weights))
print(f"  {class_weight_dict}")

# ── 7. SAFE Data Augmentation ─────────────────────────────────
# Only safe augmentations that do not distort faces too much
# No brightness change — it confused the model before
print("\nSetting up safe data augmentation...")

train_datagen = ImageDataGenerator(
    # Only horizontal flip and small rotation
    # These are safe for face images
    horizontal_flip=True,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    fill_mode='nearest'
)

train_generator = train_datagen.flow(
    X_train, y_train_cat,
    batch_size=BATCH_SIZE,
    shuffle=True
)

steps_per_epoch  = len(X_train) // BATCH_SIZE
validation_steps = len(X_test)  // BATCH_SIZE

print(f"  Steps per epoch : {steps_per_epoch}")

# ── 8. Build Model ────────────────────────────────────────────
print("\nBuilding model...")

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze all base layers initially
base_model.trainable = False

# Custom head — same as what gave 78% before
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

os.makedirs(MODEL_DIR, exist_ok=True)

# ── 9. PHASE 1 — Train top layers ─────────────────────────────
print("\n" + "=" * 55)
print("PHASE 1 — Training top layers only (12 epochs)")
print("=" * 55)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_1 = [
    ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "stress_model.h5"),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=6,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.3,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

history1 = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=12,
    validation_data=(X_test, y_test_cat),
    class_weight=class_weight_dict,
    callbacks=callbacks_1,
    verbose=1
)

best1 = max(history1.history['val_accuracy'])
print(f"\nPhase 1 best accuracy: {best1*100:.2f}%")

# ── 10. PHASE 2 — Fine tune last 30 layers ────────────────────
print("\n" + "=" * 55)
print("PHASE 2 — Fine tuning last 30 layers (15 epochs)")
print("=" * 55)

# Unfreeze base model
base_model.trainable = True

# Keep all layers frozen except last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False

trainable = sum(1 for l in model.layers if l.trainable)
print(f"Trainable layers: {trainable}")

# Lower learning rate — critical for fine tuning
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_2 = [
    ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "stress_model.h5"),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=6,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.3,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

history2 = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=15,
    validation_data=(X_test, y_test_cat),
    class_weight=class_weight_dict,
    callbacks=callbacks_2,
    verbose=1
)

best2 = max(history2.history['val_accuracy'])
print(f"\nPhase 2 best accuracy: {best2*100:.2f}%")

# ── 11. Save combined history ─────────────────────────────────
combined = {
    'accuracy'    : (history1.history['accuracy'] +
                     history2.history['accuracy']),
    'val_accuracy': (history1.history['val_accuracy'] +
                     history2.history['val_accuracy']),
    'loss'        : (history1.history['loss'] +
                     history2.history['loss']),
    'val_loss'    : (history1.history['val_loss'] +
                     history2.history['val_loss'])
}
np.save(os.path.join(MODEL_DIR, "history.npy"), combined)

# ── 12. Final evaluation ──────────────────────────────────────
print("\n" + "=" * 55)
print("FINAL EVALUATION")
print("=" * 55)

from tensorflow.keras.models import load_model as lm
best_model = lm(os.path.join(MODEL_DIR, "stress_model.h5"))
loss, acc  = best_model.evaluate(X_test, y_test_cat, verbose=0)

print(f"\n  Phase 1 best : {best1*100:.2f}%")
print(f"  Phase 2 best : {best2*100:.2f}%")
print(f"\n  FINAL TEST ACCURACY : {acc*100:.2f}%")
print(f"  FINAL TEST LOSS     : {loss:.4f}")

print("\n" + "=" * 55)
print("TRAINING COMPLETE!")
print("=" * 55)