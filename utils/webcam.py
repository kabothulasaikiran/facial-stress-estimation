# utils/webcam.py
# Phase 5 - Real time webcam with face tracking
#
# Features:
# 1. Detects multiple faces simultaneously
# 2. Assigns unique ID to each face
# 3. Tracks same face across frames
# 4. Shows separate stress prediction per face
# 5. Shows wellness suggestions for each face
# ============================================================

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.recommendations import get_recommendations
from utils.face_tracker import FaceTracker

# ── 1. Configuration ──────────────────────────────────────────
MODEL_PATH  = os.path.join("model", "stress_model.h5")
IMG_SIZE    = 224
CLASS_NAMES = ["Low Stress", "Moderate Stress", "High Stress"]

CLASS_COLORS = {
    "Low Stress"      : (0, 200, 0),
    "Moderate Stress" : (0, 165, 255),
    "High Stress"     : (0, 0, 220),
    "Detecting..."    : (200, 200, 200)
}

PREDICT_EVERY_N_FRAMES = 8

# ── 2. Load model ─────────────────────────────────────────────
print("=" * 55)
print("REAL TIME STRESS DETECTION WITH FACE TRACKING")
print("=" * 55)
print("\nLoading model...")
model = load_model(MODEL_PATH)
print("  Model loaded!")

# ── 3. Load face detector ─────────────────────────────────────
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
print("  Face detector loaded!")

# ── 4. Initialize face tracker ────────────────────────────────
# max_disappeared=25: face can be missing 25 frames before ID removed
# max_distance=100: centroids must be within 100px to match same face
tracker = FaceTracker(max_disappeared=25, max_distance=100)
print("  Face tracker initialized!")

# ── 5. Unique colors for each face ID ─────────────────────────
# Each face ID gets its own box color for easy identification
ID_COLORS = [
    (255, 100, 100),   # ID 0 - Blue
    (100, 255, 100),   # ID 1 - Green
    (100, 100, 255),   # ID 2 - Red
    (255, 255, 100),   # ID 3 - Cyan
    (255, 100, 255),   # ID 4 - Magenta
    (100, 255, 255),   # ID 5 - Yellow
]

def get_id_color(face_id):
    """Returns a consistent color for a given face ID."""
    return ID_COLORS[face_id % len(ID_COLORS)]


# ── 6. Prediction function ────────────────────────────────────
def predict_from_face(face_img):
    """Predicts stress level from cropped face image."""
    if face_img is None or face_img.size == 0:
        return "Detecting...", 0.0, [0.33, 0.33, 0.34]

    img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    preds      = model.predict(img, verbose=0)
    idx        = np.argmax(preds[0])
    label      = CLASS_NAMES[idx]
    confidence = float(preds[0][idx]) * 100
    all_probs  = [float(p) * 100 for p in preds[0]]

    return label, confidence, all_probs


# ── 7. Draw face info on frame ────────────────────────────────
def draw_face_info(frame, face_id, bbox, prediction):
    """
    Draws bounding box, face ID, stress label and
    confidence for one face.
    """
    x, y, w, h = bbox
    label       = prediction["label"]
    confidence  = prediction["confidence"]
    id_color    = get_id_color(face_id)
    stress_color = CLASS_COLORS.get(label, (200, 200, 200))

    # Draw face bounding box with unique ID color
    cv2.rectangle(frame, (x, y), (x+w, y+h), id_color, 2)

    # Draw corner accents for style
    corner_len = 15
    corner_thickness = 3

    # Top-left corner
    cv2.line(frame, (x, y), (x+corner_len, y), id_color, corner_thickness)
    cv2.line(frame, (x, y), (x, y+corner_len), id_color, corner_thickness)

    # Top-right corner
    cv2.line(frame, (x+w, y), (x+w-corner_len, y), id_color, corner_thickness)
    cv2.line(frame, (x+w, y), (x+w, y+corner_len), id_color, corner_thickness)

    # Bottom-left corner
    cv2.line(frame, (x, y+h), (x+corner_len, y+h), id_color, corner_thickness)
    cv2.line(frame, (x, y+h), (x, y+h-corner_len), id_color, corner_thickness)

    # Bottom-right corner
    cv2.line(frame, (x+w, y+h), (x+w-corner_len, y+h), id_color, corner_thickness)
    cv2.line(frame, (x+w, y+h), (x+w, y+h-corner_len), id_color, corner_thickness)

    # Label text above face box
    label_text = f"ID:{face_id} | {label} {confidence:.0f}%"
    (tw, th), _ = cv2.getTextSize(
        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)

    # Background for label
    cv2.rectangle(frame,
                  (x, y - th - 12),
                  (x + tw + 8, y),
                  id_color, -1)

    # Label text
    cv2.putText(frame, label_text,
                (x + 4, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 2)

    return frame


# ── 8. Draw suggestions panel ─────────────────────────────────
def draw_suggestions_panel(frame, tracked_predictions):
    """
    Draws wellness suggestions for all tracked faces
    on the right side of the frame.
    """
    fh, fw  = frame.shape[:2]
    panel_x = fw - 300
    panel_y = 10

    if not tracked_predictions:
        return frame

    # Dark background for entire panel
    overlay = frame.copy()
    panel_h = min(30 + len(tracked_predictions) * 130, fh - 20)
    cv2.rectangle(overlay,
                  (panel_x - 8, panel_y),
                  (fw - 5, panel_y + panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    # Title
    cv2.putText(frame, "Wellness Suggestions",
                (panel_x, panel_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1)

    current_y = panel_y + 40

    for face_id, pred in tracked_predictions.items():
        label    = pred["label"]
        id_color = get_id_color(face_id)

        if label == "Detecting..." or current_y > fh - 100:
            continue

        rec  = get_recommendations(label)
        tips = rec["tips"][:3]  # Show only top 3 tips per face

        # Face ID header
        cv2.putText(frame,
                    f"Face ID {face_id} — {label}",
                    (panel_x, current_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, id_color, 1)
        current_y += 18

        # Divider line
        cv2.line(frame,
                 (panel_x, current_y),
                 (fw - 10, current_y),
                 id_color, 1)
        current_y += 12

        # Tips
        for tip in tips:
            if current_y > fh - 30:
                break

            # Bullet
            cv2.circle(frame, (panel_x + 5, current_y - 4),
                       3, id_color, -1)

            # Tip text — truncate if too long
            tip_text = tip[:38] + ".." if len(tip) > 38 else tip
            cv2.putText(frame, tip_text,
                        (panel_x + 14, current_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (200, 200, 200), 1)
            current_y += 18

        current_y += 10

    # Disclaimer
    cv2.putText(frame,
                "Not medical advice.",
                (panel_x, fh - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (120, 120, 120), 1)

    return frame


# ── 9. Main webcam loop ───────────────────────────────────────
def run_webcam():
    """
    Opens webcam, tracks faces with unique IDs,
    and shows stress predictions per face.
    Press Q to quit.
    """
    print("\nOpening webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("  ERROR: Could not open webcam!")
        return

    print("  Webcam opened!")
    print("  Multiple faces will each get their own ID.")
    print("  Press Q to quit.\n")

    frame_count = 0

    # Create and bring window to front
    cv2.namedWindow("Stress Detection — Multi Face Tracking",
                    cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Stress Detection — Multi Face Tracking",
                     960, 540)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]

        # ── Detect faces every N frames ────────────────────────
        if frame_count % PREDICT_EVERY_N_FRAMES == 0:

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect all faces in frame
            # scaleFactor=1.05 is more sensitive than 1.1
            # minNeighbors=4 catches more faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=4,
                minSize=(60, 60)
            )

            # Convert to list of tuples
            detections = [(x, y, w, h) for (x, y, w, h) in faces] \
                         if len(faces) > 0 else []

            # Update tracker with new detections
            tracker.update(detections)

            # Run prediction for each tracked face
            for face_id, bbox in tracker.bboxes.items():
                x, y, w, h = bbox

                # Safely crop face
                y1 = max(0, y)
                y2 = min(fh, y + h)
                x1 = max(0, x)
                x2 = min(fw, x + w)

                face_crop = frame[y1:y2, x1:x2]

                if face_crop.size > 0:
                    label, confidence, all_probs = \
                        predict_from_face(face_crop)

                    # Update prediction for this face ID
                    tracker.update_prediction(
                        face_id, label, confidence, all_probs)

                    # Print to terminal
                    if frame_count % 40 == 0:
                        print(f"  Face ID {face_id}: "
                              f"{label} ({confidence:.1f}%)")

        # ── Draw info for each tracked face ────────────────────
        for face_id, bbox in tracker.bboxes.items():
            pred = tracker.predictions.get(face_id, {
                "label"     : "Detecting...",
                "confidence": 0.0,
                "all_probs" : [33, 33, 34]
            })
            frame = draw_face_info(frame, face_id, bbox, pred)

        # ── Draw suggestions panel ─────────────────────────────
        frame = draw_suggestions_panel(frame, tracker.predictions)

        # ── Face count top left ────────────────────────────────
        face_count = len(tracker.bboxes)
        cv2.putText(frame,
                    f"Faces detected: {face_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)

        # ── Disclaimer ─────────────────────────────────────────
        cv2.putText(frame,
                    "Estimated stress only. Not a medical diagnosis.",
                    (10, fh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (150, 150, 150), 1)

        # ── Quit instruction ───────────────────────────────────
        cv2.putText(frame, "Press Q to quit",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1)

        # ── Show frame ─────────────────────────────────────────
        cv2.imshow("Stress Detection — Multi Face Tracking", frame)

        # Bring to front on first frame
        if frame_count == 1:
            cv2.setWindowProperty(
                "Stress Detection — Multi Face Tracking",
                cv2.WND_PROP_TOPMOST, 1)

        # Quit on Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n  Q pressed. Stopping.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("  Webcam closed.")
    print(f"  Total unique faces tracked: {tracker.next_id}")


# ── 10. Run directly ──────────────────────────────────────────
if __name__ == "__main__":
    run_webcam()