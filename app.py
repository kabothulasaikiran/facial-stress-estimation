# app.py
# Phase 6 - Main Flask Web Application
#
# This file runs the entire website.
# It handles all pages and connects
# the webcam, model, and recommendations together.
# ============================================================

from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import sys
import base64
import sqlite3
from datetime import datetime
# Load pickle files at startup
import pickle

# Load model config
with open(os.path.join("model", "model_config.pkl"), 'rb') as f:
    config = pickle.load(f)

# Load label encoder
with open(os.path.join("model", "label_encoder.pkl"), 'rb') as f:
    label_encoder = pickle.load(f)

# Use config values everywhere
IMG_SIZE    = config["img_size"]
CLASS_NAMES = config["class_names"]

# Import our utility functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.recommendations import get_recommendations
from utils.database import init_db, save_prediction, get_all_predictions

# ── 1. Initialize Flask app ───────────────────────────────────
app = Flask(__name__)

# ── 2. Load model once at startup ────────────────────────────
print("Loading stress detection model...")
MODEL_PATH  = os.path.join("model", "stress_model.h5")
IMG_SIZE    = 224
CLASS_NAMES = ["Low Stress", "Moderate Stress", "High Stress"]
model       = load_model(MODEL_PATH)
print("Model loaded successfully!")

# ── 3. Load face detector ─────────────────────────────────────
# Download cascade if not exists
import urllib.request
cascade_path = "haarcascade_frontalface_default.xml"
if not os.path.exists(cascade_path):
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, cascade_path)
face_cascade = cv2.CascadeClassifier(cascade_path)

# ── 4. Initialize database ────────────────────────────────────
init_db()

# ── 5. Global webcam variable ─────────────────────────────────
# We keep one camera object open while streaming
camera = None

# ── 6. Prediction function ────────────────────────────────────
def predict_stress(face_img):
    """
    Takes a face image array and returns stress prediction.
    """
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

# ── 7. Video stream generator ─────────────────────────────────
def generate_frames():
    """
    Captures frames from webcam and yields them as JPEG bytes.
    This powers the live video feed on the camera page.
    """
    global camera
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Flip frame like a mirror
        frame = cv2.flip(frame, 1)

        # Detect face and draw box
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1,
            minNeighbors=5, minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 0), 2)
            cv2.putText(frame, "Face Detected",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 200, 0), 2)

        # Instruction text on frame
        cv2.putText(frame,
                    "Position your face in the box then click Capture",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield as multipart response for browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'
               + frame_bytes + b'\r\n')

# ── 8. Routes (Pages) ─────────────────────────────────────────

# HOME PAGE
@app.route('/')
def index():
    """
    Home page — shows welcome message and Start button.
    """
    return render_template('index.html')


# CAMERA PAGE
@app.route('/camera')
def camera_page():
    """
    Camera page — shows live webcam feed and Capture button.
    """
    return render_template('camera.html')


# VIDEO FEED
@app.route('/video_feed')
def video_feed():
    """
    Streams live webcam video to the camera page.
    The HTML img tag src points to this URL.
    """
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# CAPTURE AND PREDICT
@app.route('/capture', methods=['POST'])
def capture():
    """
    Called when user clicks Capture button.
    Takes current frame, runs prediction, saves to database,
    redirects to result page.
    """
    global camera

    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Read current frame
    success, frame = camera.read()

    if not success:
        return jsonify({"error": "Could not capture frame"}), 500

    frame = cv2.flip(frame, 1)

    # Detect face
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1,
        minNeighbors=5, minSize=(80, 80)
    )

    if len(faces) == 0:
        # No face found — send error back to page
        return jsonify({"error": "No face detected. Please center your face."}), 400

    # Use largest face
    faces = sorted(faces, key=lambda f: f[2]*f[3])
    x, y, w, h = faces[-1]
    face_crop   = frame[y:y+h, x:x+w]

    # Run prediction
    label, confidence, all_probs = predict_stress(face_crop)

    # Get wellness recommendations
    rec = get_recommendations(label)

    # Save captured image as base64 to show on result page
    _, buffer    = cv2.imencode('.jpg', frame)
    img_base64   = base64.b64encode(buffer).decode('utf-8')

    # Save prediction to database
    save_prediction(label, confidence)

    # Release camera after capture
    if camera:
        camera.release()
        camera = None

    # Store result in session via URL parameters
    return jsonify({
        "success"    : True,
        "label"      : label,
        "confidence" : round(confidence, 1),
        "all_probs"  : {
            "Low Stress"      : round(all_probs[0], 1),
            "Moderate Stress" : round(all_probs[1], 1),
            "High Stress"     : round(all_probs[2], 1)
        },
        "message"    : rec["message"],
        "tips"       : rec["tips"],
        "image"      : img_base64
    })


# RESULT PAGE
@app.route('/result')
def result():
    """
    Result page — shows stress level and wellness tips.
    Data is passed via URL query parameters.
    """
    return render_template('result.html')


# HISTORY PAGE
@app.route('/history')
def history():
    """
    History page — shows all past predictions from database.
    """
    predictions = get_all_predictions()
    return render_template('history.html', predictions=predictions)


# HISTORY DATA API
@app.route('/history_data')
def history_data():
    """
    Returns prediction history as JSON for charts.
    """
    predictions = get_all_predictions()
    return jsonify(predictions)


# STOP CAMERA
@app.route('/stop_camera')
def stop_camera():
    """
    Releases the webcam when user leaves camera page.
    """
    global camera
    if camera:
        camera.release()
        camera = None
    return jsonify({"status": "Camera stopped"})


# ── 9. Run the app ────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port,
            debug=False, use_reloader=False)