# Real-Time Facial Stress Estimation Using Deep Learning

## Project Overview
A web application that estimates stress levels from facial expressions
using deep learning. Built as a final year project.

## ⚠️ Disclaimer
This application provides an **estimated stress level** based on
facial expressions only. It is **NOT a medical diagnosis.**
Please consult a healthcare professional for any health concerns.

## Stress Levels
- 😊 **Low Stress** — Calm and relaxed
- 😐 **Moderate Stress** — Slightly stressed
- 😰 **High Stress** — Quite stressed

## Technology Stack
- Python 3.11
- TensorFlow / Keras (MobileNetV2)
- OpenCV
- MediaPipe
- Flask
- SQLite
- HTML / CSS / JavaScript / Bootstrap

## Dataset
- FER-2013 (Facial Expression Recognition)
- 35,887 face images
- 7 emotion classes mapped to 3 stress levels
- Source: Kaggle

## Model Performance
- Base Model: MobileNetV2 (Transfer Learning)
- Training Accuracy: ~76%
- Test Accuracy: 76.75%

## Project Structure