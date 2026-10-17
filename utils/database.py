# utils/database.py
# Phase 7 - SQLite database for storing prediction history
#
# What this file does:
# 1. Creates a SQLite database file
# 2. Saves each prediction with timestamp
# 3. Retrieves all past predictions
# ============================================================

import sqlite3
import os
from datetime import datetime

# Database file will be created in project root
DB_PATH = "stress_history.db"

def init_db():
    """
    Creates the database and predictions table if they don't exist.
    Called once when Flask app starts.
    """
    # connect() creates the file if it doesn't exist
    conn = sqlite3.connect(DB_PATH)

    # cursor lets us run SQL commands
    cursor = conn.cursor()

    # CREATE TABLE IF NOT EXISTS means only create if not already there
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            stress     TEXT    NOT NULL,
            confidence REAL    NOT NULL,
            timestamp  TEXT    NOT NULL
        )
    ''')

    # Save changes and close connection
    conn.commit()
    conn.close()
    print("  Database initialized!")


def save_prediction(stress_level, confidence):
    """
    Saves one prediction result to the database.

    Parameters:
        stress_level (str): 'Low Stress', 'Moderate Stress', or 'High Stress'
        confidence  (float): confidence percentage 0-100
    """
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get current date and time as string
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # INSERT INTO adds a new row to the table
    cursor.execute('''
        INSERT INTO predictions (stress, confidence, timestamp)
        VALUES (?, ?, ?)
    ''', (stress_level, round(confidence, 1), timestamp))

    conn.commit()
    conn.close()


def get_all_predictions():
    """
    Returns all predictions from database as a list of dicts.
    Most recent predictions come first.
    """
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # ORDER BY id DESC means newest first
    cursor.execute('''
        SELECT id, stress, confidence, timestamp
        FROM predictions
        ORDER BY id DESC
    ''')

    rows = cursor.fetchall()
    conn.close()

    # Convert each row tuple to a dictionary
    predictions = []
    for row in rows:
        predictions.append({
            "id"        : row[0],
            "stress"    : row[1],
            "confidence": row[2],
            "timestamp" : row[3]
        })

    return predictions