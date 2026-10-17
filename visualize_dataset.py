# visualize_dataset.py
# This script creates 2 charts:
# Chart 1 — bar chart showing image count per emotion
# Chart 2 — pie chart showing stress level distribution

import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Path to training data
TRAIN_DIR = os.path.join("dataset", "raw", "train")

# Emotion to stress level mapping
STRESS_MAP = {
    "happy"   : "Low",
    "surprise": "Low",
    "neutral" : "Moderate",
    "disgust" : "Moderate",
    "angry"   : "High",
    "fear"    : "High",
    "sad"     : "High"
}

# Colors for each stress level
COLORS = {
    "Low"      : "#2ecc71",
    "Moderate" : "#f39c12",
    "High"     : "#e74c3c"
}

# Count images per emotion class
emotions = sorted(os.listdir(TRAIN_DIR))
counts   = []
colors   = []

for emotion in emotions:
    folder = os.path.join(TRAIN_DIR, emotion)
    if os.path.isdir(folder):
        count = len(os.listdir(folder))
        counts.append(count)
        stress = STRESS_MAP.get(emotion, "Low")
        colors.append(COLORS[stress])

# ── Chart 1: Bar chart ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(emotions, counts, color=colors, edgecolor='black', linewidth=0.5)

# Show number on top of each bar
for bar, count in zip(bars, counts):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 50,
        str(count),
        ha='center', fontsize=9, fontweight='bold'
    )

ax.set_title("Training Images per Emotion Class", fontsize=13, fontweight='bold')
ax.set_xlabel("Emotion")
ax.set_ylabel("Number of Images")

# Legend
legend = [
    Patch(facecolor='#2ecc71', label='Low Stress'),
    Patch(facecolor='#f39c12', label='Moderate Stress'),
    Patch(facecolor='#e74c3c', label='High Stress')
]
ax.legend(handles=legend)
plt.tight_layout()

# Save the chart
os.makedirs(os.path.join("static", "images"), exist_ok=True)
plt.savefig(os.path.join("static", "images", "class_distribution.png"), dpi=150)
print("Chart 1 saved → static/images/class_distribution.png")
plt.show()

# ── Chart 2: Pie chart ────────────────────────────────────────
stress_totals = {"Low": 0, "Moderate": 0, "High": 0}
for emotion, count in zip(emotions, counts):
    stress = STRESS_MAP.get(emotion, "Low")
    stress_totals[stress] += count

fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.pie(
    stress_totals.values(),
    labels=[f"{k} Stress" for k in stress_totals],
    colors=[COLORS[k] for k in stress_totals],
    autopct='%1.1f%%',
    startangle=140
)
ax2.set_title("Stress Level Distribution", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join("static", "images", "stress_distribution.png"), dpi=150)
print("Chart 2 saved → static/images/stress_distribution.png")
plt.show()

print("\nVisualization complete!")