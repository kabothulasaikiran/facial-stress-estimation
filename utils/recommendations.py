# utils/recommendations.py
# Phase 8 - Wellness recommendations based on stress level
#
# IMPORTANT DISCLAIMER:
# These are general wellness suggestions only.
# They are NOT medical advice or diagnosis.
# Always consult a healthcare professional for health concerns.
# ============================================================

# ── Wellness suggestions for each stress level ────────────────
RECOMMENDATIONS = {

    "Low Stress": {
        "message"  : "You seem calm and relaxed. Keep it up!",
        "emoji"    : "😊",
        "color"    : (0, 200, 0),   # Green
        "tips"     : [
            "Stay hydrated — drink a glass of water",
            "Take a short walk to maintain energy",
            "Maintain your sleep schedule (7-8 hours)",
            "Keep up your positive routine",
            "Practice gratitude — note 3 good things today"
        ]
    },

    "Moderate Stress": {
        "message"  : "You seem a little stressed. Take a short break.",
        "emoji"    : "😐",
        "color"    : (0, 165, 255),  # Orange
        "tips"     : [
            "Try deep breathing: inhale 4s, hold 4s, exhale 4s",
            "Do a 5 minute stretching exercise",
            "Take a 10 minute break from your screen",
            "Drink water or herbal tea",
            "Listen to calming music for a few minutes"
        ]
    },

    "High Stress": {
        "message"  : "You seem quite stressed. Please take care of yourself.",
        "emoji"    : "😰",
        "color"    : (0, 0, 220),    # Red
        "tips"     : [
            "Try box breathing: inhale 4s, hold 4s, exhale 4s, hold 4s",
            "Do 10 minutes of meditation or mindfulness",
            "Try gentle yoga or stretching for 10 minutes",
            "Reduce screen time — take a proper break",
            "Talk to a friend or family member",
            "Go outside for fresh air and a short walk",
            "Avoid caffeine — drink water instead"
        ]
    }
}

def get_recommendations(stress_level):
    """
    Returns wellness recommendations for a given stress level.

    Parameters:
        stress_level (str): 'Low Stress', 'Moderate Stress', or 'High Stress'

    Returns:
        dict: message, emoji, color, and list of tips
    """
    # Return recommendations for this stress level
    # If stress level not found, return Moderate as default
    return RECOMMENDATIONS.get(stress_level, RECOMMENDATIONS["Moderate Stress"])


def get_tips_as_text(stress_level):
    """
    Returns tips as a single formatted string.
    Useful for displaying in terminal or simple text areas.
    """
    rec   = get_recommendations(stress_level)
    lines = []
    lines.append(f"\n{rec['emoji']}  {rec['message']}")
    lines.append("\nWellness Suggestions (Not medical advice):")
    lines.append("-" * 40)
    for i, tip in enumerate(rec["tips"], 1):
        lines.append(f"  {i}. {tip}")
    lines.append("-" * 40)
    lines.append("⚠️  These are general wellness suggestions only.")
    lines.append("    Please consult a doctor for medical advice.")
    return "\n".join(lines)