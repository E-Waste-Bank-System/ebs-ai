"""
Helper functions for E-waste detection system
"""

import uuid
from typing import List
from src.config.settings import LOW_CONFIDENCE_THRESHOLD


def generate_unique_id() -> str:
    """Generate unique ID for detections"""
    return str(uuid.uuid4())


def calculate_risk_level(category: str, confidence: float) -> int:
    """Calculate risk level 1-5"""
    base_risk = {
        "TV": 4, "Monitor": 4, "Refrigerator": 5, "Air-Conditioner": 5,
        "Smartphone": 4, "Laptop": 4, "Desktop-PC": 3, "Printer": 3,
        "Keyboard": 2, "Mouse": 2
    }
    
    risk = base_risk.get(category, 3)
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        risk = min(5, risk + 1)
    return risk
