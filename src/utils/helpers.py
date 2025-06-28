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
    """
    Calculate risk level 1-10 based on YOLO category and confidence
    Higher risk = more dangerous to environment/health
    """
    # Risk levels based on YOLO class names (1-5 base scale)
    # Categories with high environmental/health risks
    high_risk_categories = {
        "Television", "Fridge", "Microwave", "Washing Machine", 
        "Rice Cooker", "Iron"
    }
    
    # Categories with medium-high risks (batteries, screens, complex electronics)
    medium_high_risk_categories = {
        "Laptop", "Phone", "Monitor", "Battery", "Powerbank",
        "GPU", "Motherboard", "PC Case", "CPU Component"
    }
    
    # Categories with medium risks (general electronics)
    medium_risk_categories = {
        "Printer", "Speaker", "Router", "Solar Panel", "DVD Player",
        "Radio", "Microphone", "Harddisk", "Stick Ps"
    }
    
    # Categories with lower risks (peripherals, small devices)
    low_risk_categories = {
        "Keyboard", "Mouse", "Charger", "Electronic Socket", "Cables",
        "Calculator", "Clock", "Walkie Talkie", "Body Weight Scale", "Remote"
    }
    
    # Categories with minimal risks (simple devices)
    minimal_risk_categories = {
        "Fan", "Lamp", "Flashlight"
    }
    
    # Determine base risk level
    if category in high_risk_categories:
        base_risk = 5
    elif category in medium_high_risk_categories:
        base_risk = 4
    elif category in medium_risk_categories:
        base_risk = 3
    elif category in low_risk_categories:
        base_risk = 2
    elif category in minimal_risk_categories:
        base_risk = 1
    else:
        # Unknown category - assign medium risk
        base_risk = 3
    
    # Adjust based on confidence level
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        # Low confidence increases risk (uncertainty is risky)
        base_risk = min(5, base_risk + 1)
    elif confidence > 0.9:
        # Very high confidence slightly reduces risk
        base_risk = max(1, base_risk - 1)
    
    # Scale to 1-10 range
    scaled_risk = base_risk * 2
    return min(10, max(1, scaled_risk))



