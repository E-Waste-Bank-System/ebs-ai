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
    Calculate risk level 1-10 based on category and confidence
    Higher risk = more dangerous to environment/health
    """
    # Base risk levels for different categories (1-5 scale, then doubled for 1-10)
    base_risk = {
        # High risk - large appliances with refrigerants/hazardous materials
        "TV": 5, "Komponen Kulkas": 5, "AC": 5, "Mesin Cuci": 4,
        
        # Medium-high risk - electronics with batteries/screens
        "Laptop": 4, "Handphone": 4, "Monitor": 4, "Microwave": 4,
        
        # Medium risk - electronics with some hazardous components
        "Printer": 3, "CPU Intel": 3, "Komponen CPU": 3, "Speaker": 3,
        "Router": 3, "Panel Surya": 3, "Camera": 3,
        
        # Lower risk - smaller electronics
        "Keyboard": 2, "Mouse": 2, "Hardisk": 2, "Baterai Laptop": 3,
        "Adaptor /Kilo": 2, "Flashdisk": 2, "Remot": 2,
        
        # Variable risk based on type
        "Lampu": 2, "Kipas": 2, "Senter": 2, "Jam Tangan": 2,
        "Seterika": 3, "Hair Dryer": 3, "Kompor Listrik": 4,
        "Oven": 4, "Solder": 3, "Alat Tensi": 2, "Alat Tes Vol": 2,
        "PS2": 3, "Telefon": 2, "Vacum Cleaner": 3, "Neon Box": 3,
        "Aki Motor": 4
    }
    
    # Get base risk (default to 3 for unknown categories)
    risk = base_risk.get(category, 3)
    
    # Adjust based on confidence
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        risk = min(5, risk + 1)  # Increase risk if low confidence
    elif confidence > 0.9:
        risk = max(1, risk - 1)  # Decrease risk if very high confidence
    
    # Scale to 1-10 and ensure bounds
    scaled_risk = risk * 2
    return min(10, max(1, scaled_risk))


def calculate_base_damage_level(category: str) -> int:
    """
    Calculate base damage level expectation for different categories (1-10 scale)
    This represents typical wear/damage for each category type
    """
    base_damage = {
        # Categories that typically show more wear
        "Laptop": 6, "Handphone": 5, "Keyboard": 4, "Mouse": 4,
        "Seterika": 7, "Hair Dryer": 6, "Kompor Listrik": 6,
        
        # Categories that are often well-preserved
        "TV": 4, "Monitor": 4, "Printer": 5, "CPU Intel": 3,
        "Speaker": 4, "Router": 3, "Camera": 5,
        
        # Categories with variable condition
        "Adaptor /Kilo": 5, "Baterai Laptop": 6, "Hardisk": 4,
        "Lampu": 3, "Kipas": 5, "Senter": 4, "Jam Tangan": 4,
        
        # Heavy-duty items that might show wear
        "Mesin Cuci": 7, "Microwave": 5, "Oven": 6, "Vacum Cleaner": 6,
        "Komponen Kulkas": 5, "AC": 6, "Panel Surya": 3,
        
        # Small components
        "Komponen CPU": 4, "Flashdisk": 3, "Remot": 4, "Telefon": 5,
        "Alat Tensi": 4, "Alat Tes Vol": 4, "PS2": 5, "Solder": 5,
        "Neon Box": 4, "Aki Motor": 7
    }
    
    return base_damage.get(category, 5)  # Default to middle range
