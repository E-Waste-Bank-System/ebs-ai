"""
Category mappings for YOLO classes to price model categories

Pipeline: YOLO Detection → Gemini Validation → YOLO-to-Price Mapping → Price Prediction
This module handles stage 3: Maps validated YOLO categories (37) to price categories (33)

The mapping ensures that validated YOLO predictions can be used for price prediction.
"""

import logging

logger = logging.getLogger(__name__)

# YOLO class names (37 classes, indices 0-36)
# These are the categories that YOLO can detect
CLASS_NAMES = {
   0: 'Battery', 1: 'Body Weight Scale', 2: 'CPU Component', 3: 'Cables', 
   4: 'Calculator', 5: 'Charger', 6: 'Clock', 7: 'DVD Player', 
   8: 'Electronic Socket', 9: 'Fan', 10: 'Flashlight', 11: 'Fridge', 
   12: 'GPU', 13: 'Harddisk', 14: 'Iron', 15: 'Keyboard', 
   16: 'Lamp', 17: 'Laptop', 18: 'Microphone', 19: 'Microwave', 
   20: 'Monitor', 21: 'Motherboard', 22: 'Mouse', 23: 'PC Case', 
   24: 'Phone', 25: 'Powerbank', 26: 'Printer', 27: 'Radio', 
   28: 'Remote', 29: 'Rice Cooker', 30: 'Router', 31: 'Solar Panel', 
   32: 'Speaker', 33: 'Stick Ps', 34: 'Television', 35: 'Walkie Talkie', 
   36: 'Washing Machine'
}

# Map YOLO class names (37 categories) to Price model categories (33 categories)
# This mapping happens AFTER Gemini validation in the pipeline
YOLO_TO_PRICE_MAP = {
   # Electronics and Devices
   "Battery": "Baterai Laptop",
   "Body Weight Scale": "Alat Tensi",
   "CPU Component": "Komponen CPU",
   "Cables": "Adaptor /Kilo",
   "Calculator": "Alat Tes Vol",
   "Charger": "Charger Laptop",
   "Clock": "Jam Dinding",
   "DVD Player": "DVD Player",
   "Electronic Socket": "Adaptor /Kilo",
   
   # Appliances
   "Fan": "Kipas",
   "Flashlight": "Senter",
   "Fridge": "Komponen Kulkas",
   "Iron": "Seterika",
   "Lamp": "Lampu",
   "Microwave": "Microwave",
   "Rice Cooker": "Magicom",
   "Washing Machine": "Mesin Cuci",
   
   # Computing Devices
   "GPU": "Komponen CPU",
   "Harddisk": "Hardisk",
   "Keyboard": "Keyboard",
   "Laptop": "Laptop",
   "Monitor": "Monitor",
   "Motherboard": "Motherboard",
   "Mouse": "Mouse",
   "PC Case": "Komponen CPU",
   "Printer": "Printer",
   "Router": "Router",
   
   # Communication and Entertainment
   "Microphone": "Microfon",
   "Phone": "Handphone",           # Smartphones/mobile phones
   "Powerbank": "Power Bank",
   "Radio": "Radio",
   "Remote": "Remot",
   "Speaker": "Speaker",
   "Stick Ps": "Stik Ps",             # Gaming controllers
   "Television": "TV",
   "Walkie Talkie": "Walkie Talkie",    # Two-way radios
   
   # Specialized Equipment
   "Solar Panel": "Panel Surya"
}

# Supported price categories - derived from YOLO_TO_PRICE_MAP values (33 unique categories)
PRICE_CATEGORIES = set(YOLO_TO_PRICE_MAP.values())


def get_mapped_category(yolo_category: str) -> str:
    """
    Get price model category for validated YOLO category (Stage 3 of pipeline)
    
    Args:
        yolo_category: Validated YOLO category name (37 classes)
        
    Returns:
        Price model category name (33 classes)
    """
    mapped_category = YOLO_TO_PRICE_MAP.get(yolo_category, "Handphone")  # Default fallback
    if mapped_category != YOLO_TO_PRICE_MAP.get(yolo_category):
        logger.warning(f"Unknown YOLO category '{yolo_category}', using fallback: {mapped_category}")
    return mapped_category


def is_valid_price_category(category: str) -> bool:
    """
    Check if category is supported by price model
    
    Args:
        category: Category name to validate
        
    Returns:
        True if category is in the 33 price categories
    """
    return category in PRICE_CATEGORIES


def get_class_name_for_index(class_idx: int) -> str:
    """
    Get YOLO class name for class index
    
    Args:
        class_idx: YOLO class index (0-36)
        
    Returns:
        YOLO class name or fallback for invalid indices
    """
    if class_idx in CLASS_NAMES:
        return CLASS_NAMES[class_idx]
    
    # For indices outside our range (0-36), return a generic fallback
    logger.warning(f"Unknown YOLO class index: {class_idx}")
    return f"Unknown Device {class_idx}"


def get_all_yolo_classes() -> list:
    """Get all available YOLO class names (37 categories)"""
    return list(CLASS_NAMES.values())


def get_all_price_categories() -> list:
    """Get all available price categories (33 categories)"""
    return sorted(list(PRICE_CATEGORIES))


def get_mapping_info() -> dict:
    """
    Get information about the YOLO to Price mapping
    
    Returns:
        Dictionary with mapping statistics and information
    """
    return {
        "yolo_categories": len(CLASS_NAMES),
        "price_categories": len(PRICE_CATEGORIES),
        "mapping_count": len(YOLO_TO_PRICE_MAP),
        "pipeline_stage": "3 - Category Mapping",
        "input": "Validated YOLO categories (37 classes)",
        "output": "Price model categories (33 classes)",
        "note": "Happens after Gemini validation, before price prediction"
    }
