"""
Category mappings for YOLO classes to price model categories
"""

# YOLO class names (37 classes, indices 0-36)
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

# Map YOLO class names to Price model categories
YOLO_TO_PRICE_MAP = {
   "Battery": "Baterai Laptop",
   "Body Weight Scale": "Alat Tensi",
   "CPU Component": "Komponen CPU",
   "Cables": "Adaptor /Kilo",
   "Calculator": "Alat Tes Vol",
   "Charger": "Adaptor /Kilo",
   "Clock": "Jam Tangan",
   "DVD Player": "TV",
   "Electronic Socket": "Adaptor /Kilo",
   "Fan": "Kipas",
   "Flashlight": "Senter",
   "Fridge": "Komponen Kulkas",
   "GPU": "Komponen CPU",
   "Harddisk": "Hardisk",
   "Iron": "Seterika",
   "Keyboard": "Keyboard",
   "Lamp": "Lampu",
   "Laptop": "Laptop",
   "Microphone": "Speaker",
   "Microwave": "Microwave",
   "Monitor": "Monitor",
   "Motherboard": "Komponen CPU",
   "Mouse": "Mouse",
   "PC Case": "CPU Intel",
   "Phone": "Handphone",
   "Powerbank": "Baterai Laptop",
   "Printer": "Printer",
   "Radio": "Speaker",
   "Remote": "Remot",
   "Rice Cooker": "Kompor Listrik",
   "Router": "Router",
   "Solar Panel": "Panel Surya",
   "Speaker": "Speaker",
   "Stick Ps": "PS2",
   "Television": "TV",
   "Walkie Talkie": "Telefon",
   "Washing Machine": "Mesin Cuci"
}

# Supported price categories - derived from YOLO_TO_PRICE_MAP values
PRICE_CATEGORIES = set(YOLO_TO_PRICE_MAP.values())


def get_mapped_category(yolo_class_name: str) -> str:
    """Get price category for YOLO class name"""
    return YOLO_TO_PRICE_MAP.get(yolo_class_name, "Handphone")  # Default fallback


def is_valid_price_category(category: str) -> bool:
    """Check if category is supported by price model"""
    return category in PRICE_CATEGORIES


def get_class_name_for_index(class_idx: int) -> str:
    """
    Get class name for YOLO class index
    Returns the class name or a fallback for invalid indices
    """
    if class_idx in CLASS_NAMES:
        return CLASS_NAMES[class_idx]
    
    # For indices outside our range (0-36), return a generic fallback
    return f"Unknown Device {class_idx}"


def get_all_yolo_classes() -> list:
    """Get all available YOLO class names"""
    return list(CLASS_NAMES.values())


def get_all_price_categories() -> list:
    """Get all available price categories"""
    return list(PRICE_CATEGORIES)
