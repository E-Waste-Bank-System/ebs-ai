"""
Category mappings for YOLO classes to price model categories
"""

# YOLO class names (77 classes)
CLASS_NAMES = {
    0: "Air-Conditioner", 1: "Bar-Phone", 2: "Battery", 3: "Blood-Pressure-Monitor",
    4: "Boiler", 5: "CRT-Monitor", 6: "CRT-TV", 7: "Calculator", 8: "Camera",
    9: "Ceiling-Fan", 10: "Christmas-Lights", 11: "Clothes-Iron", 12: "Coffee-Machine",
    13: "Compact-Fluorescent-Lamps", 14: "Computer-Keyboard", 15: "Computer-Mouse",
    16: "Cooled-Dispenser", 17: "Cooling-Display", 18: "Dehumidifier", 19: "Desktop-PC",
    20: "Digital-Oscilloscope", 21: "Dishwasher", 22: "Drone", 23: "Electric-Bicycle",
    24: "Electric-Guitar", 25: "Electrocardiograph-Machine", 26: "Electronic-Keyboard",
    27: "Exhaust-Fan", 28: "Flashlight", 29: "Flat-Panel-Monitor", 30: "Flat-Panel-TV",
    31: "Floor-Fan", 32: "Freezer", 33: "Glucose-Meter", 34: "HDD", 35: "Hair-Dryer",
    36: "Headphone", 37: "LED-Bulb", 38: "Laptop", 39: "Microwave", 40: "Music-Player",
    41: "Neon-Sign", 42: "Network-Switch", 43: "Non-Cooled-Dispenser", 44: "Oven",
    45: "PCB", 46: "Patient-Monitoring-System", 47: "Photovoltaic-Panel", 48: "PlayStation-5",
    49: "Power-Adapter", 50: "Printer", 51: "Projector", 52: "Pulse-Oximeter",
    53: "Range-Hood", 54: "Refrigerator", 55: "Rotary-Mower", 56: "Router", 57: "SSD",
    58: "Server", 59: "Smart-Watch", 60: "Smartphone", 61: "Smoke-Detector",
    62: "Soldering-Iron", 63: "Speaker", 64: "Stove", 65: "Straight-Tube-Fluorescent-Lamp",
    66: "Street-Lamp", 67: "TV-Remote-Control", 68: "Table-Lamp", 69: "Tablet",
    70: "Telephone-Set", 71: "Toaster", 72: "Tumble-Dryer", 73: "USB-Flash-Drive",
    74: "Vacuum-Cleaner", 75: "Washing-Machine", 76: "Xbox-Series-X"
}

# Map YOLO class names (77 classes) to Price model categories (33 categories)
YOLO_TO_PRICE_MAP = {
    # Computing Devices
    "Computer-Keyboard": "Keyboard",
    "Electronic-Keyboard": "Keyboard", 
    "Computer-Mouse": "Mouse",
    "Desktop-PC": "Komponen CPU",
    "Server": "Komponen CPU",
    "PCB": "Komponen CPU",
    "HDD": "Hardisk",
    "SSD": "Hardisk",
    "USB-Flash-Drive": "Flashdisk",
    "Laptop": "Laptop",

    # Display Devices
    "Flat-Panel-Monitor": "Monitor",
    "CRT-Monitor": "Monitor",
    "Digital-Oscilloscope": "Monitor",
    "Patient-Monitoring-System": "Monitor",
    "Projector": "Monitor",
    "Flat-Panel-TV": "TV",
    "CRT-TV": "TV",
    "TV-Remote-Control": "Remot",
    
    # Mobile Devices
    "Smartphone": "Handphone",
    "Bar-Phone": "Handphone",
    "Smart-Watch": "Jam Tangan",
    "Tablet": "Handphone",
    "Camera": "Camera",
    "PlayStation-5": "PS2",
    "Xbox-Series-X": "PS2",
    
    # Audio Devices
    "Speaker": "Speaker",
    "Headphone": "Speaker",
    "Music-Player": "Speaker",
    "Electric-Guitar": "Speaker",
    
    # Power Devices
    "Power-Adapter": "Adaptor /Kilo",
    "Battery": "Baterai Laptop",
    
    # Kitchen Devices
    "Microwave": "Microwave",
    "Coffee-Machine": "Oven",
    "Oven": "Oven",
    "Stove": "Kompor Listrik",
    "Toaster": "Oven",
    
    # Cooling Devices
    "Refrigerator": "Komponen Kulkas",
    "Freezer": "Komponen Kulkas",
    "Cooled-Dispenser": "Komponen Kulkas",
    "Non-Cooled-Dispenser": "Komponen Kulkas",
    "Cooling-Display": "Komponen Kulkas",

    # Home Devices
    "Clothes-Iron": "Seterika",
    "Boiler": "Kompor Listrik",  # Critical: Maps Boiler to correct category, not Printer
    "Hair-Dryer": "Hair Dryer",
    "Rotary-Mower": "Kipas",
    "Soldering-Iron": "Solder",
    "Vacuum-Cleaner": "Vacum Cleaner",
    "Washing-Machine": "Mesin Cuci",
    "Dishwasher": "Mesin Cuci",
    "Tumble-Dryer": "Mesin Cuci",

    # Air Control
    "Ceiling-Fan": "Kipas",
    "Floor-Fan": "Kipas",
    "Exhaust-Fan": "Kipas",
    "Range-Hood": "Kipas",
    "Air-Conditioner": "AC",
    "Dehumidifier": "AC",

    # Office Equipment
    "Printer": "Printer",
    "Calculator": "Alat Tes Vol",

    # Networking
    "Router": "Router",
    "Network-Switch": "Router",

    # Lighting
    "LED-Bulb": "Lampu",
    "Table-Lamp": "Lampu",
    "Straight-Tube-Fluorescent-Lamp": "Lampu",
    "Compact-Fluorescent-Lamps": "Lampu",
    "Christmas-Lights": "Lampu",
    "Neon-Sign": "Neon Box",
    "Street-Lamp": "Lampu",

    # Health Devices
    "Blood-Pressure-Monitor": "Alat Tensi",
    "Electrocardiograph-Machine": "Monitor",
    "Glucose-Meter": "Alat Tes Vol",
    "Pulse-Oximeter": "Alat Tes Vol",

    # Vehicle & Others
    "Drone": "Kipas",
    "Electric-Bicycle": "Aki Motor",
    "Photovoltaic-Panel": "Panel Surya",
    "Telephone-Set": "Telefon",
    "Flashlight": "Senter",
    "Smoke-Detector": "Monitor"
}

# Supported price categories (33 categories)
PRICE_CATEGORIES = {
    "AC", "Adaptor /Kilo", "Aki Motor", "Alat Tensi", "Alat Tes Vol", 
    "Baterai Laptop", "Camera", "CPU Intel", "Flashdisk", "Hair Dryer",
    "Handphone", "Hardisk", "Jam Tangan", "Keyboard", "Kipas",
    "Komponen CPU", "Komponen Kulkas", "Kompor Listrik", "Lampu", "Laptop",
    "Mesin Cuci", "Microwave", "Monitor", "Mouse", "Neon Box",
    "Oven", "Panel Surya", "Printer", "PS2", "Remot",
    "Router", "Senter", "Seterika", "Solder", "Speaker", "Telefon", 
    "TV", "Vacum Cleaner"
}

def get_mapped_category(yolo_class_name: str) -> str:
    """Get price category for YOLO class name"""
    return YOLO_TO_PRICE_MAP.get(yolo_class_name, yolo_class_name)

def is_valid_price_category(category: str) -> bool:
    """Check if category is supported by price model"""
    return category in PRICE_CATEGORIES
