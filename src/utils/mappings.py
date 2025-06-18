"""
Category mappings for YOLO classes to price model categories
"""

# YOLO class names (77 classes)
CLASS_NAMES = {
   0:'Battery', 1:'Body Weight Scale', 2:'Calculator', 3:'Clock', 4:'DVD Player', 5:'DVD ROM', 6:'Electronic Socket', 7:'Fan', 8:'Flashlight', 9:'Fridge', 10:'GPU', 11:'Handphone', 12:'Harddisk', 13:'Insect Killer', 14:'Iron', 15:'Keyboard', 16:'Lamp', 17:'Laptop', 18:'Laptop Charger', 19:'Microphone', 20:'Microwave', 21:'Monitor', 22:'Motherboard', 23:'Mouse', 24:'PC Case', 25:'Power Supply', 26:'Powerbank', 27:'Printer', 28:'Printer Ink', 29:'Radio', 30:'Rice Cooker', 31:'Router', 32:'Solar Panel', 33:'Speaker', 34:'Television', 35:'Toaster', 36:'Walkie Talkie', 37:'Washing Machine'
}

# Map YOLO class names (38 classes) to Price model categories
YOLO_TO_PRICE_MAP = {
   "Battery": "Baterai Laptop",
   "Body Weight Scale": "Alat Tensi",
   "Calculator": "Alat Tes Vol",
   "Clock": "Jam Tangan",
   "DVD Player": "TV",
   "DVD ROM": "TV",
   "Electronic Socket": "Adaptor /Kilo",
   "Fan": "Kipas",
   "Flashlight": "Senter",
   "Fridge": "Komponen Kulkas",
   "GPU": "Komponen CPU",
   "Handphone": "Handphone",
   "Harddisk": "Hardisk",
   "Insect Killer": "TV",
   "Iron": "Seterika",
   "Keyboard": "Keyboard",
   "Lamp": "Lampu",
   "Laptop": "Laptop",
   "Laptop Charger": "Adaptor /Kilo",
   "Microphone": "Speaker",
   "Microwave": "Microwave",
   "Monitor": "Monitor",
   "Motherboard": "Komponen CPU",
   "Mouse": "Mouse",
   "PC Case": "CPU Intel",
   "Power Supply": "Komponen CPU",
   "Powerbank": "Baterai Laptop",
   "Printer": "Printer",
   "Printer Ink": "Printer",
   "Radio": "Speaker",
   "Rice Cooker": "Kompor Listrik",
   "Router": "Router",
   "Solar Panel": "Panel Surya",
   "Speaker": "Speaker",
   "Television": "TV",
   "Toaster": "Oven",
   "Walkie Talkie": "Telefon",
   "Washing Machine": "Mesin Cuci"
}

# Supported price categories (38 categories)
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
    # First check if it's a known class
    if yolo_class_name in YOLO_TO_PRICE_MAP:
        return YOLO_TO_PRICE_MAP[yolo_class_name]
    
    # If it's an unknown class, map it to a price category
    if (yolo_class_name.startswith("Unknown Device") or 
        yolo_class_name not in YOLO_TO_PRICE_MAP):
        return map_unknown_to_price_category(yolo_class_name)
    
    # Fallback to the original name
    return yolo_class_name

def is_valid_price_category(category: str) -> bool:
    """Check if category is supported by price model"""
    return category in PRICE_CATEGORIES

def get_class_name_for_index(class_idx: int) -> str:
    """
    Get class name for YOLO class index, with fallback for unknown indices
    """
    if class_idx in CLASS_NAMES:
        return CLASS_NAMES[class_idx]
    
    # Handle unknown classes by mapping to generic categories
    # This is a temporary solution until we get the complete class mapping
    unknown_class_mapping = {
        # Common indices that might be missing
        38: "Electronic Device",  # Generic electronic device
        39: "Small Appliance",    # Generic small appliance
        40: "Computer Component", # Generic computer part
        41: "Audio Device",       # Generic audio equipment
        42: "Kitchen Appliance",  # Generic kitchen item
        43: "Tool",              # Generic tool
        44: "Charger",           # Generic charger
        45: "Cable",             # Generic cable
        46: "Remote Control",    # Generic remote
        47: "Gaming Device",     # Generic gaming equipment
        48: "Sensor",            # Generic sensor
        49: "Motor",             # Generic motor
        50: "Display",           # Generic display
        51: "Electronic Component", # Generic electronic component
        52: "Power Device",      # Generic power-related device
    }
    
    if class_idx in unknown_class_mapping:
        return unknown_class_mapping[class_idx]
    
    # For any other unknown index, return a generic name
    return f"Unknown Device {class_idx}"

def map_unknown_to_price_category(unknown_class_name: str) -> str:
    """
    Map unknown class names to appropriate price categories
    """
    unknown_to_price_mapping = {
        "Electronic Device": "Handphone",        # Generic electronics
        "Small Appliance": "Kompor Listrik",     # Small appliances
        "Computer Component": "Komponen CPU",     # Computer parts
        "Audio Device": "Speaker",               # Audio equipment
        "Kitchen Appliance": "Microwave",        # Kitchen items
        "Tool": "Solder",                        # Tools
        "Charger": "Adaptor /Kilo",             # Chargers
        "Cable": "Adaptor /Kilo",               # Cables
        "Remote Control": "Remot",               # Remotes
        "Gaming Device": "PS2",                  # Gaming
        "Sensor": "Alat Tes Vol",               # Sensors
        "Motor": "Kipas",                        # Motors
        "Display": "Monitor",                    # Displays
        "Electronic Component": "Komponen CPU",   # Electronic parts
        "Power Device": "Adaptor /Kilo",        # Power devices
    }
    
    # Check if it's a pattern like "Unknown Device 51"
    if unknown_class_name.startswith("Unknown Device"):
        return "Handphone"  # Default to generic electronics
    
    return unknown_to_price_mapping.get(unknown_class_name, "Handphone")
