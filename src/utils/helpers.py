"""
Helper functions for E-waste detection system
"""

import uuid
from typing import List
from src.config.settings import LOW_CONFIDENCE_THRESHOLD


def generate_unique_id() -> str:
    """Generate unique ID for detections"""
    return str(uuid.uuid4())


def generate_description(category: str, confidence: float) -> str:
    """Generate visual description for detected e-waste item (max 20 words in Indonesian)"""
    descriptions = {
        # Computing Devices
        "Smartphone": "Perangkat elektronik smartphone dengan layar sentuh",
        "Handphone": "Perangkat komunikasi genggam dengan layar sentuh dan tombol",
        "Music-Player": "Perangkat pemutar musik portable dengan tombol kontrol",
        "Laptop": "Komputer portable dengan layar lipat dan keyboard terpasang",
        "Komponen CPU": "Unit pemrosesan komputer dengan casing logam dan port",
        "Monitor": "Layar tampilan komputer dengan bezel hitam dan kabel",
        "Keyboard": "Papan ketik komputer dengan tombol huruf dan angka",
        "Mouse": "Perangkat penggerak kursor dengan tombol klik kiri-kanan",
        "Hardisk": "Media penyimpanan data dengan casing logam persegi",
        "Flashdisk": "Perangkat penyimpanan USB kecil portable dan ringan",
        
        # Home Appliances  
        "TV": "Televisi dengan layar besar dan remote control",
        "AC": "Unit pendingin ruangan dengan ventilasi dan kompresor",
        "Komponen Kulkas": "Bagian kulkas dengan kompressor dan pipa pendingin",
        "Microwave": "Oven microwave dengan pintu kaca dan tombol kontrol",
        "Mesin Cuci": "Mesin pencuci pakaian dengan drum dan panel kontrol",
        "Kipas": "Alat pendingin dengan baling-baling dan motor penggerak",
        
        # Office Equipment
        "Printer": "Mesin pencetak dokumen dengan tray kertas dan cartridge",
        "Speaker": "Perangkat audio dengan driver suara dan amplifier",
        "Camera": "Kamera digital dengan lensa dan layar LCD",
        
        # Power & Tools
        "Adaptor /Kilo": "Adaptor listrik dengan kabel dan colokan dinding",
        "Baterai Laptop": "Baterai portable persegi dengan terminal listrik",
        "Seterika": "Alat setrika listrik dengan permukaan logam panas",
        "Hair Dryer": "Pengering rambut dengan kipas dan elemen pemanas",
        
        # Lighting
        "Lampu": "Bohlam atau lampu LED dengan fitting standar",
        "Neon Box": "Papan neon dengan tabung cahaya fluorescent",
        
        # Networking & Accessories
        "Router": "Perangkat jaringan dengan antena dan port ethernet",
        "Remot": "Remote control dengan tombol angka dan navigasi",
        "Jam Tangan": "Jam tangan digital atau smartwatch dengan layar",
        
        # Kitchen Appliances
        "Oven": "Oven listrik dengan pintu dan rak pemanggang",
        "Kompor Listrik": "Kompor dengan elemen pemanas listrik dan kontrol",
        
        # Others
        "PS2": "Konsol game dengan controller dan kabel AV",
        "Senter": "Lampu senter portable dengan baterai dan lensa",
        "Solder": "Alat solder listrik dengan mata besi dan pegangan",
        "Vacum Cleaner": "Penyedot debu dengan selang dan kantong debu",
        "Telefon": "Telepon rumah dengan gagang dan tombol angka",
        "Panel Surya": "Panel fotovoltaik dengan sel surya dan frame",
        "Aki Motor": "Baterai motor dengan terminal plus-minus dan casing",
        "Alat Tensi": "Tensimeter digital dengan manset dan layar",
        "Alat Tes Vol": "Alat pengukur voltase dengan probe dan display"
    }
    
    base_desc = descriptions.get(category, f"Perangkat elektronik {category.lower()}")
    
    # Add confidence note if low
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return f"{base_desc} (perlu verifikasi)"
    
    return base_desc


def generate_suggestions(category: str) -> List[str]:
    """Generate disposal suggestions"""
    suggestions_map = {
        "Smartphone": [
            "Hapus semua data personal dan factory reset",
            "Lepas battery jika memungkinkan",
            "Bawa ke pusat daur ulang e-waste bersertifikat"
        ],
        "Laptop": [
            "Backup dan hapus semua data dengan secure wipe",
            "Lepas battery dan hard drive",
            "Donasi jika masih berfungsi atau ke e-waste center"
        ],
        "Desktop-PC": [
            "Remove dan destroy hard drive secara aman",
            "Pisahkan komponen logam dan plastik",
            "Bawa ke fasilitas daur ulang komputer"
        ],
        "Printer": [
            "Keluarkan semua cartridge tinta/toner",
            "Pisahkan tray kertas dan kabel",
            "Bawa ke pusat daur ulang office equipment"
        ],
        "Refrigerator": [
            "Hubungi teknisi untuk recovery refrigerant",
            "Kosongkan dan bersihkan interior",
            "Disposal melalui program appliance recycling"
        ]
    }
    
    return suggestions_map.get(category, [
        "Periksa panduan manufacturer untuk disposal",
        "Jangan buang ke tempat sampah rumah tangga",
        "Bawa ke pusat daur ulang e-waste terdekat"
    ])


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
