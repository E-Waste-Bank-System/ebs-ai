#!/usr/bin/env python3
"""
Test script for new description function
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.absolute())
sys.path.append(project_root)

from src.utils.helpers import generate_description

def test_descriptions():
    """Test the new description function"""
    print("=== Testing New Visual Descriptions (Max 20 words in Indonesian) ===\n")
    
    test_categories = [
        'Printer', 'Handphone', 'Laptop', 'Monitor', 'TV',
        'AC', 'Keyboard', 'Mouse', 'Speaker', 'Camera'
    ]
    
    for category in test_categories:
        # Test high confidence
        desc_high = generate_description(category, 0.9)
        word_count_high = len(desc_high.split())
        
        # Test low confidence
        desc_low = generate_description(category, 0.4)
        word_count_low = len(desc_low.split())
        
        print(f"📱 {category}:")
        print(f"   High confidence: {desc_high} ({word_count_high} words)")
        print(f"   Low confidence:  {desc_low} ({word_count_low} words)")
        print()
    
    print("=== Example of Previous vs New Description ===")
    print("❌ OLD: 'Printer dengan cartridge tinta, pisahkan cartridge sebelum recycle (deteksi perlu verifikasi). Corrected from Kompor Listrik to Printer: The image clearly shows an HP DeskJet printer...'")
    print("✅ NEW Description: 'Mesin pencetak dokumen dengan tray kertas dan cartridge (perlu verifikasi)'")
    print("✅ NEW Validation Feedback: 'Corrected from Kompor Listrik to Printer: The image clearly shows an HP DeskJet printer.'")
    print("\n🎯 Benefits:")
    print("- Description: Visual appearance only (9 words)")
    print("- Validation feedback: Technical details separate")
    print("- Clean separation of concerns")

if __name__ == "__main__":
    test_descriptions()
