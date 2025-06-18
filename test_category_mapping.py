#!/usr/bin/env python3
"""
Test script to demonstrate category mapping
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.mappings import get_mapped_category

def test_category_mapping():
    print("Category Mapping Test")
    print("=" * 50)
    print("YOLO Category -> Price Category (for pricing only)")
    print("-" * 50)
    
    test_categories = [
        "Fan", "Handphone", "Laptop", "Monitor", "Keyboard", 
        "Mouse", "Printer", "Speaker", "Television", "Router",
        "Electronic Component"  # This is from unknown class mapping
    ]
    
    for yolo_cat in test_categories:
        price_cat = get_mapped_category(yolo_cat)
        print(f"{yolo_cat:<20} -> {price_cat}")
    
    print("\n" + "=" * 50)
    print("Now in the API response:")
    print("- 'category' field = YOLO category (for display)")
    print("- Price calculation uses mapped category")
    print("- Risk/damage calculation uses mapped category")
    print("- Description uses YOLO category")

if __name__ == "__main__":
    test_category_mapping() 