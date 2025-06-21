#!/usr/bin/env python3

import asyncio
import sys
import os
from PIL import Image

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from services.gemini_service import GeminiService

async def test_validation_improvements():
    """Test the improved validation system"""
    print("🔧 Testing Improved Validation System")
    print("=" * 50)
    
    service = GeminiService()
    
    if not service.is_available:
        print("❌ Gemini service not available")
        return
    
    # Create a test image
    test_image = "/tmp/test_validation.jpg"
    img = Image.new('RGB', (200, 200), color='blue')
    img.save(test_image)
    
    try:
        print("🧪 Test 1: Reasoning Extraction")
        print("-" * 30)
        
        # Simulate a response where Gemini mentions washing machine in reasoning
        test_response = '''{"is_valid_ewaste": true, "best_category": null, "reasoning": "I see a washing machine with control panel"}'''
        
        result = service._process_validation_response(
            test_response, "Printer", "Printer"
        )
        
        print(f"Input: YOLO='Printer', Reasoning='I see a washing machine'")
        print(f"Result: {result.final_category} ({result.detection_source})")
        print(f"Expected: Mesin Cuci (Reasoning Extract)")
        
        if result.final_category == "Mesin Cuci":
            print("✅ PASS - Correctly extracted 'Mesin Cuci' from reasoning")
        else:
            print("❌ FAIL - Did not extract correct category")
        
        print("\n🧪 Test 2: TV Detection")
        print("-" * 30)
        
        test_response2 = '''{"is_valid_ewaste": true, "best_category": null, "reasoning": "This looks like a television screen"}'''
        
        result2 = service._process_validation_response(
            test_response2, "Monitor", "Monitor"
        )
        
        print(f"Input: YOLO='Monitor', Reasoning='television screen'")
        print(f"Result: {result2.final_category} ({result2.detection_source})")
        
        if result2.final_category == "TV":
            print("✅ PASS - Correctly extracted 'TV' from reasoning")
        else:
            print("❌ FAIL - Did not extract correct category")
        
        print("\n🧪 Test 3: No Match Fallback")
        print("-" * 30)
        
        test_response3 = '''{"is_valid_ewaste": true, "best_category": null, "reasoning": "Some unknown device"}'''
        
        result3 = service._process_validation_response(
            test_response3, "Printer", "Printer"
        )
        
        print(f"Input: YOLO='Printer', Reasoning='unknown device'")
        print(f"Result: {result3.final_category} ({result3.detection_source})")
        
        if result3.final_category == "Printer" and result3.detection_source == "YOLO":
            print("✅ PASS - Correctly used fallback to YOLO prediction")
        else:
            print("❌ FAIL - Did not use correct fallback")
        
    finally:
        # Clean up
        if os.path.exists(test_image):
            os.remove(test_image)
    
    print("\n🎯 Summary:")
    print("- Enhanced validation prompt with full category list")
    print("- Reasoning extraction for null best_category responses")
    print("- Better debugging and logging")
    print("- Fallback to YOLO when no valid category found")

if __name__ == "__main__":
    asyncio.run(test_validation_improvements()) 