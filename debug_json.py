#!/usr/bin/env python3
"""
Debug script to test JSON parsing improvements
"""

import json
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent)
sys.path.append(project_root)

def test_json_extraction(response_text):
    """Test the improved JSON extraction logic"""
    print(f"🔍 Testing JSON extraction for: {response_text[:100]}...")
    
    # More robust response cleaning (same as in gemini_service.py)
    cleaned_text = response_text.strip()
    
    # Remove any text before the first { and after the last }
    start_idx = cleaned_text.find('{')
    end_idx = cleaned_text.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        cleaned_text = cleaned_text[start_idx:end_idx + 1]
    else:
        # If no valid JSON brackets found, try to extract from markdown
        if '```json' in cleaned_text:
            start = cleaned_text.find('```json') + 7
            end = cleaned_text.find('```', start)
            if end != -1:
                cleaned_text = cleaned_text[start:end].strip()
        elif '```' in cleaned_text:
            start = cleaned_text.find('```') + 3
            end = cleaned_text.find('```', start)
            if end != -1:
                cleaned_text = cleaned_text[start:end].strip()
    
    print(f"✂️  Cleaned text: {cleaned_text}")
    
    try:
        result = json.loads(cleaned_text)
        print(f"✅ Successfully parsed JSON: {result}")
        return result
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing failed: {e}")
        return None

if __name__ == "__main__":
    print("🧪 JSON Extraction Test")
    print("=" * 50)
    
    # Test case 1: The problematic response from the logs
    test_case_1 = """YOLO: Speaker

```json
{
  "is_valid_ewaste": true,
  "best_category": "Speaker",
  "reasoning": "The object is identified as a speaker, which is an electronic device and therefore considered e-waste when discarded."
}
```"""
    
    print("🧪 Test Case 1: Response with prefix text")
    result1 = test_json_extraction(test_case_1)
    
    print("\n" + "-" * 30 + "\n")
    
    # Test case 2: Clean JSON
    test_case_2 = """{
  "is_valid_ewaste": true,
  "best_category": "Speaker",
  "reasoning": "Valid e-waste"
}"""
    
    print("🧪 Test Case 2: Clean JSON")
    result2 = test_json_extraction(test_case_2)
    
    print("\n" + "-" * 30 + "\n")
    
    # Test case 3: Markdown wrapped
    test_case_3 = """```json
{
  "damage_level": 5,
  "analysis": "Moderate wear visible"
}
```"""
    
    print("🧪 Test Case 3: Markdown wrapped JSON")
    result3 = test_json_extraction(test_case_3)
    
    print("\n" + "=" * 50)
    print("🎯 Test Results:")
    print(f"   Test 1: {'✅ PASS' if result1 else '❌ FAIL'}")
    print(f"   Test 2: {'✅ PASS' if result2 else '❌ FAIL'}")
    print(f"   Test 3: {'✅ PASS' if result3 else '❌ FAIL'}")
    
    if all([result1, result2, result3]):
        print("\n🎉 All tests passed! JSON extraction is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the extraction logic.") 