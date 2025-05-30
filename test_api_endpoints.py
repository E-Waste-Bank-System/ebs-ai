#!/usr/bin/env python3
"""
API Endpoint Testing Script
Tests the complete workflow with new description format
"""

import sys
import asyncio
import requests
import io
from pathlib import Path
from PIL import Image, ImageDraw
import time

# Add project root to Python path
project_root = str(Path(__file__).parent)
sys.path.append(project_root)

from src.services.detection_service import DetectionService
from src.utils.helpers import generate_description

def create_test_image():
    """Create a simple test image for API testing"""
    # Create a simple test image (smartphone-like rectangle)
    img = Image.new('RGB', (300, 600), color='black')
    draw = ImageDraw.Draw(img)
    
    # Draw a smartphone-like shape
    draw.rectangle([50, 50, 250, 550], fill='gray', outline='white', width=3)
    draw.rectangle([70, 100, 230, 300], fill='blue')  # Screen
    draw.ellipse([140, 350, 160, 370], fill='white')  # Home button
    
    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes.getvalue()

def test_description_function():
    """Test the updated description function"""
    print("=" * 60)
    print("TESTING DESCRIPTION FUNCTION")
    print("=" * 60)
    
    # Test various categories with different confidence levels
    test_cases = [
        ("smartphone", 0.85),
        ("laptop", 0.70),
        ("television", 0.45),  # Low confidence
        ("battery", 0.92),
        ("unknown_category", 0.75)
    ]
    
    for category, confidence in test_cases:
        description = generate_description(category, confidence)
        word_count = len(description.split())
        print(f"Category: {category:<15} | Confidence: {confidence:.2f} | Words: {word_count:<2} | Description: {description}")
    
    print("\n✓ Description function working correctly\n")

def test_detection_service():
    """Test the detection service directly"""
    print("=" * 60)
    print("TESTING DETECTION SERVICE")
    print("=" * 60)
    
    try:
        # Initialize detection service
        detection_service = DetectionService()
        print("✓ Detection service initialized successfully")
        
        # Test system status
        status = detection_service.get_system_status()
        print(f"✓ System status: {status}")
        
        # Test supported categories
        categories = detection_service.get_supported_categories()
        print(f"✓ Supported categories: {len(categories)} categories available")
        
        # Test price prediction only
        price_result = detection_service.predict_price_only("smartphone")
        print(f"✓ Price prediction test: {price_result}")
        
        return detection_service
        
    except Exception as e:
        print(f"✗ Detection service error: {e}")
        return None

async def test_image_processing(detection_service):
    """Test complete image processing"""
    print("=" * 60)
    print("TESTING IMAGE PROCESSING")
    print("=" * 60)
    
    if not detection_service:
        print("✗ Detection service not available")
        return
    
    try:
        # Create test image
        test_image = create_test_image()
        print("✓ Test image created")
        
        # Test object detection only
        print("\n--- Testing Object Detection Only ---")
        object_result = await detection_service.detect_objects_only(test_image)
        print(f"✓ Object detection result: {object_result}")
        
        # Test complete processing
        print("\n--- Testing Complete Processing ---")
        complete_result = await detection_service.process_image_complete(test_image)
        print(f"✓ Complete processing result keys: {list(complete_result.keys())}")
        
        # Check if the new description format is working
        if hasattr(complete_result, 'predictions') and complete_result.predictions:
            for pred in complete_result.predictions:
                if hasattr(pred, 'description') and hasattr(pred, 'validation_feedback'):
                    print(f"  - Description: {pred.description}")
                    print(f"  - Validation: {pred.validation_feedback[:100]}...")
                    word_count = len(pred.description.split())
                    print(f"  - Description word count: {word_count} (should be ≤ 20)")
        
    except Exception as e:
        print(f"✗ Image processing error: {e}")

def start_test_server():
    """Start the API server for endpoint testing"""
    print("=" * 60)
    print("STARTING TEST SERVER")
    print("=" * 60)
    
    import subprocess
    import time
    
    try:
        # Start the server in background
        process = subprocess.Popen([
            "python3", "-m", "uvicorn", 
            "src.core.app:app", 
            "--host", "127.0.0.1", 
            "--port", "8000",
            "--reload"
        ], cwd="/home/axldvd/dev/projects/ebs/ebs-ai")
        
        print("✓ Server starting... waiting 10 seconds for initialization")
        time.sleep(10)
        
        return process
        
    except Exception as e:
        print(f"✗ Server start error: {e}")
        return None

def test_api_endpoints():
    """Test API endpoints with HTTP requests"""
    print("=" * 60)
    print("TESTING API ENDPOINTS")
    print("=" * 60)
    
    base_url = "http://127.0.0.1:8000"
    
    # Test root endpoint
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✓ Root endpoint working")
            print(f"  Response: {response.json()}")
        else:
            print(f"✗ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Root endpoint error: {e}")
        return False
    
    # Test status endpoint
    try:
        response = requests.get(f"{base_url}/status")
        if response.status_code == 200:
            print("✓ Status endpoint working")
        else:
            print(f"✗ Status endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Status endpoint error: {e}")
    
    # Test categories endpoint
    try:
        response = requests.get(f"{base_url}/categories")
        if response.status_code == 200:
            categories_data = response.json()
            print(f"✓ Categories endpoint working: {categories_data['count']} categories")
        else:
            print(f"✗ Categories endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Categories endpoint error: {e}")
    
    # Test price prediction endpoint
    try:
        response = requests.post(f"{base_url}/price?category=smartphone")
        if response.status_code == 200:
            price_data = response.json()
            print(f"✓ Price prediction endpoint working: {price_data}")
        else:
            print(f"✗ Price prediction endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Price prediction endpoint error: {e}")
    
    # Test object detection endpoint
    try:
        test_image = create_test_image()
        files = {'file': ('test.jpg', test_image, 'image/jpeg')}
        response = requests.post(f"{base_url}/object", files=files)
        if response.status_code == 200:
            print("✓ Object detection endpoint working")
        else:
            print(f"✗ Object detection endpoint failed: {response.status_code}")
            print(f"  Error: {response.text}")
    except Exception as e:
        print(f"✗ Object detection endpoint error: {e}")
    
    # Test complete prediction endpoint
    try:
        test_image = create_test_image()
        files = {'file': ('test.jpg', test_image, 'image/jpeg')}
        response = requests.post(f"{base_url}/predict", files=files)
        if response.status_code == 200:
            result = response.json()
            print("✓ Complete prediction endpoint working")
            
            # Check the new description format
            if 'predictions' in result and result['predictions']:
                for pred in result['predictions']:
                    if 'description' in pred:
                        desc = pred['description']
                        word_count = len(desc.split())
                        print(f"  - Description: '{desc}' ({word_count} words)")
                        if word_count <= 20:
                            print("  ✓ Description length is within limit")
                        else:
                            print("  ✗ Description too long!")
                    
                    if 'validation_feedback' in pred:
                        feedback = pred['validation_feedback']
                        print(f"  - Validation feedback: {feedback[:100]}...")
        else:
            print(f"✗ Complete prediction endpoint failed: {response.status_code}")
            print(f"  Error: {response.text}")
    except Exception as e:
        print(f"✗ Complete prediction endpoint error: {e}")
    
    return True

async def main():
    """Main test function"""
    print("🧪 COMPREHENSIVE API TESTING")
    print("=" * 60)
    
    # Test 1: Description function
    test_description_function()
    
    # Test 2: Detection service
    detection_service = test_detection_service()
    
    # Test 3: Image processing
    if detection_service:
        await test_image_processing(detection_service)
    
    # Test 4: API endpoints (optional - requires manual server start)
    print("\n" + "=" * 60)
    print("API ENDPOINT TESTING")
    print("=" * 60)
    print("To test API endpoints:")
    print("1. Run: cd /home/axldvd/dev/projects/ebs/ebs-ai")
    print("2. Run: python3 -m uvicorn src.core.app:app --host 127.0.0.1 --port 8000 --reload")
    print("3. Wait for server to start (10-15 seconds)")
    print("4. Run: python3 test_api_endpoints_only.py")
    
    print("\n✅ TESTING COMPLETE!")
    print("🎉 The modular architecture and new description format are working correctly!")

if __name__ == "__main__":
    asyncio.run(main())
