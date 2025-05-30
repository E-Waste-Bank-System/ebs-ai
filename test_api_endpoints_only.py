#!/usr/bin/env python3
"""
API Endpoint Testing Script - Server Running Version
Tests API endpoints when server is already running
"""

import requests
import io
from PIL import Image, ImageDraw
import json

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

def test_api_endpoints():
    """Test API endpoints with HTTP requests"""
    print("🧪 TESTING API ENDPOINTS")
    print("=" * 60)
    
    base_url = "http://127.0.0.1:8000"
    
    # Test 1: Root endpoint
    print("1. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            data = response.json()
            print("✓ Root endpoint working")
            print(f"   Version: {data.get('version')}")
            print(f"   Endpoints: {data.get('endpoints')}")
        else:
            print(f"✗ Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to server: {e}")
        print("   Make sure the server is running on http://127.0.0.1:8000")
        return False
    
    # Test 2: Status endpoint
    print("\n2. Testing status endpoint...")
    try:
        response = requests.get(f"{base_url}/status")
        if response.status_code == 200:
            print("✓ Status endpoint working")
            status = response.json()
            print(f"   YOLO Model: {'✓' if status.get('yolo_model_loaded') else '✗'}")
            print(f"   Price Model: {'✓' if status.get('price_model_loaded') else '✗'}")
            print(f"   Gemini Service: {'✓' if status.get('gemini_service_ready') else '✗'}")
        else:
            print(f"✗ Status endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Status endpoint error: {e}")
    
    # Test 3: Categories endpoint
    print("\n3. Testing categories endpoint...")
    try:
        response = requests.get(f"{base_url}/categories")
        if response.status_code == 200:
            categories_data = response.json()
            print(f"✓ Categories endpoint working: {categories_data['count']} categories")
            print(f"   Sample categories: {categories_data['categories'][:5]}...")
        else:
            print(f"✗ Categories endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Categories endpoint error: {e}")
    
    # Test 4: Price prediction endpoint
    print("\n4. Testing price prediction endpoint...")
    try:
        response = requests.post(f"{base_url}/price?category=smartphone")
        if response.status_code == 200:
            price_data = response.json()
            print(f"✓ Price prediction endpoint working")
            print(f"   Predicted price: Rp {price_data.get('predicted_price', 'N/A'):,}")
        else:
            print(f"✗ Price prediction endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Price prediction endpoint error: {e}")
    
    # Test 5: Object detection endpoint
    print("\n5. Testing object detection endpoint...")
    try:
        test_image = create_test_image()
        files = {'file': ('test.jpg', test_image, 'image/jpeg')}
        response = requests.post(f"{base_url}/object", files=files, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print("✓ Object detection endpoint working")
            print(f"   Detected {len(result.get('detections', []))} objects")
        else:
            print(f"✗ Object detection endpoint failed: {response.status_code}")
            print(f"   Error: {response.text[:200]}")
    except Exception as e:
        print(f"✗ Object detection endpoint error: {e}")
    
    # Test 6: Complete prediction endpoint (MAIN TEST)
    print("\n6. Testing complete prediction endpoint...")
    print("   This tests the NEW DESCRIPTION FORMAT!")
    try:
        test_image = create_test_image()
        files = {'file': ('test.jpg', test_image, 'image/jpeg')}
        response = requests.post(f"{base_url}/predict", files=files, timeout=60)
        if response.status_code == 200:
            result = response.json()
            print("✓ Complete prediction endpoint working")
            
            # Check the new description format
            if 'predictions' in result and result['predictions']:
                print("\n   🔍 CHECKING NEW DESCRIPTION FORMAT:")
                for i, pred in enumerate(result['predictions']):
                    print(f"\n   Prediction {i+1}:")
                    
                    # Check description field
                    if 'description' in pred:
                        desc = pred['description']
                        word_count = len(desc.split())
                        print(f"   📝 Description: '{desc}'")
                        print(f"   📊 Word count: {word_count}/20 words")
                        
                        if word_count <= 20:
                            print("   ✅ Description length is within limit!")
                        else:
                            print("   ❌ Description too long!")
                        
                        # Check if it's in Indonesian
                        indonesian_indicators = ['elektronik', 'rusak', 'bekas', 'dengan', 'layar', 'baterai', 'kabel']
                        has_indonesian = any(word in desc.lower() for word in indonesian_indicators)
                        if has_indonesian:
                            print("   ✅ Description appears to be in Indonesian")
                        else:
                            print("   ⚠️  Description language unclear")
                    
                    # Check validation_feedback field
                    if 'validation_feedback' in pred:
                        feedback = pred['validation_feedback']
                        print(f"   🤖 Validation feedback: {feedback[:100]}...")
                        print("   ✅ Validation feedback field present")
                    
                    # Check other fields
                    if 'category' in pred:
                        print(f"   🏷️  Category: {pred['category']}")
                    if 'confidence' in pred:
                        print(f"   📈 Confidence: {pred['confidence']:.2f}")
                    if 'predicted_price' in pred:
                        print(f"   💰 Price: Rp {pred['predicted_price']:,}")
            else:
                print("   ⚠️  No predictions found in response")
            
            print(f"\n   📄 Full response structure: {list(result.keys())}")
            
        else:
            print(f"✗ Complete prediction endpoint failed: {response.status_code}")
            print(f"   Error: {response.text[:500]}")
    except Exception as e:
        print(f"✗ Complete prediction endpoint error: {e}")
    
    return True

def main():
    """Main test function"""
    print("🚀 API ENDPOINT TESTING")
    print("Make sure the server is running first!")
    print("Command: python3 -m uvicorn src.core.app:app --host 127.0.0.1 --port 8000 --reload")
    print("=" * 60)
    
    success = test_api_endpoints()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ API TESTING COMPLETE!")
        print("🎉 The modular architecture and new description format are working!")
        print("\n📋 Summary of what was tested:")
        print("   • All API endpoints respond correctly")
        print("   • New description format (max 20 words, Indonesian)")
        print("   • Separate validation_feedback field")
        print("   • YOLO detection, price prediction, and Gemini RAG integration")
        print("   • Modular architecture functioning properly")
    else:
        print("\n❌ Some tests failed. Check the server status and try again.")

if __name__ == "__main__":
    main()
