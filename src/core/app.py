"""
Main FastAPI Application
E-waste detection API with complete pipeline processing

Pipeline Flow: YOLO Detection → Gemini Validation → YOLO-to-Price Mapping → Price Prediction
- Detects 37 YOLO categories, validates with Gemini, maps to 33 price categories
"""

import sys
import uvicorn
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config.settings import API_TITLE, API_DESCRIPTION, API_VERSION, HOST, PORT
from src.services.detection_service import DetectionService
from src.models.response_models import FullResponse, ObjectResponse, PriceResponse

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detection service
detection_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize detection service on startup"""
    global detection_service
    detection_service = DetectionService()

# API Endpoints

@app.get("/")
def root():
    """Health check and system status"""
    status = detection_service.get_system_status() if detection_service else {}
    return {
        "status": "ok", 
        "message": "E-Waste Detection API is running",
        "version": API_VERSION,
        "pipeline_flow": "YOLO Detection → Gemini Validation → YOLO-to-Price Mapping → Price Prediction",
        "endpoints": ["/predict", "/object", "/price"],
        "categories": {
            "yolo_categories": 37,
            "price_categories": 33
        },
        "system_status": status
    }

@app.post(
    "/predict",
    response_model=FullResponse,
    summary="Complete e-waste analysis pipeline",
    tags=["Complete Analysis"]
)
async def predict(file: UploadFile = File(..., description="Image file for complete analysis")):
    """
    Complete e-waste analysis pipeline including:
    
    **Pipeline Flow:**
    1. **YOLO Detection**: Detects objects in 37 YOLO categories
    2. **Gemini Validation**: AI validates/corrects YOLO predictions
    3. **Category Mapping**: Maps validated YOLO categories to 33 price categories
    4. **Price Prediction**: Predicts price using KNR model
    5. **Content Generation**: Creates descriptions, suggestions, damage assessment
    
    **Output:** Validated YOLO categories with price predictions and disposal guidance
    """
    if not detection_service:
        raise HTTPException(status_code=503, detail="Detection service not initialized")
    
    try:
        contents = await file.read()
        result = await detection_service.process_image_complete(contents)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post(
    "/object",
    response_model=ObjectResponse,
    summary="YOLO object detection only",
    tags=["Object Detection"]
)
async def detect_objects(file: UploadFile = File(..., description="Image file for object detection")):
    """
    YOLO object detection only - returns detected objects in raw YOLO categories
    
    **No validation, mapping, or price prediction**
    - Direct YOLO output (37 categories)
    - Fastest endpoint for simple detection needs
    - Returns bounding boxes and confidence scores
    """
    if not detection_service:
        raise HTTPException(status_code=503, detail="Detection service not initialized")
    
    try:
        contents = await file.read()
        result = await detection_service.detect_objects_only(contents)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")

@app.post(
    "/price",
    response_model=PriceResponse,
    summary="Price prediction only",
    tags=["Price Prediction"]
)
async def predict_price(category: str, condition: str = "Baik"):
    """
    Price prediction only - given a **price model category** and condition, return estimated price
    
    **Important:** Uses the 33 price model categories, not YOLO categories
    
    **Parameters:**
    - **category**: Price model category (33 categories)
    - **condition**: Item condition - "Baik" (good), "Biasa" (average), "Buruk" (poor)
    
    **Examples:**
    - category="Handphone", condition="Baik" (not "Phone")
    - category="Laptop", condition="Biasa"
    - category="Baterai Laptop", condition="Buruk" (not "Battery")
    - category="Adaptor /Kilo", condition="Baik" (not "Charger")
    
    Use `/categories` endpoint to get the full list of supported price categories.
    """
    if not detection_service:
        raise HTTPException(status_code=503, detail="Detection service not initialized")
    
    # Validate condition
    valid_conditions = ["Baik", "Biasa", "Buruk"]
    if condition not in valid_conditions:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Invalid condition: {condition}",
                "valid_conditions": valid_conditions,
                "note": "Condition must be one of: Baik (good), Biasa (average), Buruk (poor)"
            }
        )
    
    # Validate category
    supported_categories = detection_service.get_supported_categories()
    if category not in supported_categories:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Invalid category: {category}",
                "supported_categories": supported_categories,
                "note": "This endpoint only accepts the 33 price model categories",
                "hint": "Use /categories to see all supported price categories"
            }
        )
    
    result = detection_service.predict_price_only(category, condition)
    if result is None:
        raise HTTPException(status_code=500, detail="Price prediction failed")
    
    return result

@app.get("/categories")
def get_supported_categories():
    """Get list of supported price prediction categories (33 price model categories)"""
    if not detection_service:
        return {"categories": [], "count": 0, "note": "Detection service not initialized"}
    
    categories = detection_service.get_supported_categories()
    return {
        "categories": categories,
        "count": len(categories),
        "type": "price_model_categories",
        "note": "These are the 33 price categories used for price prediction, not YOLO categories"
    }

@app.get("/yolo-categories")
def get_yolo_categories():
    """Get list of YOLO detection categories (37 YOLO categories)"""
    from src.utils.mappings import get_all_yolo_classes
    
    yolo_categories = get_all_yolo_classes()
    return {
        "categories": yolo_categories,
        "count": len(yolo_categories),
        "type": "yolo_detection_categories",
        "note": "These are the 37 YOLO categories used for object detection"
    }

@app.get("/mapping")
def get_category_mapping():
    """Get YOLO to Price category mapping"""
    from src.utils.mappings import YOLO_TO_PRICE_MAP
    
    return {
        "mapping": YOLO_TO_PRICE_MAP,
        "yolo_categories": len(YOLO_TO_PRICE_MAP),
        "price_categories": len(set(YOLO_TO_PRICE_MAP.values())),
        "note": "Shows how YOLO categories (37) map to price categories (33)"
    }

@app.get("/status")
def get_system_status():
    """Get detailed system status"""
    if not detection_service:
        return {"error": "Detection service not initialized"}
    
    return detection_service.get_system_status()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
