"""
Main FastAPI Application
Simplified and modular e-waste detection API
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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
        "endpoints": ["/predict", "/object", "/price"],
        "system_status": status
    }

@app.post(
    "/predict",
    response_model=FullResponse,
    summary="Complete e-waste analysis with YOLO, pricing, and RAG",
    tags=["Complete Analysis"]
)
async def predict(file: UploadFile = File(..., description="Image file for complete analysis")):
    """
    Complete e-waste analysis including:
    - YOLO object detection
    - Category mapping and validation
    - Gemini AI validation and correction
    - Price prediction
    - RAG-based disposal guidance
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
    YOLO object detection only - returns detected objects with bounding boxes
    No category mapping, validation, or price prediction
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
async def predict_price(category: str):
    """
    Price prediction only - given a category name, return estimated price
    Uses the 33 supported price categories only
    """
    if not detection_service:
        raise HTTPException(status_code=503, detail="Detection service not initialized")
    
    # Validate category
    supported_categories = detection_service.get_supported_categories()
    if category not in supported_categories:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Invalid category: {category}",
                "supported_categories": supported_categories,
                "note": "This endpoint only accepts the 33 price model categories"
            }
        )
    
    result = detection_service.predict_price_only(category)
    if result is None:
        raise HTTPException(status_code=500, detail="Price prediction failed")
    
    return result

@app.get("/categories")
def get_supported_categories():
    """Get list of supported price prediction categories"""
    if not detection_service:
        return {"categories": [], "count": 0}
    
    categories = detection_service.get_supported_categories()
    return {
        "categories": categories,
        "count": len(categories)
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
