from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
import os
from src.utils.cloud_storage import CloudStorage
import tempfile
import uuid
import httpx
from typing import List, Dict, Any, Optional
from fastapi import status
from pydantic import BaseModel, Field

# Try to import YOLO from ultralytics, fallback to placeholder if not available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

app = FastAPI(title="E-Waste YOLOv11 Inference API")

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup (update path as needed)
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '/models/v3.pt'))
if YOLO_AVAILABLE:
    model = YOLO(MODEL_PATH)
else:
    model = None  # Placeholder

# Initialize GCS uploader
cloud_storage = CloudStorage()

class Prediction(BaseModel):
    class_: int = Field(..., alias="class", description="Class index of the detected object")
    confidence: float = Field(..., description="Confidence score of the detection")
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")

class PredictResponse(BaseModel):
    predictions: List[Prediction]
    image_url: Optional[str]

@app.get("/")
def root():
    return {"status": "ok", "message": "YOLOv11 FastAPI is running"}

@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Run YOLOv11 inference on an uploaded image",
    response_description="Predictions and image URL",
    status_code=status.HTTP_200_OK,
    tags=["Inference"]
)
async def predict(file: UploadFile = File(..., description="Image file (jpg, png, etc.) to run inference on")):
    """
    Upload an image and run YOLOv11 inference.
    - **Request:** multipart/form-data with an image file.
    - **Response:** List of predictions (class, confidence, bbox) and the public image URL.
    """
    if not YOLO_AVAILABLE or model is None:
        return JSONResponse(status_code=503, content={"error": "YOLO model not available"})
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        results = model(image)
        predictions = [
            {
                "class": int(box.cls),
                "confidence": float(box.conf),
                "bbox": [float(x) for x in box.xyxy[0].tolist()]
            }
            for box in results[0].boxes
        ]
        run_id = str(uuid.uuid4())
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            image.save(tmp.name, format='JPEG')
            tmp_path = tmp.name
        public_url = None
        try:
            public_url = cloud_storage.upload_detection_result(tmp_path, run_id)
        except Exception:
            public_url = None
        finally:
            os.remove(tmp_path)
        return {"predictions": predictions, "image_url": public_url}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# To run: uvicorn core.app:app --reload

# Requirements (add to requirements.txt):
# fastapi
# uvicorn
# torch
# ultralytics
# pillow
