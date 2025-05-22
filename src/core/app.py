import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from fastapi import FastAPI, File, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
from src.utils.cloud_storage import CloudStorage
import tempfile
import uuid
import httpx
from typing import List, Dict, Any, Optional
from fastapi import status
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.neighbors import KNeighborsRegressor

CLASS_NAMES = {
    0: "Air-Conditioner",
    1: "Bar-Phone",
    2: "Battery",
    3: "Blood-Pressure-Monitor",
    4: "Boiler",
    5: "CRT-Monitor",
    6: "CRT-TV",
    7: "Calculator",
    8: "Camera",
    9: "Ceiling-Fan",
    10: "Christmas-Lights",
    11: "Clothes-Iron",
    12: "Coffee-Machine",
    13: "Compact-Fluorescent-Lamps",
    14: "Computer-Keyboard",
    15: "Computer-Mouse",
    16: "Cooled-Dispenser",
    17: "Cooling-Display",
    18: "Dehumidifier",
    19: "Desktop-PC",
    20: "Digital-Oscilloscope",
    21: "Dishwasher",
    22: "Drone",
    23: "Electric-Bicycle",
    24: "Electric-Guitar",
    25: "Electrocardiograph-Machine",
    26: "Electronic-Keyboard",
    27: "Exhaust-Fan",
    28: "Flashlight",
    29: "Flat-Panel-Monitor",
    30: "Flat-Panel-TV",
    31: "Floor-Fan",
    32: "Freezer",
    33: "Glucose-Meter",
    34: "HDD",
    35: "Hair-Dryer",
    36: "Headphone",
    37: "LED-Bulb",
    38: "Laptop",
    39: "Microwave",
    40: "Music-Player",
    41: "Neon-Sign",
    42: "Network-Switch",
    43: "Non-Cooled-Dispenser",
    44: "Oven",
    45: "PCB",
    46: "Patient-Monitoring-System",
    47: "Photovoltaic-Panel",
    48: "PlayStation-5",
    49: "Power-Adapter",
    50: "Printer",
    51: "Projector",
    52: "Pulse-Oximeter",
    53: "Range-Hood",
    54: "Refrigerator",
    55: "Rotary-Mower",
    56: "Router",
    57: "SSD",
    58: "Server",
    59: "Smart-Watch",
    60: "Smartphone",
    61: "Smoke-Detector",
    62: "Soldering-Iron",
    63: "Speaker",
    64: "Stove",
    65: "Straight-Tube-Fluorescent-Lamp",
    66: "Street-Lamp",
    67: "TV-Remote-Control",
    68: "Table-Lamp",
    69: "Tablet",
    70: "Telephone-Set",
    71: "Toaster",
    72: "Tumble-Dryer",
    73: "USB-Flash-Drive",
    74: "Vacuum-Cleaner",
    75: "Washing-Machine",
    76: "Xbox-Series-X"
}

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

app = FastAPI(title="E-Waste YOLOv11 Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class KNRModelManager:
    def __init__(self, encoder=None, model=None):
        self.encoder = encoder if encoder is not None else TargetEncoder()
        self.model = model if model is not None else KNeighborsRegressor(n_neighbors=12)
        self.fitted = False

    def predict(self, item_names):
        print("predict")
        item_name = pd.DataFrame([item_names], columns=["Nama Item"])
        X_encoded = self.encoder.transform(item_name["Nama Item"])
        return self.model.predict(X_encoded)

    def load(self, model_path=None, encoder_path=None):
        model_path = model_path or os.environ.get('KNR_MODEL_PATH', 'knr_models/model_knr_best.joblib')
        encoder_path = encoder_path or os.environ.get('KNR_ENCODER_PATH', 'knr_models/encoder_target.joblib')
        self.model = joblib.load(model_path)
        self.encoder = joblib.load(encoder_path)
        self.fitted = True

def get_model_path():
    model_path = os.environ.get('MODEL_PATH', 'models/v4.pt')
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.getcwd(), model_path)
    return model_path

if YOLO_AVAILABLE:
    model = YOLO(get_model_path())
else:
    model = None  

cloud_storage = CloudStorage()

class Prediction(BaseModel):
    class_: int = Field(..., alias="class", description="Class index of the detected object")
    class_name: str = Field(..., description="Name of the detected object class")
    confidence: float = Field(..., description="Confidence score of the detection")
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")

class PredictResponse(BaseModel):
    predictions: List[Prediction]
    image_url: Optional[str]
    
class DetectResponse(BaseModel):
    price: int  

@app.get("/")
def root():
    return {"status": "ok", "message": "YOLOv11 and KNeighborRegression FastAPI is running, and at your service"}

@app.post(
    "/object",
    response_model=PredictResponse,
    summary="Run YOLOv11 inference on an uploaded image",
    response_description="Predictions and image URL",
    status_code=status.HTTP_200_OK,
    tags=["Inference"]
)
async def predict(file: UploadFile = File(..., description="Image file (jpg, png, etc.) to run inference on")):
    if not YOLO_AVAILABLE or model is None:
        return JSONResponse(status_code=503, content={"error": "YOLO model not available"})
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        results = model(image)
        predictions = [
            {
                "class": int(box.cls),
                "class_name": CLASS_NAMES.get(int(box.cls), "unknown"),
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
    
@app.post(
    "/price",
    response_model=DetectResponse,
    summary="Run KNeighborRegression inference on an e-waste category",
    response_description="Predictions Price",
    status_code=status.HTTP_200_OK,
    tags=["Inference"]
)

async def detect(object: str = Body(..., embed=True)):
    try:

        knr_manager_loaded = KNRModelManager()
        knr_manager_loaded.load()
        pred = knr_manager_loaded.predict(object)
        return {"price": int(pred[0])}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})