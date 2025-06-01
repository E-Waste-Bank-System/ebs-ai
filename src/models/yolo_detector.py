"""
YOLO Object Detection Module
Handles YOLO model loading and inference
"""

import os
import logging
from typing import List, Optional, Tuple
from PIL import Image

from src.config.settings import YOLO_MODEL_PATH, YOLO_AVAILABLE
from src.utils.mappings import CLASS_NAMES
from src.models.response_models import Detection
from src.utils.helpers import generate_unique_id

logger = logging.getLogger(__name__)


class YOLODetector:
    """YOLO Object Detection Manager"""
    
    def __init__(self):
        self.model = None
        self.is_loaded = False
    
    def load_model(self) -> bool:
        """Load YOLO model from file"""
        if not YOLO_AVAILABLE:
            logger.warning("YOLO not available - ultralytics not installed")
            return False
        
        try:
            if os.path.exists(YOLO_MODEL_PATH):
                from ultralytics import YOLO
                self.model = YOLO(YOLO_MODEL_PATH)
                self.is_loaded = True
                logger.info("YOLO model loaded successfully")
                return True
            else:
                logger.warning(f"YOLO model not found: {YOLO_MODEL_PATH}")
                return False
        except Exception as e:
            logger.error(f"Error loading YOLO model: {str(e)}")
            return False
    
    def detect_objects(self, image_path: str, save_annotated_path: str = None) -> List[Detection]:
        """
        Detect objects in image using YOLO
        Optionally save annotated image using Ultralytics' built-in method.
        """
        if not self.is_loaded or self.model is None:
            logger.error("YOLO model not loaded")
            return []
        try:
            # Run YOLO prediction
            results = self.model.predict(source=image_path, show=False, verbose=False)
            if save_annotated_path:
                results[0].save(filename=save_annotated_path)
            detections = []
            for box in results[0].boxes:
                class_idx = int(box.cls)
                class_name = CLASS_NAMES[class_idx]
                confidence = float(box.conf)
                bbox = [float(x) for x in box.xyxy[0].tolist()]
                detection = Detection(
                    id=generate_unique_id(),
                    category=class_name,
                    confidence=round(confidence, 3),
                    bbox=bbox
                )
                detections.append(detection)
            logger.info(f"YOLO detected {len(detections)} objects")
            return detections
        except Exception as e:
            logger.error(f"Error in YOLO detection: {str(e)}")
            return []
    
    def get_detection_details(self, image_path: str) -> List[Tuple[str, float, List[float]]]:
        """
        Get raw detection details for further processing
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of tuples (class_name, confidence, bbox)
        """
        if not self.is_loaded or self.model is None:
            logger.error("YOLO model not loaded")
            return []
        
        try:
            results = self.model.predict(source=image_path, show=False, verbose=False)
            
            details = []
            for box in results[0].boxes:
                class_idx = int(box.cls)
                class_name = CLASS_NAMES[class_idx]
                confidence = float(box.conf)
                bbox = [float(x) for x in box.xyxy[0].tolist()]
                
                details.append((class_name, confidence, bbox))
            
            return details
            
        except Exception as e:
            logger.error(f"Error getting YOLO detection details: {str(e)}")
            return []
