"""
YOLO Object Detection Module
Handles YOLO model loading and inference - First stage of the pipeline

Pipeline: YOLO Detection → Gemini Validation → YOLO-to-Price Mapping → Price Prediction
This module handles stage 1: Detects objects in 37 YOLO categories
"""

import os
import logging
from typing import List, Optional, Tuple
from PIL import Image

from src.config.settings import YOLO_MODEL_PATH, YOLO_AVAILABLE
from src.utils.mappings import CLASS_NAMES, get_class_name_for_index
from src.models.response_models import Detection
from src.utils.helpers import generate_unique_id

logger = logging.getLogger(__name__)


class YOLODetector:
    """
    YOLO Object Detection Manager
    
    Responsible for the first stage of the pipeline:
    - Loads YOLO model (37 categories)
    - Detects e-waste objects in images
    - Returns raw YOLO predictions before validation
    """
    
    def __init__(self):
        self.model = None
        self.is_loaded = False
    
    def load_model(self) -> bool:
        """
        Load YOLO model from file
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if not YOLO_AVAILABLE:
            logger.warning("YOLO not available - ultralytics not installed")
            return False
        
        try:
            if os.path.exists(YOLO_MODEL_PATH):
                from ultralytics import YOLO
                self.model = YOLO(YOLO_MODEL_PATH)
                self.is_loaded = True
                logger.info(f"YOLO model loaded successfully from {YOLO_MODEL_PATH}")
                logger.info("YOLO will detect 37 e-waste categories")
                return True
            else:
                logger.warning(f"YOLO model not found: {YOLO_MODEL_PATH}")
                return False
        except Exception as e:
            logger.error(f"Error loading YOLO model: {str(e)}")
            return False
    
    def detect_objects(self, image_path: str, save_annotated_path: str = None) -> List[Detection]:
        """
        Detect e-waste objects in image using YOLO (Stage 1 of pipeline)
        
        Args:
            image_path: Path to input image
            save_annotated_path: Optional path to save annotated image
            
        Returns:
            List of Detection objects with YOLO categories (37 classes)
        """
        if not self.is_loaded or self.model is None:
            logger.error("YOLO model not loaded")
            return []
            
        try:
            # Run YOLO prediction
            results = self.model.predict(source=image_path, show=False, verbose=False)
            
            # Save annotated image if requested
            if save_annotated_path:
                results[0].save(filename=save_annotated_path)
                logger.info(f"Annotated image saved to: {save_annotated_path}")
            
            detections = []
            for box in results[0].boxes:
                class_idx = int(box.cls)
                
                # Get class name using the mapping function
                class_name = get_class_name_for_index(class_idx)
                
                confidence = float(box.conf)
                bbox = [float(x) for x in box.xyxy[0].tolist()]
                
                detection = Detection(
                    id=generate_unique_id(),
                    category=class_name,
                    confidence=round(confidence, 3),
                    bbox=bbox
                )
                detections.append(detection)
                
            logger.info(f"YOLO detected {len(detections)} objects in 37 categories")
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
            List of tuples (yolo_category, confidence, bbox)
        """
        if not self.is_loaded or self.model is None:
            logger.error("YOLO model not loaded")
            return []
        
        try:
            results = self.model.predict(source=image_path, show=False, verbose=False)
            
            details = []
            for box in results[0].boxes:
                class_idx = int(box.cls)
                
                # Get class name using the mapping function
                class_name = get_class_name_for_index(class_idx)
                
                confidence = float(box.conf)
                bbox = [float(x) for x in box.xyxy[0].tolist()]
                
                details.append((class_name, confidence, bbox))
            
            logger.info(f"YOLO raw detection details: {len(details)} objects")
            return details
            
        except Exception as e:
            logger.error(f"Error getting YOLO detection details: {str(e)}")
            return []
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded YOLO model
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_loaded": self.is_loaded,
            "model_path": YOLO_MODEL_PATH if self.is_loaded else None,
            "categories_count": 37,
            "pipeline_stage": "1 - Object Detection",
            "output": "Raw YOLO categories before validation"
        }
