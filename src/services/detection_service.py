"""
Detection Service Module
Orchestrates the complete e-waste detection workflow
"""

import os
import tempfile
import logging
from typing import List, Optional

from src.models.yolo_detector import YOLODetector
from src.models.price_predictor import PricePredictor
from src.services.gemini_service import GeminiService
from src.models.response_models import (
    Detection, FullPrediction, FullResponse, 
    ObjectResponse, PriceResponse
)
from src.utils.mappings import get_mapped_category
from src.utils.helpers import (
    generate_unique_id, generate_description, 
    generate_suggestions, calculate_risk_level
)

logger = logging.getLogger(__name__)


class DetectionService:
    """Main detection service orchestrating all components"""
    
    def __init__(self):
        # Initialize all components
        self.yolo_detector = YOLODetector()
        self.price_predictor = PricePredictor()
        self.gemini_service = GeminiService()
        
        # Load models
        self.yolo_loaded = self.yolo_detector.load_model()
        self.price_loaded = self.price_predictor.load_models()
        
        logger.info(f"Detection service initialized - YOLO: {self.yolo_loaded}, Price: {self.price_loaded}")
    
    async def process_image_complete(self, image_bytes: bytes) -> FullResponse:
        """
        Complete e-waste analysis pipeline
        
        Args:
            image_bytes: Image file bytes
            
        Returns:
            FullResponse with complete analysis
        """
        if not self.yolo_loaded:
            return FullResponse(
                predictions=[],
                rag_summary="YOLO model tidak tersedia untuk deteksi objek."
            )
        
        # Save image to temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        
        try:
            # Step 1: YOLO Detection
            raw_detections = self.yolo_detector.get_detection_details(tmp_path)
            
            if not raw_detections:
                return FullResponse(
                    predictions=[],
                    rag_summary="Tidak ada objek e-waste yang terdeteksi pada gambar ini."
                )
            
            # Step 2: Process each detection
            predictions = []
            detected_items = []
            
            for yolo_class_name, confidence, bbox in raw_detections:
                # Generate unique ID
                prediction_id = generate_unique_id()
                
                # Map YOLO prediction to price category
                mapped_category = get_mapped_category(yolo_class_name)
                
                # Enhanced Gemini validation workflow
                validation_result = await self.gemini_service.validate_detection(
                    tmp_path, yolo_class_name, mapped_category
                )
                
                # Skip if detection is invalid
                if not validation_result.is_valid:
                    logger.info(f"Detection rejected: {validation_result.gemini_feedback}")
                    continue
                
                final_category = validation_result.final_category
                detection_source = validation_result.detection_source
                
                # Price prediction
                price = None
                if self.price_loaded and final_category:
                    price = self.price_predictor.predict_price(final_category)
                
                # Generate metadata
                description = generate_description(final_category or yolo_class_name, confidence)
                suggestions = generate_suggestions(final_category or yolo_class_name)
                risk_level = calculate_risk_level(final_category or yolo_class_name, confidence)
                
                # Create prediction
                prediction = FullPrediction(
                    id=prediction_id,
                    category=final_category or yolo_class_name,
                    confidence=round(confidence, 2),
                    regression_result=price,
                    description=description,
                    validation_feedback=validation_result.gemini_feedback,
                    suggestion=suggestions,
                    risk_lvl=risk_level,
                    detection_source=detection_source
                )
                
                predictions.append(prediction)
                detected_items.append(final_category or yolo_class_name)
            
            # Step 3: Generate RAG summary
            if detected_items:
                rag_summary = await self.gemini_service.generate_disposal_summary(
                    list(set(detected_items))
                )
            else:
                rag_summary = "Tidak ada e-waste yang valid terdeteksi setelah validasi."
            
            return FullResponse(
                predictions=predictions,
                rag_summary=rag_summary
            )
            
        finally:
            # Clean up temporary file
            os.remove(tmp_path)
    
    async def detect_objects_only(self, image_bytes: bytes) -> ObjectResponse:
        """
        YOLO detection only
        
        Args:
            image_bytes: Image file bytes
            
        Returns:
            ObjectResponse with detected objects
        """
        if not self.yolo_loaded:
            return ObjectResponse(detections=[])
        
        # Save image to temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        
        try:
            detections = self.yolo_detector.detect_objects(tmp_path)
            return ObjectResponse(detections=detections)
        finally:
            os.remove(tmp_path)
    
    def predict_price_only(self, category: str) -> Optional[PriceResponse]:
        """
        Price prediction only
        
        Args:
            category: Category name
            
        Returns:
            PriceResponse or None if failed
        """
        if not self.price_loaded:
            return None
        
        if not self.price_predictor.is_category_supported(category):
            return None
        
        price = self.price_predictor.predict_price(category)
        if price is not None:
            return PriceResponse(category=category, price=price)
        return None
    
    def get_supported_categories(self) -> List[str]:
        """Get list of supported price categories"""
        if self.price_loaded:
            return self.price_predictor.get_supported_categories()
        return []
    
    def get_system_status(self) -> dict:
        """Get system component status"""
        return {
            "yolo_available": self.yolo_loaded,
            "price_prediction_available": self.price_loaded,
            "gemini_available": self.gemini_service.is_service_available(),
            "supported_categories_count": len(self.get_supported_categories()) if self.price_loaded else 0
        }
