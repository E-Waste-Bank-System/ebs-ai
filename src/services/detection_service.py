"""
Detection Service Module
Handles e-waste detection using YOLO and Gemini
"""

import logging
import os
import tempfile
import time
import asyncio
import re
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from src.config.settings import (
    YOLO_AVAILABLE, YOLO_MODEL_PATH, LOW_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD
)
from src.models.response_models import (
    Detection, FullPrediction, FullResponse,
    ValidationResult, ObjectResponse, PriceResponse
)
from src.models.yolo_detector import YOLODetector
from src.models.price_predictor import PricePredictor
from src.services.gemini_service import GeminiService
from src.utils.helpers import generate_unique_id, calculate_risk_level
from src.utils.mappings import get_mapped_category, is_valid_price_category

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
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes
        
        Args:
            box1: [x1, y1, x2, y2] coordinates of first box
            box2: [x1, y1, x2, y2] coordinates of second box
            
        Returns:
            IoU score between 0 and 1
        """
        # Get coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Calculate intersection area
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0
    
    def _filter_overlapping_detections(self, detections: List[Detection], iou_threshold: float = 0.5) -> List[Detection]:
        """
        Filter out overlapping detections keeping the one with highest confidence
        
        Args:
            detections: List of detections
            iou_threshold: IoU threshold for considering boxes as overlapping
            
        Returns:
            Filtered list of detections
        """
        if not detections:
            return []
        
        # Sort by confidence
        sorted_detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        filtered_detections = []
        
        for det in sorted_detections:
            # Check overlap with already accepted detections
            is_overlapping = False
            for accepted_det in filtered_detections:
                iou = self._calculate_iou(det.bbox, accepted_det.bbox)
                if iou > iou_threshold:
                    is_overlapping = True
                    logger.info(f"Filtered out overlapping detection: {det.category} (IoU: {iou:.2f})")
                    break
            
            if not is_overlapping:
                filtered_detections.append(det)
        
        return filtered_detections
    
    def _clean_description(self, description: str) -> str:
        """
        Clean up description text
        
        Args:
            description: Raw description text
            
        Returns:
            Cleaned description text
        """
        # Remove markdown and special characters
        description = re.sub(r'[*_#]', '', description)
        
        # Remove category and analysis prefixes
        description = re.sub(r'^(kategori|analisis|deskripsi)[:.]\s*', '', description, flags=re.IGNORECASE)
        
        # Remove numbered lists and bullet points
        description = re.sub(r'^\d+\.\s*', '', description, flags=re.MULTILINE)
        description = re.sub(r'^[-•]\s*', '', description, flags=re.MULTILINE)
        
        # Remove extra whitespace and newlines
        description = re.sub(r'\s+', ' ', description)
        description = description.strip()
        
        # Take only the first sentence if it's too long
        if len(description) > 100:
            description = description.split('.')[0] + '.'
        
        return description
    
    async def _generate_content_with_timeout(
        self, image_path: str, category: str, timeout: float = 10.0,
        extra_image_path: str = None, prompt_context: dict = None
    ) -> Tuple[str, List[str]]:
        """
        Generate description and suggestions with timeout, supporting extra image and prompt context.
        """
        try:
            async with asyncio.timeout(timeout):
                description = await self.gemini_service.generate_description(
                    image_path, category, extra_image_path=extra_image_path, prompt_context=prompt_context
                )
                suggestions = await self.gemini_service.generate_suggestions(
                    image_path, category, extra_image_path=extra_image_path, prompt_context=prompt_context
                )
                return self._clean_description(description), suggestions
        except asyncio.TimeoutError:
            logger.warning(f"Content generation timed out after {timeout} seconds")
            return f"{category} terdeteksi dalam gambar.", ["Bawa ke pusat daur ulang e-waste terdekat."]
    
    def _get_category_color(self, category: str) -> Tuple[int, int, int]:
        """Assign a unique color to each category using a hash."""
        import random
        random.seed(hash(category) & 0xFFFFFFFF)
        return tuple(random.randint(64, 255) for _ in range(3))

    def _save_cropped_bbox(self, image_path: str, bbox: List[float], label: str) -> str:
        """
        Crop the image to the bounding box and save to a temp file. Returns the file path.
        """
        image = Image.open(image_path).convert("RGB")
        # Ensure bbox is int
        bbox_int = [int(x) for x in bbox]
        cropped = image.crop(bbox_int)
        temp_cropped = tempfile.NamedTemporaryFile(suffix=f'_{label}.jpg', delete=False)
        cropped.save(temp_cropped.name)
        logger.info(f"Cropped image for '{label}' saved: {temp_cropped.name}")
        return temp_cropped.name

    def _is_valid_crop(self, crop_path: str, min_size: int = 32) -> bool:
        """Check if the cropped image is large enough for Gemini."""
        try:
            with Image.open(crop_path) as img:
                width, height = img.size
                return width >= min_size and height >= min_size
        except Exception as e:
            logger.warning(f"Failed to open crop image {crop_path}: {e}")
            return False

    async def process_image_complete(
        self, 
        image_bytes: bytes,
        confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD
    ) -> FullResponse:
        """
        Process image with complete pipeline including validation and pricing
        """
        if not YOLO_AVAILABLE:
            logger.error("YOLO not available")
            return FullResponse(predictions=[])
        
        # Save image to temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        
        annotated_path = "ebs-ai/tmp/annotated_yolo.jpg"
        cropped_paths = []
        try:
            # Run YOLO detection with timing and save annotated image using Ultralytics
            yolo_start = time.time()
            detections = self.yolo_detector.detect_objects(tmp_path, save_annotated_path=annotated_path)
            yolo_time = time.time() - yolo_start
            logger.info(f"YOLO detection completed in {yolo_time:.2f} seconds")
            logger.info(f"Annotated image saved for debugging: {annotated_path}")
            
            if not detections:
                logger.info("No detections found")
                return FullResponse(predictions=[])
            
            # Log YOLO detections
            for det in detections:
                logger.info(f"YOLO detected: {det.category} with confidence {det.confidence:.3f}")
            
            # Filter overlapping detections
            filtered_detections = self._filter_overlapping_detections(detections)
            logger.info(f"Filtered {len(detections) - len(filtered_detections)} overlapping detections")
            
            # Process each detection
            predictions = []
            for det in filtered_detections:
                mapped_category = get_mapped_category(det.category)
                logger.info(f"Mapped YOLO category '{det.category}' to price category '{mapped_category}'")
                cropped_path = self._save_cropped_bbox(tmp_path, det.bbox, det.category)
                cropped_paths.append(cropped_path)
                # --- Minimum crop size check ---
                if not self._is_valid_crop(cropped_path):
                    logger.warning(f"Cropped image too small for Gemini: {cropped_path}. Using YOLO result.")
                    final_category = mapped_category
                    price = self.price_predictor.predict_price(final_category)
                    description = f"Perangkat elektronik {final_category.lower()}"
                    suggestions = [
                        "Periksa panduan manufacturer",
                        "Pisahkan komponen berbahaya",
                        "Bawa ke pusat daur ulang e-waste"
                    ]
                    risk_level = calculate_risk_level(final_category, det.confidence)
                    damage_level = 3  # Default damage level
                    prediction = FullPrediction(
                        id=generate_unique_id(),
                        category=final_category,
                        confidence=det.confidence,
                        regression_result=price,
                        description=description,
                        bbox=det.bbox,
                        suggestion=suggestions,
                        risk_lvl=risk_level,
                        damage_level=damage_level,
                        detection_source="YOLO (crop too small)"
                    )
                    predictions.append(prediction)
                    continue
                # --- Robust Gemini validation ---
                try:
                    gemini_start = time.time()
                    validation = await self.gemini_service.validate_detection(
                        cropped_path, det.category, mapped_category,
                        extra_image_path=None,
                        prompt_context={
                            "all_detections": [
                                {"category": d.category, "confidence": d.confidence, "bbox": d.bbox} for d in filtered_detections
                            ],
                            "focus_bbox": det.bbox,
                            "focus_label": det.category
                        }
                    )
                    gemini_time = time.time() - gemini_start
                    logger.info(f"Gemini validation completed in {gemini_time:.2f} seconds")
                    if not validation or not hasattr(validation, 'is_valid') or not validation.is_valid:
                        logger.warning(f"Gemini validation failed or invalid: {getattr(validation, 'gemini_feedback', '')}")
                        final_category = mapped_category
                        detection_source = "YOLO (Gemini fail)"
                    else:
                        final_category = validation.final_category or mapped_category
                        detection_source = validation.detection_source
                        if validation.final_category:
                            logger.info(f"Gemini corrected category from '{mapped_category}' to '{final_category}'")
                except Exception as e:
                    logger.error(f"Gemini validation exception: {e}")
                    final_category = mapped_category
                    detection_source = "YOLO (Gemini exception)"
                
                # --- Damage level analysis ---
                try:
                    damage_level, damage_analysis = await self.gemini_service.analyze_damage_level(
                        cropped_path, final_category,
                        extra_image_path=None,
                        prompt_context={
                            "all_detections": [
                                {"category": d.category, "confidence": d.confidence, "bbox": d.bbox} for d in filtered_detections
                            ],
                            "focus_bbox": det.bbox,
                            "focus_label": final_category
                        }
                    )
                    logger.info(f"Damage level analysis: {damage_level} - {damage_analysis}")
                except Exception as e:
                    logger.error(f"Damage level analysis exception: {e}")
                    damage_level = 3  # Default damage level
                
                # --- Price prediction ---
                price = self.price_predictor.predict_price(final_category)
                # --- Robust Gemini description/suggestions ---
                try:
                    gemini_start = time.time()
                    description, suggestions = await self._generate_content_with_timeout(
                        cropped_path, final_category,
                        extra_image_path=None,
                        prompt_context={
                            "all_detections": [
                                {"category": d.category, "confidence": d.confidence, "bbox": d.bbox} for d in filtered_detections
                            ],
                            "focus_bbox": det.bbox,
                            "focus_label": final_category
                        }
                    )
                    gemini_time = time.time() - gemini_start
                    logger.info(f"Gemini content generation completed in {gemini_time:.2f} seconds")
                    if not description or not suggestions or not isinstance(suggestions, list):
                        raise ValueError("Empty or invalid Gemini description/suggestions")
                except Exception as e:
                    logger.error(f"Gemini content generation exception: {e}")
                    description = f"Perangkat elektronik {final_category.lower()}"
                    suggestions = [
                        "Periksa panduan manufacturer",
                        "Pisahkan komponen berbahaya",
                        "Bawa ke pusat daur ulang e-waste"
                    ]
                risk_level = calculate_risk_level(final_category, det.confidence)
                prediction = FullPrediction(
                    id=generate_unique_id(),
                    category=final_category,
                    confidence=det.confidence,
                    regression_result=price,
                    description=description,
                    bbox=det.bbox,
                    suggestion=suggestions,
                    risk_lvl=risk_level,
                    damage_level=damage_level,
                    detection_source=detection_source
                )
                predictions.append(prediction)
            
            return FullResponse(predictions=predictions)
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return FullResponse(predictions=[])
        finally:
            os.remove(tmp_path)
            for cp in cropped_paths:
                if cp and os.path.exists(cp):
                    os.remove(cp)
    
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
            yolo_start = time.time()
            detections = self.yolo_detector.detect_objects(tmp_path)
            yolo_time = time.time() - yolo_start
            logger.info(f"YOLO detection completed in {yolo_time:.2f} seconds")
            
            # Log YOLO detections
            for det in detections:
                logger.info(f"YOLO detected: {det.category} with confidence {det.confidence:.3f}")
            
            # Filter overlapping detections
            filtered_detections = self._filter_overlapping_detections(detections)
            logger.info(f"Filtered {len(detections) - len(filtered_detections)} overlapping detections")
                
            return ObjectResponse(detections=filtered_detections)
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
