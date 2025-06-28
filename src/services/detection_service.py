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
from PIL import Image

from src.config.settings import (
    YOLO_AVAILABLE, YOLO_MODEL_PATH, LOW_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD, GEMINI_TIMEOUT, GEMINI_ENABLE_CROSS_VALIDATION
)
from src.models.response_models import (
    Detection, FullPrediction, FullResponse,
    ValidationResult, ObjectResponse, PriceResponse
)
from src.models.yolo_detector import YOLODetector
from src.models.price_predictor import PricePredictor
from src.services.gemini_service import GeminiService
from src.utils.helpers import (
    generate_unique_id, calculate_risk_level, create_fallback_prediction,
    safe_execute, log_execution_time
)
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
        return safe_execute(
            lambda: self._check_image_size(crop_path, min_size),
            False,
            f"Failed to validate crop image {crop_path}"
        )
    
    def _check_image_size(self, crop_path: str, min_size: int) -> bool:
        """Helper method to check image size"""
        with Image.open(crop_path) as img:
            width, height = img.size
            return width >= min_size and height >= min_size

    @log_execution_time("Complete image processing")
    async def process_image_complete(
        self, 
        image_bytes: bytes,
        confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD
    ) -> FullResponse:
        """
        Process image with complete pipeline including validation and pricing
        OPTIMIZED for speed with parallel processing
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
            
            if not detections:
                logger.info("No detections found")
                return FullResponse(predictions=[])
            
            # Filter overlapping detections
            filtered_detections = self._filter_overlapping_detections(detections)
            logger.info(f"Processing {len(filtered_detections)} detections after filtering")
            
            # Process all detections in parallel for maximum speed
            prediction_tasks = []
            for det in filtered_detections:
                task = self._process_single_detection_optimized(
                    det, tmp_path, filtered_detections
                )
                prediction_tasks.append(task)
            
            # Execute all detection processing in parallel
            gemini_start = time.time()
            predictions = await asyncio.gather(*prediction_tasks, return_exceptions=True)
            gemini_time = time.time() - gemini_start
            logger.info(f"All Gemini processing completed in {gemini_time:.2f} seconds")
            
            # Filter out any failed predictions and log errors
            valid_predictions = []
            for i, pred in enumerate(predictions):
                if isinstance(pred, Exception):
                    logger.error(f"Detection {i} failed: {pred}")
                else:
                    valid_predictions.append(pred)
            
            return FullResponse(predictions=valid_predictions)
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return FullResponse(predictions=[])
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            for cp in cropped_paths:
                if cp and os.path.exists(cp):
                    os.remove(cp)
    
    async def _process_single_detection_optimized(
        self, 
        det: Detection, 
        image_path: str, 
        all_detections: List[Detection]
    ) -> FullPrediction:
        """Process a single detection with optimized parallel Gemini calls"""
        # Keep original category for display
        display_category = det.category
        
        # Get mapped category for price prediction
        price_category = get_mapped_category(det.category)
        logger.info(f"Processing '{display_category}' -> '{price_category}'")
        
        # For critical mappings, add extra protection
        if display_category.lower() == "phone" and price_category != "Handphone":
            logger.error(f"CRITICAL: Phone mapping error detected! Expected 'Handphone', got '{price_category}'")
            price_category = "Handphone"  # Force correct mapping
            
        # Crop the detection
        cropped_path = self._save_cropped_bbox(image_path, det.bbox, det.category)
        
        try:
            # Check crop validity
            if not self._is_valid_crop(cropped_path):
                logger.warning(f"Small crop for {display_category}, using fallback processing")
                
                # For small crops, do minimal processing
                price = safe_execute(
                    self.price_predictor.predict_price,
                    None,
                    f"Price prediction failed for {price_category}",
                    price_category
                )
                
                return create_fallback_prediction(
                    display_category, det.confidence, det.bbox, price, "YOLO (small crop)"
                )
            
            # Prepare context for Gemini
            prompt_context = {
                "all_detections": [
                    {"category": d.category, "confidence": d.confidence, "bbox": d.bbox} 
                    for d in all_detections
                ],
                "focus_bbox": det.bbox,
                "focus_label": display_category
            }
            
            # Use batch processing for all Gemini operations
            batch_result = await safe_execute(
                self.gemini_service.process_batch_analysis,
                self._get_default_batch_result(display_category, price_category),
                f"Gemini batch analysis failed for {display_category}",
                cropped_path,
                display_category,
                yolo_prediction=det.category,
                mapped_category=price_category,
                extra_image_path=None,
                prompt_context=prompt_context
            )
            
            # Extract results from batch processing
            validation = batch_result.get("validation")
            description = batch_result.get("description", f"Perangkat elektronik {display_category.lower()}")
            suggestions = batch_result.get("suggestions", [
                "Periksa panduan manufacturer",
                "Pisahkan komponen berbahaya", 
                "Bawa ke pusat daur ulang e-waste"
            ])
            damage_level = batch_result.get("damage_level")
            
            # Determine final categories based on validation
            final_display_category, final_price_category, detection_source = self._determine_final_categories(
                validation, display_category, price_category, description
            )
            
            # CRITICAL SAFEGUARD: For Phone detections, ensure we don't end up with wrong categories
            if display_category.lower() == "phone":
                if final_price_category not in ["Handphone", "Telefon"]:
                    logger.error(f"CRITICAL: Phone validation resulted in wrong category '{final_price_category}', forcing to 'Handphone'")
                    final_price_category = "Handphone"
                    final_display_category = "Phone"
                    detection_source = "YOLO (corrected)"
                    
                # If it's Telefon but original was Phone, validate this is correct
                if final_price_category == "Telefon":
                    logger.warning(f"Phone -> Telefon conversion detected. Checking if this is a walkie-talkie...")
                    # For now, prefer Handphone for Phone detections unless there's strong evidence
                    if "walkie" not in description.lower() and "radio" not in description.lower():
                        logger.info("No walkie-talkie evidence found, keeping as Handphone")
                        final_price_category = "Handphone"
                        final_display_category = "Phone"
                        detection_source = "YOLO (corrected)"
            
            # Price prediction and risk calculation
            price = safe_execute(
                self.price_predictor.predict_price,
                None,
                f"Price prediction failed for {final_price_category}",
                final_price_category
            )
            
            # Log the final result for debugging
            logger.info(f"Final prediction: {display_category} -> {final_display_category}, price_category: {final_price_category}, price: {price}, source: {detection_source}")
            
            risk_level = calculate_risk_level(final_price_category, det.confidence)
            
            return FullPrediction(
                id=generate_unique_id(),
                category=final_display_category,
                confidence=det.confidence,
                regression_result=price,
                description=description,
                bbox=det.bbox,
                suggestion=suggestions,
                risk_lvl=risk_level,
                damage_level=damage_level,
                detection_source=detection_source
            )
            
        except Exception as e:
            logger.error(f"Error processing detection {display_category}: {str(e)}")
            # Return basic prediction on error
            price = safe_execute(
                self.price_predictor.predict_price,
                None,
                f"Fallback price prediction failed for {price_category}",
                price_category
            )
            
            return create_fallback_prediction(
                display_category, det.confidence, det.bbox, price, "YOLO (error fallback)"
            )
        finally:
            # Cleanup cropped image
            if cropped_path and os.path.exists(cropped_path):
                os.remove(cropped_path)
    
    def _get_default_batch_result(self, display_category: str, price_category: str) -> Dict[str, Any]:
        """Get default batch result for fallback scenarios"""
        return {
            "validation": ValidationResult(
                is_valid=True,
                final_category=price_category,
                detection_source="YOLO",
                gemini_feedback="Batch processing fallback"
            ),
            "description": f"Perangkat elektronik {display_category.lower()}",
            "suggestions": [
                "Periksa panduan manufacturer",
                "Pisahkan komponen berbahaya",
                "Bawa ke pusat daur ulang e-waste"
            ],
            "damage_level": None
        }
    
    def _determine_final_categories(
        self, 
        validation: Optional[ValidationResult], 
        display_category: str, 
        price_category: str, 
        description: str
    ) -> Tuple[str, str, str]:
        """Determine final categories and detection source based on validation results"""
        final_display_category = display_category
        final_price_category = price_category
        detection_source = "YOLO"
        
        if validation and validation.is_valid:
            if validation.final_category and validation.final_category != price_category:
                final_price_category = validation.final_category
                final_display_category = validation.final_category
                logger.info(f"Gemini corrected: '{display_category}' -> '{final_display_category}'")
            detection_source = validation.detection_source
        elif validation and not validation.is_valid:
            logger.warning(f"Gemini rejected detection: {validation.gemini_feedback}")
            detection_source = "Rejected"
        else:
            # Fallback: Use cross-validation with description if Gemini validation didn't work
            if GEMINI_ENABLE_CROSS_VALIDATION:
                cross_validated_category = safe_execute(
                    self.gemini_service.cross_validate_category,
                    price_category,
                    f"Cross-validation failed for {display_category}",
                    description, display_category, price_category
                )
                if cross_validated_category != price_category:
                    final_price_category = cross_validated_category
                    final_display_category = cross_validated_category
                    detection_source = "Cross-validated"
                    logger.info(f"Cross-validation corrected: '{display_category}' -> '{final_display_category}'")
        
        return final_display_category, final_price_category, detection_source
    
    @log_execution_time("YOLO detection only")
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
