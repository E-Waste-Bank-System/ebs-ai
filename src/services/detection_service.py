"""
Detection Service Module
Handles e-waste detection pipeline: YOLO Detection → Gemini Validation → YOLO-to-Price Mapping → Price Prediction

Flow:
1. YOLO detects objects in 37 YOLO categories
2. Gemini validates/corrects YOLO predictions (works with YOLO categories)
3. Validated YOLO categories are mapped to 33 price categories
4. Price prediction uses final price categories
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
    """
    Main detection service orchestrating the complete e-waste detection pipeline
    
    Pipeline: YOLO Detection → Gemini Validation → YOLO-to-Price Mapping → Price Prediction
    """
    
    def __init__(self):
        # Initialize all components
        self.yolo_detector = YOLODetector()
        self.price_predictor = PricePredictor()
        self.gemini_service = GeminiService()
        
        # Load models
        self.yolo_loaded = self.yolo_detector.load_model()
        self.price_loaded = self.price_predictor.load_models()
        
        logger.info(f"Detection service initialized - YOLO: {self.yolo_loaded}, Price: {self.price_loaded}, Gemini: {self.gemini_service.is_service_available()}")
    
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
        Crop the image to the bounding box with 20% padding for context and save to a temp file.
        
        Args:
            image_path: Path to source image
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            label: Label for the crop (used in filename)
            
        Returns:
            Path to the cropped image file
        """
        image = Image.open(image_path).convert("RGB")
        img_width, img_height = image.size
        
        # Extract bbox coordinates
        x1, y1, x2, y2 = bbox
        
        # Add 20% padding for context
        width = x2 - x1
        height = y2 - y1
        padding_x = width * 0.2
        padding_y = height * 0.2
        
        # Calculate padded coordinates, ensuring they stay within image bounds
        padded_x1 = max(0, x1 - padding_x)
        padded_y1 = max(0, y1 - padding_y)
        padded_x2 = min(img_width, x2 + padding_x)
        padded_y2 = min(img_height, y2 + padding_y)
        
        # Ensure bbox is int for cropping
        bbox_int = [int(padded_x1), int(padded_y1), int(padded_x2), int(padded_y2)]
        cropped = image.crop(bbox_int)
        
        temp_cropped = tempfile.NamedTemporaryFile(suffix=f'_{label}.jpg', delete=False)
        cropped.save(temp_cropped.name)
        logger.info(f"Cropped image for '{label}' with 20% padding saved: {temp_cropped.name} (size: {cropped.size})")
        return temp_cropped.name

    def _is_valid_crop(self, crop_path: str, min_size: int = 50) -> bool:
        """
        Check if the cropped image is large enough for Gemini analysis.
        
        Args:
            crop_path: Path to cropped image
            min_size: Minimum width/height in pixels (default 50x50)
            
        Returns:
            True if crop is valid for analysis
        """
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
        Process image with complete pipeline: YOLO → Gemini Validation → Mapping → Pricing
        OPTIMIZED for speed with parallel processing
        
        Args:
            image_bytes: Input image as bytes
            confidence_threshold: Minimum confidence for YOLO detections
            
        Returns:
            FullResponse with complete predictions
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
            # Step 1: YOLO Detection with timing and save annotated image
            yolo_start = time.time()
            detections = self.yolo_detector.detect_objects(tmp_path, save_annotated_path=annotated_path)
            yolo_time = time.time() - yolo_start
            logger.info(f"YOLO detection completed in {yolo_time:.2f} seconds")
            
            if not detections:
                logger.info("No detections found")
                return FullResponse(predictions=[])
            
            # Filter overlapping detections
            filtered_detections = self._filter_overlapping_detections(detections)
            logger.info(f"Processing {len(filtered_detections)} detections after filtering (removed {len(detections) - len(filtered_detections)} overlaps)")
            
            # Step 2-4: Process all detections in parallel through the complete pipeline
            prediction_tasks = []
            for det in filtered_detections:
                task = self._process_single_detection_complete_pipeline(
                    det, tmp_path, filtered_detections
                )
                prediction_tasks.append(task)
            
            # Execute all detection processing in parallel
            pipeline_start = time.time()
            predictions = await asyncio.gather(*prediction_tasks, return_exceptions=True)
            pipeline_time = time.time() - pipeline_start
            logger.info(f"Complete pipeline processing completed in {pipeline_time:.2f} seconds")
            
            # Filter out failed predictions, rejections, and log results
            valid_predictions = []
            rejected_count = 0
            error_count = 0
            
            for i, pred in enumerate(predictions):
                if isinstance(pred, Exception):
                    logger.error(f"Detection {i} failed: {pred}")
                    error_count += 1
                elif pred is None:
                    # Detection was rejected by Gemini
                    rejected_count += 1
                else:
                    valid_predictions.append(pred)
            
            logger.info(f"Pipeline results: {len(valid_predictions)} valid, {rejected_count} rejected, {error_count} errors")
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
    
    async def _process_single_detection_complete_pipeline(
        self, 
        det: Detection, 
        image_path: str, 
        all_detections: List[Detection]
    ) -> Optional[FullPrediction]:
        """
        Process a single detection through the complete pipeline:
        YOLO Detection → Gemini Validation → YOLO-to-Price Mapping → Price Prediction
        
        Args:
            det: YOLO Detection object
            image_path: Path to source image
            all_detections: List of all detections for context
            
        Returns:
            FullPrediction with complete analysis, or None if detection is rejected
        """
        # Step 1: YOLO Detection (already done - we have det)
        yolo_category = det.category
        yolo_confidence = det.confidence
        logger.info(f"Processing YOLO detection: '{yolo_category}' (confidence: {yolo_confidence:.3f})")
        
        # Crop the detection for Gemini analysis
        cropped_path = self._save_cropped_bbox(image_path, det.bbox, yolo_category)
        
        try:
            # Check crop validity
            if not self._is_valid_crop(cropped_path):
                logger.warning(f"Small crop for {yolo_category}, using fallback processing")
                
                # For small crops, skip Gemini and go straight to mapping and pricing
                validated_yolo_category = yolo_category
                detection_source = "YOLO (small crop)"
                
                # Map to price category and predict price
                price_category = get_mapped_category(validated_yolo_category)
                price = safe_execute(
                    self.price_predictor.predict_price,
                    None,
                    f"Price prediction failed for {price_category}",
                    price_category
                )
                
                return create_fallback_prediction(
                    yolo_category, yolo_confidence, det.bbox, price, detection_source
                )
            
            # Prepare context for Gemini
            prompt_context = {
                "all_detections": [
                    {"category": d.category, "confidence": d.confidence, "bbox": d.bbox} 
                    for d in all_detections
                ],
                "focus_bbox": det.bbox,
                "focus_label": yolo_category
            }
            
            # Step 2: Gemini Validation (validates YOLO categories)
            batch_result = await safe_execute(
                self.gemini_service.process_batch_analysis,
                self._get_default_batch_result(yolo_category),
                f"Gemini batch analysis failed for {yolo_category}",
                cropped_path,
                yolo_category,  # For content generation
                yolo_prediction=yolo_category,  # For validation
                yolo_confidence=yolo_confidence,
                extra_image_path=None,
                prompt_context=prompt_context
            )
            
            # Extract results from batch processing
            validation = batch_result.get("validation")
            description = batch_result.get("description", f"Perangkat elektronik {yolo_category.lower()}")
            suggestions = batch_result.get("suggestions", [
                "Periksa panduan manufacturer",
                "Pisahkan komponen berbahaya", 
                "Bawa ke pusat daur ulang e-waste"
            ])
            damage_level = batch_result.get("damage_level")
            
            # Step 3: Determine validated YOLO category
            validated_yolo_category, detection_source = self._determine_validated_yolo_category(
                validation, yolo_category, description
            )
            
            # Check if detection was rejected - if so, return None to exclude from results
            if detection_source == "Rejected":
                logger.info(f"Detection rejected by Gemini: {yolo_category} - excluding from results")
                return None
            
            # Step 4: YOLO-to-Price Mapping (after Gemini validation)
            price_category = get_mapped_category(validated_yolo_category)
            logger.info(f"YOLO→Price mapping: '{validated_yolo_category}' → '{price_category}'")
            
            # Step 5: Price Prediction
            price = safe_execute(
                self.price_predictor.predict_price,
                None,
                f"Price prediction failed for {price_category}",
                price_category
            )
            
            # Calculate risk level based on final categories
            risk_level = calculate_risk_level(validated_yolo_category, yolo_confidence)
            
            # Log the complete pipeline result
            logger.info(f"Complete pipeline: {yolo_category} → {validated_yolo_category} → {price_category}, price: {price}, source: {detection_source}")
            
            return FullPrediction(
                id=generate_unique_id(),
                category=validated_yolo_category,  # Display the validated YOLO category
                confidence=yolo_confidence,
                regression_result=price,
                description=description,
                bbox=det.bbox,
                suggestion=suggestions,
                risk_lvl=risk_level,
                damage_level=damage_level,
                detection_source=detection_source
            )
            
        except Exception as e:
            logger.error(f"Error processing detection {yolo_category}: {str(e)}")
            # Return basic prediction on error
            price_category = get_mapped_category(yolo_category)
            price = safe_execute(
                self.price_predictor.predict_price,
                None,
                f"Fallback price prediction failed for {price_category}",
                price_category
            )
            
            return create_fallback_prediction(
                yolo_category, yolo_confidence, det.bbox, price, "YOLO (error fallback)"
            )
        finally:
            # Cleanup cropped image
            if cropped_path and os.path.exists(cropped_path):
                os.remove(cropped_path)
    
    def _get_default_batch_result(self, yolo_category: str) -> Dict[str, Any]:
        """Get default batch result for fallback scenarios"""
        return {
            "validation": ValidationResult(
                is_valid=True,
                final_category=yolo_category,
                detection_source="YOLO",
                gemini_feedback="Batch processing fallback"
            ),
            "description": f"Perangkat elektronik {yolo_category.lower()} terdeteksi dalam kondisi tidak dapat dianalisis",  # 10-15 words
            "suggestions": [
                "Periksa panduan dari manufacturer resmi",        # 6 words
                "Pisahkan komponen berbahaya dengan hati hati",  # 7 words
                "Bawa ke pusat daur ulang terdekat"              # 7 words
            ],
            "damage_level": None
        }
    
    def _determine_validated_yolo_category(
        self, 
        validation: Optional[ValidationResult], 
        original_yolo_category: str, 
        description: str
    ) -> Tuple[str, str]:
        """
        Determine the final validated YOLO category and detection source.
        
        Args:
            validation: Gemini validation result
            original_yolo_category: Original YOLO prediction
            description: Generated description for cross-validation
            
        Returns:
            Tuple of (validated_yolo_category, detection_source)
        """
        validated_yolo_category = original_yolo_category
        detection_source = "YOLO"
        
        if validation and validation.is_valid:
            if validation.final_category and validation.final_category != original_yolo_category:
                validated_yolo_category = validation.final_category
                logger.info(f"Gemini corrected YOLO: '{original_yolo_category}' → '{validated_yolo_category}'")
            detection_source = validation.detection_source
        elif validation and not validation.is_valid:
            logger.warning(f"Gemini rejected detection: {validation.gemini_feedback}")
            detection_source = "Rejected"
        else:
            # Fallback: Use cross-validation with description if Gemini validation didn't work
            if GEMINI_ENABLE_CROSS_VALIDATION:
                cross_validated_category = safe_execute(
                    self.gemini_service.cross_validate_category,
                    validated_yolo_category,
                    f"Cross-validation failed for {original_yolo_category}",
                    description, original_yolo_category, validated_yolo_category
                )
                if cross_validated_category != validated_yolo_category:
                    validated_yolo_category = cross_validated_category
                    detection_source = "Cross-validated"
                    logger.info(f"Cross-validation corrected: '{original_yolo_category}' → '{validated_yolo_category}'")
        
        return validated_yolo_category, detection_source
    
    @log_execution_time("YOLO detection only")
    async def detect_objects_only(self, image_bytes: bytes) -> ObjectResponse:
        """
        YOLO detection only - no validation or pricing
        
        Args:
            image_bytes: Image file bytes
            
        Returns:
            ObjectResponse with detected objects in YOLO categories
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
        Price prediction only - expects price model categories
        
        Args:
            category: Price model category name (33 categories)
            
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
        """Get list of supported price categories (33 categories)"""
        if self.price_loaded:
            return self.price_predictor.get_supported_categories()
        return []
    
    def get_system_status(self) -> dict:
        """Get system component status"""
        return {
            "yolo_available": self.yolo_loaded,
            "yolo_categories_count": 37,
            "price_prediction_available": self.price_loaded,
            "price_categories_count": len(self.get_supported_categories()) if self.price_loaded else 0,
            "gemini_available": self.gemini_service.is_service_available(),
            "pipeline_flow": "YOLO Detection → Gemini Validation → YOLO-to-Price Mapping → Price Prediction",
            "rejection_handling": "Rejected detections are excluded from results",
            "validation_types": ["YOLO", "Gemini Corrected", "Cross-validated", "Rejected"]
        }
