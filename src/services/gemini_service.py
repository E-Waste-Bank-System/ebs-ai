"""
Gemini AI Service Module
Handles RAG (Retrieval-Augmented Generation) and validation using Google's Gemini
"""

import json
import logging
from typing import List, Dict, Any
from PIL import Image

from src.config.settings import (
    GEMINI_AVAILABLE, GEMINI_MODEL, GEMINI_MAX_TOKENS, 
    GEMINI_TEMPERATURE, GEMINI_TOP_P
)
from src.utils.mappings import PRICE_CATEGORIES, is_valid_price_category
from src.models.response_models import ValidationResult

logger = logging.getLogger(__name__)


class GeminiService:
    """Gemini AI Service for RAG and validation"""
    
    def __init__(self):
        self.model = None
        self.is_available = GEMINI_AVAILABLE
        
        if self.is_available:
            try:
                import google.generativeai as genai
                self.model = genai.GenerativeModel(GEMINI_MODEL)
                logger.info("Gemini service initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {str(e)}")
                self.is_available = False
    
    async def validate_detection(
        self, 
        image_path: str, 
        yolo_prediction: str, 
        mapped_category: str
    ) -> ValidationResult:
        """
        Validate YOLO detection using Gemini vision
        
        Args:
            image_path: Path to image file
            yolo_prediction: Original YOLO prediction
            mapped_category: Category mapped from YOLO to price model
            
        Returns:
            ValidationResult with validation outcome
        """
        if not self.is_available or self.model is None:
            logger.warning("Gemini not available, using YOLO prediction")
            return ValidationResult(
                is_valid=True,
                final_category=mapped_category,
                detection_source="YOLO",
                gemini_feedback="Gemini validation unavailable"
            )
        
        try:
            # Load and process image
            image = Image.open(image_path)
            
            # Create validation prompt
            prompt = self._create_validation_prompt(yolo_prediction, mapped_category)
            
            # Get Gemini response
            response = self.model.generate_content(
                [prompt, image],
                generation_config={
                    "max_output_tokens": GEMINI_MAX_TOKENS,
                    "temperature": GEMINI_TEMPERATURE,
                    "top_p": GEMINI_TOP_P
                }
            )
            
            if not response.text:
                logger.warning("Empty Gemini response")
                return ValidationResult(
                    is_valid=True,
                    final_category=mapped_category,
                    detection_source="YOLO",
                    gemini_feedback="Gemini validation failed - empty response"
                )
            
            # Parse and process response
            return self._process_validation_response(
                response.text, mapped_category, yolo_prediction
            )
            
        except Exception as e:
            logger.error(f"Gemini validation error: {str(e)}")
            return ValidationResult(
                is_valid=True,
                final_category=mapped_category,
                detection_source="YOLO",
                gemini_feedback=f"Gemini validation error: {str(e)}"
            )
    
    async def generate_disposal_summary(self, detected_items: List[str]) -> str:
        """
        Generate RAG summary for disposal guidance
        
        Args:
            detected_items: List of detected e-waste items
            
        Returns:
            Disposal guidance summary in Indonesian
        """
        if not self.is_available or not detected_items:
            return "Silakan bawa e-waste ke fasilitas daur ulang bersertifikat."
        
        try:
            prompt = f"""
            Berikan panduan singkat disposal untuk e-waste items berikut: {', '.join(detected_items)}
            
            Fokus pada:
            1. Cara disposal yang aman dan ramah lingkungan
            2. Persiapan sebelum disposal (hapus data, lepas battery, dll)
            3. Tempat disposal yang tepat di Indonesia
            4. Dampak lingkungan jika tidak di-disposal dengan benar
            
            Jawab dalam bahasa Indonesia, maksimal 200 kata.
            """
            
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": GEMINI_MAX_TOKENS,
                    "temperature": GEMINI_TEMPERATURE,
                    "top_p": GEMINI_TOP_P
                }
            )
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"RAG generation error: {str(e)}")
            return "Silakan disposal e-waste items ke fasilitas daur ulang bersertifikat untuk melindungi lingkungan."
    
    def _create_validation_prompt(self, yolo_prediction: str, mapped_category: str) -> str:
        """Create validation prompt for Gemini"""
        return f"""
Analyze this image and verify if the detected object matches the predicted category.

YOLO Prediction: {yolo_prediction}
Mapped Category: {mapped_category}

Your task:
1. Identify the main e-waste object in this image
2. Determine if it matches the mapped category: "{mapped_category}"
3. If incorrect, suggest the best matching category from this list: {', '.join(PRICE_CATEGORIES)}

Respond in this exact JSON format:
{{
    "object_identified": "description of what you see",
    "is_category_correct": true/false,
    "correct_category": "category name from the list or null if correct",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}

Important: 
- Only use categories from the provided list
- Be precise about object identification
- Consider the context and main object focus
- Maximum {GEMINI_MAX_TOKENS} tokens in response
"""
    
    def _process_validation_response(
        self, 
        response_text: str, 
        mapped_category: str, 
        yolo_prediction: str
    ) -> ValidationResult:
        """Process and parse Gemini validation response"""
        try:
            # Clean the response text to handle markdown code blocks
            cleaned_text = response_text.strip()
            if cleaned_text.startswith('```json'):
                cleaned_text = cleaned_text[7:]  # Remove ```json
            if cleaned_text.endswith('```'):
                cleaned_text = cleaned_text[:-3]  # Remove ```
            cleaned_text = cleaned_text.strip()
            
            gemini_result = json.loads(cleaned_text)
            
            is_correct = gemini_result.get("is_category_correct", True)
            suggested_category = gemini_result.get("correct_category")
            reasoning = gemini_result.get("reasoning", "")
            
            if is_correct:
                return ValidationResult(
                    is_valid=True,
                    final_category=mapped_category,
                    detection_source="YOLO",
                    gemini_feedback=f"Category validated by Gemini: {reasoning}"
                )
            elif suggested_category and is_valid_price_category(suggested_category):
                return ValidationResult(
                    is_valid=True,
                    final_category=suggested_category,
                    detection_source="Gemini Interfered",
                    gemini_feedback=f"Corrected from {mapped_category} to {suggested_category}: {reasoning}"
                )
            else:
                return ValidationResult(
                    is_valid=False,
                    final_category=None,
                    detection_source="Rejected",
                    gemini_feedback=f"No valid e-waste detected: {reasoning}"
                )
                
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse Gemini JSON response: {response_text}")
            return ValidationResult(
                is_valid=True,
                final_category=mapped_category,
                detection_source="YOLO",
                gemini_feedback="Gemini response parsing failed - using YOLO prediction"
            )
    
    def is_service_available(self) -> bool:
        """Check if Gemini service is available"""
        return self.is_available
