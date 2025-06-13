"""
Gemini AI Service Module
Handles validation using Google's Gemini
"""

import json
import logging
from typing import List, Dict, Any, Tuple
from PIL import Image

from src.config.settings import (
    GEMINI_AVAILABLE, GEMINI_MODEL, GEMINI_MAX_TOKENS, 
    GEMINI_TEMPERATURE, GEMINI_TOP_P
)
from src.utils.mappings import PRICE_CATEGORIES, is_valid_price_category
from src.models.response_models import ValidationResult

logger = logging.getLogger(__name__)


class GeminiService:
    """Gemini AI Service for validation"""
    
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
        mapped_category: str,
        extra_image_path: str = None,
        prompt_context: dict = None
    ) -> ValidationResult:
        """
        Validate YOLO detection using Gemini vision, with optional extra image and context.
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
            images = [Image.open(image_path)]
            if extra_image_path:
                images.append(Image.open(extra_image_path))
            # Build prompt
            if prompt_context:
                all_dets = prompt_context.get("all_detections", [])
                focus_bbox = prompt_context.get("focus_bbox")
                focus_label = prompt_context.get("focus_label")
                prompt = f"""
Analyze the provided images. The first image shows all detected objects with bounding boxes and labels. The second image (if present) is a crop of the object to focus on.

All detected objects:
{json.dumps(all_dets, ensure_ascii=False)}

Focus only on the object with label '{focus_label}' and bounding box {focus_bbox}.

YOLO Prediction: {yolo_prediction}
Mapped Category: {mapped_category}

Your task:
1. Identify the main e-waste object in the focus region
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
- Focus only on the object in the focus region
- Maximum {GEMINI_MAX_TOKENS} tokens in response
"""
            else:
                prompt = self._create_validation_prompt(yolo_prediction, mapped_category)
            # Gemini call
            response = self.model.generate_content(
                [prompt] + images,
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

    async def generate_description(self, image_path: str, category: str, extra_image_path: str = None, prompt_context: dict = None) -> str:
        """
        Generate concise description (10-15 words) using Gemini vision based on e-waste condition, with optional extra image and context.
        """
        if not self.is_available or self.model is None:
            return f"Perangkat elektronik {category.lower()}"
        try:
            images = [Image.open(image_path)]
            if extra_image_path:
                images.append(Image.open(extra_image_path))
            if prompt_context:
                all_dets = prompt_context.get("all_detections", [])
                focus_bbox = prompt_context.get("focus_bbox")
                focus_label = prompt_context.get("focus_label")
                prompt = f"""
Analisis kondisi e-waste pada gambar. Gambar pertama berisi semua deteksi dengan bounding box dan label. Gambar kedua (jika ada) adalah crop dari objek fokus.

Semua deteksi:
{json.dumps(all_dets, ensure_ascii=False)}

Fokus hanya pada objek dengan label '{focus_label}' dan bounding box {focus_bbox}.

Kategori: {category}

Buat deskripsi singkat (maksimal 15 kata) dalam Bahasa Indonesia, fokus pada kondisi aktual perangkat di area fokus.
"""
            else:
                prompt = f"""
Analisis kondisi e-waste ini dan buat deskripsi singkat (10-15 kata) dalam Bahasa Indonesia.
Kategori: {category}

Fokus pada:
1. Kondisi fisik (rusak/utuh/berkarat/dll)
2. Usia dan model (jika terlihat)
3. Komponen yang terlihat
4. Kerusakan spesifik (jika ada)

Deskripsi harus:
- Jelas dan informatif
- Maksimal 15 kata
- Dalam Bahasa Indonesia
- Fokus pada kondisi aktual perangkat
- Sertakan detail spesifik yang terlihat

Contoh format yang diharapkan:
"Laptop Dell Latitude dengan layar retak dan keyboard aus"
"Smartphone Samsung dengan casing retak dan layar bergaris"
"Monitor LG dengan bezel hitam dan port HDMI terlihat"
"""
            response = self.model.generate_content(
                [prompt] + images,
                generation_config={
                    "max_output_tokens": 3000,
                    "temperature": 0.3,
                    "top_p": 0.8
                }
            )
            if response.text:
                return response.text.strip()
            return f"Perangkat elektronik {category.lower()}"
        except Exception as e:
            logger.error(f"Gemini description error: {str(e)}")
            return f"Perangkat elektronik {category.lower()}"

    async def generate_suggestions(self, image_path: str, category: str, extra_image_path: str = None, prompt_context: dict = None) -> List[str]:
        """
        Generate disposal suggestions using Gemini vision based on e-waste condition, with optional extra image and context.
        """
        default_suggestions = [
            "Periksa panduan manufacturer",
            "Pisahkan komponen berbahaya",
            "Bawa ke pusat daur ulang e-waste"
        ]
        if not self.is_available or self.model is None:
            return default_suggestions
        try:
            images = [Image.open(image_path)]
            if extra_image_path:
                images.append(Image.open(extra_image_path))
            if prompt_context:
                all_dets = prompt_context.get("all_detections", [])
                focus_bbox = prompt_context.get("focus_bbox")
                focus_label = prompt_context.get("focus_label")
                prompt = f"""
Analisis kondisi e-waste pada gambar. Gambar pertama berisi semua deteksi dengan bounding box dan label. Gambar kedua (jika ada) adalah crop dari objek fokus.

Semua deteksi:
{json.dumps(all_dets, ensure_ascii=False)}

Fokus hanya pada objek dengan label '{focus_label}' dan bounding box {focus_bbox}.

Kategori: {category}

Buat 3 langkah penanganan spesifik untuk objek di area fokus, maksimal 10 kata per langkah, dalam Bahasa Indonesia.
"""
            else:
                prompt = f"""
Analisis kondisi e-waste ini dan buat 3 langkah penanganan dalam Bahasa Indonesia.
Kategori: {category}

Langkah-langkah harus:
1. Spesifik untuk kondisi perangkat yang terlihat
2. Fokus pada keamanan dan lingkungan
3. Mudah diikuti
4. Maksimal 10 kata per langkah
5. Dalam Bahasa Indonesia

Format:
1. [Langkah pertama]
2. [Langkah kedua]
3. [Langkah ketiga]

Contoh untuk perangkat rusak:
1. Pisahkan komponen yang rusak dengan hati-hati
2. Simpan bagian yang masih berfungsi
3. Bawa ke pusat daur ulang e-waste

Contoh untuk perangkat utuh:
1. Backup dan hapus data dengan aman
2. Lepas komponen yang bisa dilepas
3. Bawa ke pusat daur ulang e-waste
"""
            response = self.model.generate_content(
                [prompt] + images,
                generation_config={
                    "max_output_tokens": 3000,
                    "temperature": 0.3,
                    "top_p": 0.8
                }
            )
            if response.text:
                # Parse numbered list
                suggestions = []
                for line in response.text.strip().split('\n'):
                    if line.strip() and any(line.strip().startswith(str(i)) for i in range(1, 10)):
                        suggestion = line.split('.', 1)[1].strip()
                        suggestions.append(suggestion)
                # Fill with defaults if fewer than 3
                while len(suggestions) < 3:
                    suggestions.append(default_suggestions[len(suggestions)])
                return suggestions[:3]
            return default_suggestions
        except Exception as e:
            logger.error(f"Gemini suggestions error: {str(e)}")
            return default_suggestions
    
    async def analyze_damage_level(self, image_path: str, category: str, extra_image_path: str = None, prompt_context: dict = None) -> Tuple[int, str]:
        """
        Analyze damage level of e-waste using Gemini vision.
        Returns damage level (1-5) and detailed analysis.
        """
        if not self.is_available or self.model is None:
            return 3, "Damage analysis unavailable"
        
        try:
            images = [Image.open(image_path)]
            if extra_image_path:
                images.append(Image.open(extra_image_path))
            
            prompt = f"""
Analyze the physical condition of this e-waste item.
Category: {category}

Assess the following aspects:
1. Physical damage (scratches, dents, cracks)
2. Component condition (missing parts, loose connections)
3. Wear and tear (age-related deterioration)
4. Functionality indicators (power ports, buttons, screens)
5. Overall appearance

Rate the damage level from 1 to 5:
1 = Excellent condition (like new, minimal wear)
2 = Good condition (minor wear, fully functional)
3 = Fair condition (visible wear, some damage)
4 = Poor condition (significant damage, may not function)
5 = Severe damage (extensive damage, non-functional)

Respond in this exact JSON format:
{{
    "damage_level": 1-5,
    "analysis": "Detailed analysis of the damage",
    "key_issues": ["List of main issues found"]
}}
"""
            response = self.model.generate_content(
                [prompt] + images,
                generation_config={
                    "max_output_tokens": GEMINI_MAX_TOKENS,
                    "temperature": GEMINI_TEMPERATURE,
                    "top_p": GEMINI_TOP_P
                }
            )
            
            if not response.text:
                return 3, "Damage analysis failed - empty response"
            
            # Parse response
            try:
                result = json.loads(response.text.strip())
                damage_level = int(result.get("damage_level", 3))
                analysis = result.get("analysis", "No detailed analysis available")
                return damage_level, analysis
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse damage analysis response: {e}")
                return 3, "Damage analysis parsing failed"
                
        except Exception as e:
            logger.error(f"Damage analysis error: {str(e)}")
            return 3, f"Damage analysis error: {str(e)}"
    
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
4. Assess the physical condition and damage level (1-5)

Respond in this exact JSON format:
{{
    "object_identified": "description of what you see",
    "is_category_correct": true/false,
    "correct_category": "category name from the list or null if correct",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "damage_level": 1-5,
    "damage_analysis": "brief analysis of physical condition"
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
