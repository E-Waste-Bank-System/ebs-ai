"""
Gemini AI Service Module
Handles validation using Google's Gemini with performance optimizations
"""

import json
import logging
import asyncio
from typing import List, Dict, Any, Tuple
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from src.config.settings import (
    GEMINI_AVAILABLE, GEMINI_MODEL, GEMINI_MAX_TOKENS, 
    GEMINI_TEMPERATURE, GEMINI_TOP_P, GEMINI_MAX_WORKERS,
    GEMINI_TIMEOUT, GEMINI_REQUEST_TIMEOUT
)
from src.utils.mappings import PRICE_CATEGORIES, is_valid_price_category
from src.models.response_models import ValidationResult

logger = logging.getLogger(__name__)


class GeminiService:
    """Gemini AI Service for validation with performance optimizations"""
    
    def __init__(self):
        self.model = None
        self.is_available = GEMINI_AVAILABLE
        self.executor = ThreadPoolExecutor(max_workers=GEMINI_MAX_WORKERS)
        
        # Optimized generation config
        self.generation_config = {
            "max_output_tokens": GEMINI_MAX_TOKENS,
            "temperature": GEMINI_TEMPERATURE,
            "top_p": GEMINI_TOP_P,
            "candidate_count": 1,  # Only generate one candidate for speed
        }
        
        if self.is_available:
            try:
                import google.generativeai as genai
                self.model = genai.GenerativeModel(GEMINI_MODEL)
                logger.info(f"Gemini service initialized with {GEMINI_MODEL} and {GEMINI_MAX_WORKERS} workers")
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

YOLO detected: {yolo_prediction}

Your task:
1. Identify what electronic/e-waste object you see in the focus region
2. Determine if this is valid e-waste (any electronic device, battery, or electronic component)
3. Find the best matching category from this list: {', '.join(PRICE_CATEGORIES)}

Respond in this exact JSON format:
{{
    "object_identified": "description of what you see",
    "is_valid_ewaste": true/false,
    "best_category": "category name from the list or null if not e-waste",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}

Important:
- Accept any electronic device as valid e-waste (phones, laptops, batteries, chargers, etc.)
- Choose the most appropriate category for pricing/analysis purposes
- Focus only on the object in the focus region
- Maximum {GEMINI_MAX_TOKENS} tokens in response
"""
            else:
                prompt = self._create_validation_prompt(yolo_prediction, mapped_category)
            # Gemini call
            response = await self._call_gemini_with_timeout(prompt, images)
            if not response:
                logger.warning("Empty Gemini validation response")
                return ValidationResult(
                    is_valid=True,
                    final_category=mapped_category,
                    detection_source="YOLO",
                    gemini_feedback="Gemini validation failed - empty response"
                )
            
            # Log the raw response for debugging
            logger.debug(f"Raw Gemini validation response: {response[:200]}...")
            
            return self._process_validation_response(
                response, mapped_category, yolo_prediction
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
Describe this e-waste in Indonesian (max 15 words):
Category: {category}

Focus: condition, brand/model if visible, damage.
Example: "Laptop Dell rusak layar retak keyboard aus"
"""
            else:
                prompt = f"""
Describe this e-waste in Indonesian (max 15 words):
Category: {category}

Focus: condition, brand/model if visible, damage.
Example: "Laptop Dell rusak layar retak keyboard aus"
"""
            response = await self._call_gemini_with_timeout(prompt, images)
            if response:
                description = response.strip()
                # Clean up the description
                if description and len(description) > 5:
                    return description
                else:
                    logger.warning("Empty or very short description from Gemini")
            else:
                logger.warning("No response text from Gemini for description")
            
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
Create 3 disposal steps for this e-waste in Indonesian:
Category: {category}

Format:
1. [step 1 - max 8 words]
2. [step 2 - max 8 words] 
3. [step 3 - max 8 words]

Focus on safety and recycling.
"""
            else:
                prompt = f"""
Create 3 disposal steps for this e-waste in Indonesian:
Category: {category}

Format:
1. [step 1 - max 8 words]
2. [step 2 - max 8 words] 
3. [step 3 - max 8 words]

Focus on safety and recycling.
"""
            response = await self._call_gemini_with_timeout(prompt, images)
            if response:
                # Parse numbered list
                suggestions = []
                lines = response.strip().split('\n')
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check if line starts with a number
                    if any(line.startswith(f"{i}.") for i in range(1, 10)):
                        # Split only on the first dot and check if there's content after it
                        parts = line.split('.', 1)
                        if len(parts) > 1 and parts[1].strip():
                            suggestion = parts[1].strip()
                            if suggestion:  # Only add non-empty suggestions
                                suggestions.append(suggestion)
                
                # If we didn't find numbered suggestions, try to parse without numbers
                if not suggestions:
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('#') and len(line) > 5:  # Skip headers and very short lines
                            # Remove common prefixes
                            for prefix in ['- ', '• ', '* ', '1. ', '2. ', '3. ']:
                                if line.startswith(prefix):
                                    line = line[len(prefix):].strip()
                                    break
                            if line and len(line) > 5:
                                suggestions.append(line)
                
                # Ensure we have exactly 3 suggestions
                if len(suggestions) > 3:
                    suggestions = suggestions[:3]
                elif len(suggestions) < 3:
                    # Fill with defaults if fewer than 3
                    while len(suggestions) < 3:
                        if len(suggestions) < len(default_suggestions):
                            suggestions.append(default_suggestions[len(suggestions)])
                        else:
                            suggestions.append("Bawa ke pusat daur ulang e-waste")
                
                return suggestions[:3]
            return default_suggestions
        except Exception as e:
            logger.error(f"Gemini suggestions error: {str(e)}")
            logger.debug(f"Gemini suggestions - Raw response: {getattr(response, 'text', 'No response') if 'response' in locals() else 'No response object'}")
            return default_suggestions
    
    async def analyze_damage_level(self, image_path: str, category: str, extra_image_path: str = None, prompt_context: dict = None) -> Tuple[int, str]:
        """
        Analyze damage level of e-waste using Gemini vision.
        Returns damage level (1-10) and detailed analysis.
        """
        if not self.is_available or self.model is None:
            return 5, "Damage analysis unavailable"
        
        try:
            images = [Image.open(image_path)]
            if extra_image_path:
                images.append(Image.open(extra_image_path))
            
            prompt = f"""
Rate damage level 1-10 for this e-waste:
Category: {category}

Scale:
1-2=Excellent (like new, minimal wear)
3-4=Good (light wear, fully functional)
5-6=Fair (moderate wear, some issues)
7-8=Poor (significant damage, limited function)
9-10=Severe (heavily damaged, non-functional)

Look for: scratches, cracks, missing parts, wear, functionality, corrosion, discoloration.

JSON only:
{{
    "damage_level": 1-10,
    "analysis": "brief condition description",
    "key_issues": ["main problems"]
}}
"""
            response = await self._call_gemini_with_timeout(prompt, images)
            
            if not response:
                return 5, "Damage analysis failed - empty response"
            
            # Parse response
            try:
                # Clean the response text to handle markdown code blocks
                cleaned_text = response.strip()
                if cleaned_text.startswith('```json'):
                    cleaned_text = cleaned_text[7:]  # Remove ```json
                if cleaned_text.endswith('```'):
                    cleaned_text = cleaned_text[:-3]  # Remove ```
                cleaned_text = cleaned_text.strip()
                
                if not cleaned_text:
                    logger.warning("Empty response from Gemini damage analysis")
                    return 5, "Empty response from Gemini"
                
                result = json.loads(cleaned_text)
                damage_level = int(result.get("damage_level", 5))
                analysis = result.get("analysis", "No detailed analysis available")
                
                # Validate damage level is in valid range
                if damage_level < 1 or damage_level > 10:
                    logger.warning(f"Invalid damage level {damage_level} from Gemini, defaulting to 5")
                    damage_level = 5
                
                return damage_level, analysis
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.error(f"Failed to parse damage analysis response: {e}")
                logger.debug(f"Raw response: {response[:200]}...")
                return 5, "Damage analysis parsing failed"
                
        except Exception as e:
            logger.error(f"Damage analysis error: {str(e)}")
            return 5, f"Damage analysis error: {str(e)}"
    
    def _create_validation_prompt(self, yolo_prediction: str, mapped_category: str) -> str:
        """Create optimized validation prompt for Gemini"""
        return f"""
Analyze this e-waste image quickly.

YOLO detected: {yolo_prediction}

Task: Identify the electronic object and find best category.

Valid categories: {', '.join(list(PRICE_CATEGORIES)[:20])}...

JSON response only:
{{
    "object_identified": "brief description",
    "is_valid_ewaste": true/false,
    "best_category": "category or null",
    "confidence": 0.0-1.0,
    "reasoning": "short explanation"
}}

Accept any electronic device as valid e-waste.
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
            
            # Handle both old and new response formats for backward compatibility
            is_valid_ewaste = gemini_result.get("is_valid_ewaste", gemini_result.get("is_category_correct", True))
            best_category = gemini_result.get("best_category", gemini_result.get("correct_category"))
            reasoning = gemini_result.get("reasoning", "")
            
            if not is_valid_ewaste:
                return ValidationResult(
                    is_valid=False,
                    final_category=None,
                    detection_source="Rejected",
                    gemini_feedback=f"Not valid e-waste: {reasoning}"
                )
            elif best_category and is_valid_price_category(best_category) and best_category != mapped_category:
                return ValidationResult(
                    is_valid=True,
                    final_category=best_category,
                    detection_source="Gemini Interfered",
                    gemini_feedback=f"Corrected from {mapped_category} to {best_category}: {reasoning}"
                )
            else:
                # Valid e-waste, use mapped category (or best_category if it's the same)
                final_category = best_category if best_category and is_valid_price_category(best_category) else mapped_category
                return ValidationResult(
                    is_valid=True,
                    final_category=final_category,
                    detection_source="YOLO" if final_category == mapped_category else "Gemini Interfered",
                    gemini_feedback=f"Valid e-waste confirmed: {reasoning}"
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

    async def _call_gemini_with_timeout(self, prompt: str, images: List[Image.Image]) -> str:
        """Make a Gemini call with timeout and error handling"""
        if not self.is_available or self.model is None:
            raise Exception("Gemini not available")
        
        def _sync_call():
            return self.model.generate_content(
                [prompt] + images,
                generation_config=self.generation_config
            )
        
        # Use asyncio.wait_for with executor for timeout
        try:
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(self.executor, _sync_call),
                timeout=GEMINI_REQUEST_TIMEOUT
            )
            return response.text or ""
        except asyncio.TimeoutError:
            logger.warning(f"Gemini request timed out after {GEMINI_REQUEST_TIMEOUT}s")
            return ""
        except Exception as e:
            logger.error(f"Gemini call failed: {str(e)}")
            return ""
