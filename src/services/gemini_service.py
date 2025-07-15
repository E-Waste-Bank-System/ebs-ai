"""
Gemini AI Service Module
Handles validation using Google's Gemini with performance optimizations

Flow: YOLO Detection → Gemini Validation → YOLO-to-Price Mapping → Price Prediction
Gemini validates YOLO categories (37 classes) before mapping to price categories (33 classes)
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
    GEMINI_TIMEOUT, GEMINI_REQUEST_TIMEOUT, GEMINI_MAX_CONCURRENT_REQUESTS,
    GEMINI_BATCH_SIZE, GEMINI_ENABLE_CROSS_VALIDATION
)
from src.utils.mappings import CLASS_NAMES, get_all_yolo_classes
from src.models.response_models import ValidationResult

logger = logging.getLogger(__name__)


class GeminiService:
    """Gemini AI Service for YOLO detection validation with performance optimizations"""
    
    def __init__(self):
        self.model = None
        self.is_available = GEMINI_AVAILABLE
        self.executor = ThreadPoolExecutor(max_workers=GEMINI_MAX_WORKERS)
        
        # Add semaphore for controlling concurrent requests
        self.semaphore = asyncio.Semaphore(GEMINI_MAX_CONCURRENT_REQUESTS)
        
        # Optimized generation config for speed
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
                logger.info(f"Gemini service initialized with {GEMINI_MODEL} and {GEMINI_MAX_WORKERS} workers, max concurrent: {GEMINI_MAX_CONCURRENT_REQUESTS}")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {str(e)}")
                self.is_available = False
    
    async def validate_yolo_detection(
        self, 
        image_path: str, 
        yolo_prediction: str,
        yolo_confidence: float,
        extra_image_path: str = None,
        prompt_context: dict = None
    ) -> ValidationResult:
        """
        Validate YOLO detection using Gemini vision.
        
        Args:
            image_path: Path to cropped detection image
            yolo_prediction: YOLO's predicted category (from 37 YOLO classes)
            yolo_confidence: YOLO's confidence score
            extra_image_path: Optional additional image for context
            prompt_context: Optional context with all detections
            
        Returns:
            ValidationResult with confirmed/corrected YOLO category
        """
        if not self.is_available or self.model is None:
            logger.warning("Gemini not available, using YOLO prediction")
            return ValidationResult(
                is_valid=True,
                final_category=yolo_prediction,
                detection_source="YOLO",
                gemini_feedback="Gemini validation unavailable"
            )
        
        try:
            images = [Image.open(image_path)]
            if extra_image_path:
                images.append(Image.open(extra_image_path))
            
            # Build intelligent prompt for YOLO category validation
            prompt = self._create_yolo_validation_prompt(yolo_prediction, yolo_confidence, prompt_context)
            logger.info(f"[Gemini] Prompt for validation: {prompt}")
            
            # Gemini call with timeout
            response = await self._call_gemini_with_timeout(prompt, images)
            logger.info(f"[Gemini] Raw response: {response}")
            if not response:
                logger.warning("Empty Gemini validation response")
                return ValidationResult(
                    is_valid=True,
                    final_category=yolo_prediction,
                    detection_source="YOLO",
                    gemini_feedback="Gemini validation failed - empty response"
                )
            
            # Log the raw response for debugging
            logger.debug(f"Raw Gemini validation response: {response[:200]}...")
            
            return self._process_yolo_validation_response(
                response, yolo_prediction
            )
            
        except Exception as e:
            logger.error(f"Gemini validation error: {str(e)}")
            return ValidationResult(
                is_valid=True,
                final_category=yolo_prediction,
                detection_source="YOLO",
                gemini_feedback=f"Gemini validation error: {str(e)}"
            )

    async def generate_description(self, image_path: str, category: str, extra_image_path: str = None, prompt_context: dict = None) -> str:
        """
        Generate concise description (10-15 words) using Gemini vision based on e-waste condition.
        
        Args:
            image_path: Path to cropped detection image
            category: Final category name (can be YOLO or price category)
            extra_image_path: Optional additional image for context
            prompt_context: Optional context information
            
        Returns:
            Indonesian description of the e-waste item (10-15 words)
        """
        if not self.is_available or self.model is None:
            return f"Perangkat elektronik {category.lower()} dalam kondisi tidak diketahui"
        
        try:
            images = [Image.open(image_path)]
            if extra_image_path:
                images.append(Image.open(extra_image_path))
            
            # Optimized prompt for specific word count
            prompt = f"""Describe this {category} in Indonesian (EXACTLY 10-15 words):
Focus: condition, damage level, visible wear.
Format: "[device] [condition description]"
Example: "Laptop rusak layar retak baterai bocor casing tergores kondisi buruk"
IMPORTANT: Must be between 10-15 words, no more, no less.
"""
            logger.info(f"[Gemini] Prompt for description: {prompt}")
            
            response = await self._call_gemini_with_timeout(prompt, images)
            logger.info(f"[Gemini] Raw response: {response}")
            if response and response.strip():
                description = response.strip()
                # Clean up the description - remove any extra formatting
                description = description.replace('"', '').replace("'", "")
                
                # Take first line if multiple lines
                if '\n' in description:
                    description = description.split('\n')[0].strip()
                
                # Count words and validate length
                words = description.split()
                word_count = len(words)
                
                if 10 <= word_count <= 15:
                    logger.info(f"Generated description ({word_count} words): {description}")
                    return description
                elif word_count < 10:
                    # Too short, pad with generic terms
                    padding_words = ["terdeteksi", "dalam", "sistem", "pemeriksaan", "visual", "analisis"]
                    while len(words) < 10 and padding_words:
                        words.append(padding_words.pop(0))
                    description = " ".join(words[:15])  # Cap at 15 words
                    logger.info(f"Padded short description to {len(words)} words: {description}")
                    return description
                elif word_count > 15:
                    # Too long, truncate to 15 words
                    description = " ".join(words[:15])
                    logger.info(f"Truncated long description to 15 words: {description}")
                    return description
            else:
                logger.warning("No response text from Gemini for description")
            
            # Fallback description (exactly 10 words)
            fallback = f"Perangkat elektronik {category.lower()} terdeteksi dalam kondisi tidak dapat dianalisis"
            return " ".join(fallback.split()[:15])  # Ensure max 15 words
        except Exception as e:
            logger.error(f"Gemini description error: {str(e)}")
            # Fallback description (exactly 10 words)
            fallback = f"Perangkat elektronik {category.lower()} terdeteksi dalam kondisi tidak dapat dianalisis"
            return " ".join(fallback.split()[:15])  # Ensure max 15 words

    async def generate_suggestions(self, image_path: str, category: str, extra_image_path: str = None, prompt_context: dict = None) -> List[str]:
        """
        Generate disposal suggestions using Gemini vision based on e-waste condition.
        
        Args:
            image_path: Path to cropped detection image
            category: Final category name
            extra_image_path: Optional additional image for context
            prompt_context: Optional context information
            
        Returns:
            List of 3 disposal suggestions in Indonesian (5-7 words each)
        """
        default_suggestions = [
            "Periksa panduan dari manufacturer resmi",        # 6 words
            "Pisahkan komponen berbahaya dengan hati hati",  # 7 words
            "Bawa ke pusat daur ulang terdekat"              # 7 words
        ]
        if not self.is_available or self.model is None:
            return default_suggestions
        
        try:
            images = [Image.open(image_path)]
            if extra_image_path:
                images.append(Image.open(extra_image_path))
            
            # Optimized prompt for specific word count
            prompt = f"""3 disposal steps for {category} in Indonesian:
            Each step must be EXACTLY 5-7 words.
            Format:
            1. [action - 5-7 words]
            2. [action - 5-7 words] 
            3. [action - 5-7 words]

            Example:
            1. Periksa panduan dari manufacturer resmi
            2. Pisahkan komponen berbahaya dengan hati
            3. Bawa ke pusat daur ulang

            IMPORTANT: Each suggestion must be 5-7 words only. No markdown elements.
            """
            logger.info(f"[Gemini] Prompt for suggestions: {prompt}")
            
            response = await self._call_gemini_with_timeout(prompt, images)
            logger.info(f"[Gemini] Raw response: {response}")
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
                            if suggestion:
                                # Validate word count (5-7 words)
                                words = suggestion.split()
                                word_count = len(words)
                                
                                if 5 <= word_count <= 7:
                                    suggestions.append(suggestion)
                                    logger.info(f"Valid suggestion ({word_count} words): {suggestion}")
                                elif word_count < 5:
                                    # Too short, pad with generic terms
                                    padding_words = ["dengan", "cara", "yang", "benar", "sesuai", "aturan"]
                                    while len(words) < 5 and padding_words:
                                        words.append(padding_words.pop(0))
                                    suggestion = " ".join(words[:7])  # Cap at 7 words
                                    suggestions.append(suggestion)
                                    logger.info(f"Padded short suggestion to {len(words)} words: {suggestion}")
                                elif word_count > 7:
                                    # Too long, truncate to 7 words
                                    suggestion = " ".join(words[:7])
                                    suggestions.append(suggestion)
                                    logger.info(f"Truncated long suggestion to 7 words: {suggestion}")
                
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
                            if line:
                                # Validate word count (5-7 words)
                                words = line.split()
                                word_count = len(words)
                                
                                if 5 <= word_count <= 7:
                                    suggestions.append(line)
                                elif word_count < 5:
                                    # Too short, pad
                                    padding_words = ["dengan", "cara", "yang", "benar"]
                                    while len(words) < 5 and padding_words:
                                        words.append(padding_words.pop(0))
                                    suggestion = " ".join(words[:7])
                                    suggestions.append(suggestion)
                                elif word_count > 7:
                                    # Too long, truncate
                                    suggestion = " ".join(words[:7])
                                    suggestions.append(suggestion)
                
                # Ensure we have exactly 3 suggestions with proper word counts
                validated_suggestions = []
                for suggestion in suggestions[:3]:  # Take first 3
                    words = suggestion.split()
                    if len(words) < 5:
                        # Pad to minimum 5 words
                        padding = ["dengan", "cara", "yang", "benar", "sesuai"]
                        while len(words) < 5:
                            if padding:
                                words.append(padding.pop(0))
                            else:
                                break
                    elif len(words) > 7:
                        # Truncate to maximum 7 words
                        words = words[:7]
                    
                    validated_suggestions.append(" ".join(words))
                
                # Fill with defaults if we don't have 3 suggestions
                while len(validated_suggestions) < 3:
                    if len(validated_suggestions) < len(default_suggestions):
                        validated_suggestions.append(default_suggestions[len(validated_suggestions)])
                    else:
                        validated_suggestions.append("Bawa ke pusat daur ulang terdekat")
                
                # Final validation: ensure all suggestions are 5-7 words
                final_suggestions = []
                for suggestion in validated_suggestions[:3]:
                    words = suggestion.split()
                    if len(words) < 5:
                        words.extend(["dengan", "cara", "yang"])
                        words = words[:7]
                    elif len(words) > 7:
                        words = words[:7]
                    final_suggestions.append(" ".join(words))
                
                logger.info(f"Generated {len(final_suggestions)} suggestions with proper word counts")
                return final_suggestions[:3]
            
            logger.info("Using default suggestions due to generation failure")
            return default_suggestions
        except Exception as e:
            logger.error(f"Gemini suggestions error: {str(e)}")
            return default_suggestions
    
    async def analyze_damage_level(self, image_path: str, category: str, extra_image_path: str = None, prompt_context: dict = None) -> Tuple[int, str]:
        """
        Analyze damage level of e-waste using Gemini vision.
        
        Args:
            image_path: Path to cropped detection image
            category: Final category name
            extra_image_path: Optional additional image for context
            prompt_context: Optional context information
            
        Returns:
            Tuple of (damage_level: 1-10, analysis: str)
        """
        if not self.is_available or self.model is None:
            return 5, "Damage analysis unavailable"
        
        try:
            images = [Image.open(image_path)]
            if extra_image_path:
                images.append(Image.open(extra_image_path))
            
            # Optimized prompt for speed
            prompt = f"""Rate {category} damage 1-10:
1-2=Excellent, 3-4=Good, 5-6=Fair, 7-8=Poor, 9-10=Severe

JSON only:
{{"damage_level":1-10, "analysis": "brief condition"}}"""
            logger.info(f"[Gemini] Prompt for damage analysis: {prompt}")
            
            response = await self._call_gemini_with_timeout(prompt, images)
            logger.info(f"[Gemini] Raw response: {response}")
            
            if not response:
                return 5, "Damage analysis failed - empty response"
            
            # Parse response with robust JSON extraction
            try:
                # More robust response cleaning
                cleaned_text = response.strip()
                
                # Remove any text before the first { and after the last }
                start_idx = cleaned_text.find('{')
                end_idx = cleaned_text.rfind('}')
                
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    cleaned_text = cleaned_text[start_idx:end_idx + 1]
                else:
                    # If no valid JSON brackets found, try to extract from markdown
                    if '```json' in cleaned_text:
                        start = cleaned_text.find('```json') + 7
                        end = cleaned_text.find('```', start)
                        if end != -1:
                            cleaned_text = cleaned_text[start:end].strip()
                    elif '```' in cleaned_text:
                        start = cleaned_text.find('```') + 3
                        end = cleaned_text.find('```', start)
                        if end != -1:
                            cleaned_text = cleaned_text[start:end].strip()
                
                if not cleaned_text:
                    logger.warning("Empty response from Gemini damage analysis after cleaning")
                    return 5, "Empty response from Gemini"
                
                logger.debug(f"Cleaned damage response: {cleaned_text}")
                result = json.loads(cleaned_text)
                
                # More robust damage level extraction
                damage_level = result.get("damage_level")
                if damage_level is None:
                    logger.warning("No damage_level found in response, defaulting to 5")
                    damage_level = 5
                else:
                    try:
                        damage_level = int(damage_level)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid damage_level value '{damage_level}', defaulting to 5: {e}")
                        damage_level = 5
                
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
    
    def _create_yolo_validation_prompt(self, yolo_prediction: str, yolo_confidence: float, prompt_context: dict = None) -> str:
        """
        Create optimized validation prompt for YOLO category validation.
        
        Args:
            yolo_prediction: YOLO's predicted category
            yolo_confidence: YOLO's confidence score
            prompt_context: Optional context with all detections
            
        Returns:
            Formatted prompt for Gemini validation
        """
        # Get all YOLO categories for the prompt
        yolo_categories = get_all_yolo_classes()
        categories_list = ", ".join(sorted(yolo_categories))
        
        prompt = f"""YOLO AI detected: {yolo_prediction} (confidence: {yolo_confidence:.2f})

Look at this cropped image carefully. What electronic device do you actually see?

Choose the EXACT category name from this YOLO detection list:
{categories_list}

IMPORTANT GUIDELINES:
- For smartphones/mobile phones → use "Phone"
- For walkie-talkies/two-way radios → use "Walkie Talkie"
- For desktop computers → use "PC Case" or "CPU Component"
- For gaming controllers → use "Stick Ps"
- If you see multiple similar devices, choose the most specific one
- If it's not electronic waste, return "null"

Context: This is part of an e-waste detection system. The image has been cropped from a larger image containing the detected object with 20% padding for context.
"""

        if prompt_context and "all_detections" in prompt_context:
            detections_info = [
                f"- {d['category']} (conf: {d['confidence']:.2f})"
                for d in prompt_context["all_detections"][:5]  # Limit to first 5 for brevity
            ]
            prompt += f"""

Other detections in the full image:
{chr(10).join(detections_info)}
"""

        prompt += f"""

JSON format only:
{{"is_valid_ewaste": true/false, "best_yolo_category": "exact YOLO category name from list above or null", "reasoning": "brief description of what you see", "confidence_assessment": "high/medium/low based on image clarity"}}"""

        return prompt
    
    def _process_yolo_validation_response(
        self, 
        response_text: str, 
        yolo_prediction: str
    ) -> ValidationResult:
        """
        Process and parse Gemini YOLO validation response.
        
        Args:
            response_text: Raw response from Gemini
            yolo_prediction: Original YOLO prediction
            
        Returns:
            ValidationResult with confirmed/corrected YOLO category
        """
        try:
            # More robust response cleaning
            cleaned_text = response_text.strip()
            
            # Log the raw response for debugging
            logger.debug(f"Raw Gemini validation response: {response_text}")
            
            # Remove any text before the first { and after the last }
            start_idx = cleaned_text.find('{')
            end_idx = cleaned_text.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                cleaned_text = cleaned_text[start_idx:end_idx + 1]
            else:
                # If no valid JSON brackets found, try to extract from markdown
                if '```json' in cleaned_text:
                    start = cleaned_text.find('```json') + 7
                    end = cleaned_text.find('```', start)
                    if end != -1:
                        cleaned_text = cleaned_text[start:end].strip()
                elif '```' in cleaned_text:
                    start = cleaned_text.find('```') + 3
                    end = cleaned_text.find('```', start)
                    if end != -1:
                        cleaned_text = cleaned_text[start:end].strip()
            
            logger.debug(f"Cleaned validation response: {cleaned_text}")
            gemini_result = json.loads(cleaned_text)
            
            # Extract validation results
            is_valid_ewaste = gemini_result.get("is_valid_ewaste", True)
            best_yolo_category = gemini_result.get("best_yolo_category")
            reasoning = gemini_result.get("reasoning", "")
            confidence_assessment = gemini_result.get("confidence_assessment", "medium")
            
            # Log for debugging
            logger.info(f"YOLO Validation - Original: {yolo_prediction}, Gemini: {best_yolo_category}, Valid: {is_valid_ewaste}")
            logger.debug(f"Gemini reasoning: {reasoning}")
            
            if not is_valid_ewaste:
                return ValidationResult(
                    is_valid=False,
                    final_category=None,
                    detection_source="Rejected",
                    gemini_feedback=f"Not valid e-waste: {reasoning}"
                )
            elif best_yolo_category and best_yolo_category != "null" and best_yolo_category in get_all_yolo_classes():
                # Gemini provided a valid YOLO category
                if best_yolo_category == yolo_prediction:
                    # Gemini confirmed YOLO's prediction
                    return ValidationResult(
                        is_valid=True,
                        final_category=best_yolo_category,
                        detection_source="YOLO",
                        gemini_feedback=f"YOLO detection confirmed ({confidence_assessment} confidence): {reasoning}"
                    )
                else:
                    # Only accept correction if confidence_assessment is high
                    if confidence_assessment.lower() == "high":
                        logger.info(f"Gemini corrected YOLO: {yolo_prediction} → {best_yolo_category} (high confidence)")
                        return ValidationResult(
                            is_valid=True,
                            final_category=best_yolo_category,
                            detection_source="Gemini Corrected",
                            gemini_feedback=f"Corrected from {yolo_prediction} to {best_yolo_category} ({confidence_assessment} confidence): {reasoning}"
                        )
                    else:
                        logger.info(f"Gemini suggested correction {yolo_prediction} → {best_yolo_category} but confidence was {confidence_assessment}, ignoring correction.")
                        return ValidationResult(
                            is_valid=True,
                            final_category=yolo_prediction,
                            detection_source="YOLO",
                            gemini_feedback=f"Gemini suggested correction but confidence was {confidence_assessment}, ignored."
                        )
            else:
                # No valid category provided or category not in YOLO list
                if best_yolo_category == "null" or best_yolo_category is None:
                    logger.warning(f"Gemini couldn't identify a valid YOLO category. Reasoning: '{reasoning}'")
                else:
                    logger.warning(f"Gemini provided invalid YOLO category '{best_yolo_category}' (not in YOLO class list)")
                
                # Use original YOLO prediction as fallback
                logger.info(f"Using original YOLO prediction '{yolo_prediction}' as fallback")
                return ValidationResult(
                    is_valid=True,
                    final_category=yolo_prediction,
                    detection_source="YOLO",
                    gemini_feedback=f"Used original YOLO prediction as fallback. Gemini reasoning: {reasoning}"
                )
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse Gemini JSON response: {response_text[:200]}...")
            logger.warning(f"JSON parse error: {str(e)}")
            return ValidationResult(
                is_valid=True,
                final_category=yolo_prediction,
                detection_source="YOLO",
                gemini_feedback="Gemini response parsing failed - using YOLO prediction"
            )
    
    def is_service_available(self) -> bool:
        """Check if Gemini service is available"""
        return self.is_available

    async def _call_gemini_with_timeout(self, prompt: str, images: List[Image.Image]) -> str:
        """Make a Gemini call with timeout, concurrency control, and error handling"""
        if not self.is_available or self.model is None:
            raise Exception("Gemini not available")
        
        async with self.semaphore:  # Control concurrent requests
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

    async def process_batch_analysis(
        self, 
        image_path: str, 
        category: str, 
        yolo_prediction: str = None,
        yolo_confidence: float = None,
        extra_image_path: str = None,
        prompt_context: dict = None
    ) -> Dict[str, Any]:
        """
        Process all Gemini operations in parallel for maximum speed:
        - YOLO Validation, Description, Suggestions, and Damage Analysis
        
        Args:
            image_path: Path to cropped detection image
            category: Display category for content generation
            yolo_prediction: Original YOLO prediction for validation
            yolo_confidence: YOLO confidence score
            extra_image_path: Optional additional image
            prompt_context: Optional context information
            
        Returns:
            Dictionary with all analysis results
        """
        if not self.is_available or self.model is None:
            return {
                "validation": ValidationResult(
                    is_valid=True,
                    final_category=yolo_prediction or category,
                    detection_source="YOLO",
                    gemini_feedback="Gemini not available"
                ),
                "description": f"Perangkat elektronik {category.lower()} terdeteksi dalam kondisi tidak dapat dianalisis",
                "suggestions": [
                    "Periksa panduan dari manufacturer resmi",        # 6 words
                    "Pisahkan komponen berbahaya dengan hati hati",  # 7 words
                    "Bawa ke pusat daur ulang terdekat"              # 7 words
                ],
                "damage_level": None,
                "damage_analysis": "Damage analysis unavailable"
            }
        
        try:
            # Create all tasks concurrently
            tasks = []
            
            # YOLO Validation task (if we have YOLO prediction info)
            if yolo_prediction and yolo_confidence is not None:
                tasks.append(("validation", self.validate_yolo_detection(
                    image_path, yolo_prediction, yolo_confidence, 
                    extra_image_path, prompt_context
                )))
            
            # Content generation tasks
            tasks.append(("description", self.generate_description(
                image_path, category, extra_image_path, prompt_context
            )))
            tasks.append(("suggestions", self.generate_suggestions(
                image_path, category, extra_image_path, prompt_context
            )))
            tasks.append(("damage", self.analyze_damage_level(
                image_path, category, extra_image_path, prompt_context
            )))
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)
            
            # Process results
            batch_result = {}
            for i, (task_name, _) in enumerate(tasks):
                result = results[i]
                if isinstance(result, Exception):
                    logger.error(f"Batch task {task_name} failed: {result}")
                    # Set default values for failed tasks
                    if task_name == "validation":
                        batch_result["validation"] = ValidationResult(
                            is_valid=True,
                            final_category=yolo_prediction or category,
                            detection_source="YOLO",
                            gemini_feedback=f"Validation failed: {str(result)}"
                        )
                    elif task_name == "description":
                        batch_result["description"] = f"Perangkat elektronik {category.lower()} terdeteksi dalam kondisi tidak dapat dianalisis"
                    elif task_name == "suggestions":
                        batch_result["suggestions"] = [
                            "Periksa panduan dari manufacturer resmi",        # 6 words
                            "Pisahkan komponen berbahaya dengan hati hati",  # 7 words
                            "Bawa ke pusat daur ulang terdekat"              # 7 words
                        ]
                    elif task_name == "damage":
                        batch_result["damage_level"] = 5  # Default middle value
                        batch_result["damage_analysis"] = f"Damage analysis failed: {str(result)}"
                else:
                    if task_name == "validation":
                        batch_result["validation"] = result
                    elif task_name == "description":
                        # Ensure we have a valid description
                        desc = result if result and len(result.strip()) > 3 else f"{category} elektronik terdeteksi"
                        batch_result["description"] = desc
                    elif task_name == "suggestions":
                        # Ensure we have valid suggestions
                        suggs = result if result and len(result) == 3 else [
                            "Periksa panduan dari manufacturer resmi",        # 6 words
                            "Pisahkan komponen berbahaya dengan hati hati",  # 7 words
                            "Bawa ke pusat daur ulang terdekat"              # 7 words
                        ]
                        batch_result["suggestions"] = suggs
                    elif task_name == "damage":
                        damage_level, damage_analysis = result
                        # Ensure valid damage level
                        if damage_level is None or not isinstance(damage_level, int) or damage_level < 1 or damage_level > 10:
                            damage_level = 5
                        batch_result["damage_level"] = damage_level
                        batch_result["damage_analysis"] = damage_analysis or "No analysis available"
            
            return batch_result
            
        except Exception as e:
            logger.error(f"Batch analysis failed: {str(e)}")
            return {
                "validation": ValidationResult(
                    is_valid=True,
                    final_category=yolo_prediction or category,
                    detection_source="YOLO",
                    gemini_feedback=f"Batch analysis error: {str(e)}"
                ),
                "description": f"Perangkat elektronik {category.lower()} terdeteksi dalam kondisi tidak dapat dianalisis",
                "suggestions": [
                    "Periksa panduan dari manufacturer resmi",        # 6 words
                    "Pisahkan komponen berbahaya dengan hati hati",  # 7 words
                    "Bawa ke pusat daur ulang terdekat"              # 7 words
                ],
                "damage_level": None,
                "damage_analysis": f"Batch analysis error: {str(e)}"
            }

    async def cross_validate_category(self, description: str, yolo_category: str, validated_category: str) -> str:
        """
        Cross-validate category using the generated description to catch obvious mismatches.
        
        Args:
            description: Generated description of the device
            yolo_category: Original YOLO category
            validated_category: Gemini-validated YOLO category
            
        Returns:
            Final validated YOLO category
        """
        if not description or len(description) < 5:
            return validated_category
        
        # Simple keyword matching for common mismatches
        description_lower = description.lower()
        
        # Common mismatch patterns for YOLO categories
        category_keywords = {
            "Washing Machine": ["cuci", "washing", "mesin cuci"],
            "Television": ["tv", "televisi", "television", "layar besar"],
            "Laptop": ["laptop", "notebook", "komputer"],
            "Phone": ["hp", "handphone", "phone", "smartphone"],
            "Printer": ["printer", "cetak", "print"],
            "Monitor": ["monitor", "layar komputer"],
            "Speaker": ["speaker", "audio", "suara"],
            "Microwave": ["microwave", "oven", "panggang"],
            "Fan": ["kipas", "fan", "angin"],
            "Walkie Talkie": ["walkie", "radio", "komunikasi"]
        }
        
        # Check if description suggests a different YOLO category
        for category, keywords in category_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                if category != validated_category and category in get_all_yolo_classes():
                    logger.info(f"Cross-validation suggests YOLO category change: {validated_category} → {category} based on description: '{description}'")
                    return category
        
        return validated_category
