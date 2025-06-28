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
    GEMINI_TIMEOUT, GEMINI_REQUEST_TIMEOUT, GEMINI_MAX_CONCURRENT_REQUESTS,
    GEMINI_BATCH_SIZE, GEMINI_ENABLE_CROSS_VALIDATION
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
                prompt = f"""YOLO detected: {yolo_prediction}

What electronic device do you actually see? Choose best category:
{', '.join(list(PRICE_CATEGORIES)[:15])}...

JSON only:
{{"is_valid_ewaste": true/false, "best_category": "exact category name or null", "reasoning": "what you see"}}"""
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
            
            # Optimized prompt for speed
            prompt = f"""Describe this {category} in Indonesian (max 10 words):
Focus: condition, damage.
Example: "Laptop rusak layar retak"
"""
            
            response = await self._call_gemini_with_timeout(prompt, images)
            if response and response.strip():
                description = response.strip()
                # Clean up the description - remove any extra formatting
                description = description.replace('"', '').replace("'", "")
                
                # Take first line if multiple lines
                if '\n' in description:
                    description = description.split('\n')[0].strip()
                
                # Clean up the description
                if description and len(description) > 5:
                    return description
                else:
                    logger.warning("Empty or very short description from Gemini")
            else:
                logger.warning("No response text from Gemini for description")
            
            # Fallback description
            return f"{category} elektronik terdeteksi"
        except Exception as e:
            logger.error(f"Gemini description error: {str(e)}")
            return f"{category} elektronik terdeteksi"

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
            
            # Optimized prompt for speed
            prompt = f"""3 disposal steps for {category} in Indonesian:
1. [step 1 - max 6 words]
2. [step 2 - max 6 words] 
3. [step 3 - max 6 words]
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
            
            # Optimized prompt for speed
            prompt = f"""Rate {category} damage 1-10:
1-2=Excellent, 3-4=Good, 5-6=Fair, 7-8=Poor, 9-10=Severe

JSON only:
{{"damage_level":1-10, "analysis": "brief condition"}}"""
            
            response = await self._call_gemini_with_timeout(prompt, images)
            
            if not response:
                return 5, "Damage analysis failed - empty response"
            
            # Parse response with robust JSON extraction
            try:
                # More robust response cleaning (same as validation)
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
    
    def _create_validation_prompt(self, yolo_prediction: str, mapped_category: str) -> str:
        """Create optimized validation prompt for Gemini"""
        # Get the actual price categories dynamically
        from src.utils.mappings import PRICE_CATEGORIES
        categories_list = ", ".join(sorted(list(PRICE_CATEGORIES)))
        
        return f"""YOLO AI detected: {yolo_prediction}
Mapped to price category: {mapped_category}

Look at this image carefully. What electronic device do you actually see?

Choose the EXACT category name from this price model list:
{categories_list}

IMPORTANT GUIDELINES:
- For smartphones/mobile phones → use "Handphone"
- For walkie-talkies/two-way radios → use "Telefon"
- For regular computers → use "Laptop" or "CPU Intel"
- For gaming consoles → use "PS2"

If the YOLO detection and mapping are correct, you can confirm by returning the mapped category.
If it's not electronic waste, return "null".

JSON format only:
{{"is_valid_ewaste": true/false, "best_category": "exact category name from list above or null", "reasoning": "brief description of what you see"}}"""
    
    def _process_validation_response(
        self, 
        response_text: str, 
        mapped_category: str, 
        yolo_prediction: str
    ) -> ValidationResult:
        """Process and parse Gemini validation response"""
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
            
            # Handle both old and new response formats for backward compatibility
            is_valid_ewaste = gemini_result.get("is_valid_ewaste", gemini_result.get("is_category_correct", True))
            best_category = gemini_result.get("best_category", gemini_result.get("correct_category"))
            reasoning = gemini_result.get("reasoning", "")
            
            # Log for debugging
            logger.info(f"Validation result - Valid: {is_valid_ewaste}, Best category: {best_category}, YOLO: {yolo_prediction}, Mapped: {mapped_category}")
            logger.debug(f"Gemini reasoning: {reasoning}")
            
            if not is_valid_ewaste:
                return ValidationResult(
                    is_valid=False,
                    final_category=None,
                    detection_source="Rejected",
                    gemini_feedback=f"Not valid e-waste: {reasoning}"
                )
            elif best_category and best_category != "null" and is_valid_price_category(best_category):
                # Gemini provided a valid category - but let's be smart about it
                if best_category == mapped_category:
                    # Gemini confirmed the mapping is correct
                    return ValidationResult(
                        is_valid=True,
                        final_category=best_category,
                        detection_source="YOLO",
                        gemini_feedback=f"Category confirmed: {reasoning}"
                    )
                else:
                    # Gemini suggests a different category - validate this change
                    # Special handling for known good mappings
                    yolo_lower = yolo_prediction.lower()
                    best_lower = best_category.lower()
                    mapped_lower = mapped_category.lower()
                    
                    # Don't override well-established mappings unless there's strong reason
                    questionable_overrides = [
                        (yolo_lower == "phone" and best_lower == "telefon" and mapped_lower == "handphone"),
                        (yolo_lower == "laptop" and best_category != "Laptop"),
                        (yolo_lower == "monitor" and best_category != "Monitor"),
                        (yolo_lower == "keyboard" and best_category != "Keyboard"),
                        (yolo_lower == "mouse" and best_category != "Mouse"),
                    ]
                    
                    if any(questionable_overrides):
                        logger.warning(f"Gemini suggested questionable override: {yolo_prediction} -> {best_category}, keeping mapped: {mapped_category}")
                        return ValidationResult(
                            is_valid=True,
                            final_category=mapped_category,
                            detection_source="YOLO",
                            gemini_feedback=f"Kept original mapping {mapped_category} over Gemini suggestion {best_category}: {reasoning}"
                        )
                    else:
                        # Accept Gemini's suggestion for other cases
                        logger.info(f"Gemini correction: {yolo_prediction} -> {best_category} (was mapped to {mapped_category})")
                        return ValidationResult(
                            is_valid=True,
                            final_category=best_category,
                            detection_source="Gemini Interfered",
                            gemini_feedback=f"Corrected from {yolo_prediction} to {best_category}: {reasoning}"
                        )
            else:
                # No valid category provided or category not in our list - use mapped category
                # But first check if we can extract useful information from the reasoning
                if best_category == "null" or best_category is None:
                    logger.warning(f"Gemini couldn't identify a valid category. Reasoning: '{reasoning}'")
                    
                    # For well-known good mappings, don't try reasoning extraction - just use the mapping
                    yolo_lower = yolo_prediction.lower()
                    well_known_mappings = [
                        yolo_lower == "phone",
                        yolo_lower == "laptop", 
                        yolo_lower == "monitor",
                        yolo_lower == "keyboard",
                        yolo_lower == "mouse",
                        yolo_lower == "printer",
                        yolo_lower == "speaker",
                        yolo_lower == "battery",
                        yolo_lower == "charger"
                    ]
                    
                    if any(well_known_mappings):
                        logger.info(f"Using mapped category for well-known device: {yolo_prediction} -> {mapped_category}")
                        return ValidationResult(
                            is_valid=True,
                            final_category=mapped_category,
                            detection_source="YOLO",
                            gemini_feedback=f"Used mapped category for well-known device. Gemini reasoning: {reasoning}"
                        )
                    
                    # Try to extract category hints from the reasoning text only for unknown devices
                    reasoning_lower = reasoning.lower() if reasoning else ""
                    potential_categories = []
                    
                    # Check for common device mentions in reasoning
                    device_hints = {
                        "washing machine": "Mesin Cuci",
                        "mesin cuci": "Mesin Cuci", 
                        "washer": "Mesin Cuci",
                        "television": "TV",
                        "tv": "TV",
                        "monitor": "Monitor",
                        "laptop": "Laptop",
                        "computer": "CPU Intel",
                        "smartphone": "Handphone",
                        "phone": "Handphone",
                        "mobile": "Handphone",
                        "ponsel": "Handphone",
                        "handphone": "Handphone",
                        "printer": "Printer",
                        "scanner": "Printer",
                        "speaker": "Speaker",
                        "radio": "Speaker",
                        "microwave": "Microwave",
                        "refrigerator": "Komponen Kulkas",
                        "fridge": "Komponen Kulkas",
                        "kulkas": "Komponen Kulkas",
                        "keyboard": "Keyboard",
                        "mouse": "Mouse",
                        "battery": "Baterai Laptop",
                        "baterai": "Baterai Laptop",
                        "charger": "Adaptor /Kilo",
                        "adaptor": "Adaptor /Kilo",
                        "cables": "Adaptor /Kilo",
                        "kabel": "Adaptor /Kilo",
                        "iron": "Seterika",
                        "setrika": "Seterika",
                        "fan": "Kipas",
                        "kipas": "Kipas",
                        "lamp": "Lampu",
                        "lampu": "Lampu",
                        "router": "Router",
                        "walkie talkie": "Telefon",
                        "two way radio": "Telefon",
                        "hard disk": "Hardisk",
                        "harddisk": "Hardisk",
                        "storage": "Hardisk"
                    }
                    
                    for hint, category in device_hints.items():
                        if hint in reasoning_lower and is_valid_price_category(category):
                            potential_categories.append(category)
                    
                    # If we found a potential category in the reasoning, use it
                    if potential_categories:
                        corrected_category = potential_categories[0]  # Use first match
                        logger.info(f"Extracted category from reasoning: {yolo_prediction} -> {corrected_category}")
                        return ValidationResult(
                            is_valid=True,
                            final_category=corrected_category,
                            detection_source="Reasoning Extract",
                            gemini_feedback=f"Category extracted from reasoning: {reasoning}"
                        )
                else:
                    logger.warning(f"Gemini provided invalid category '{best_category}' (not in price list)")
                
                logger.warning(f"Using mapped category '{mapped_category}' as fallback")
                return ValidationResult(
                    is_valid=True,
                    final_category=mapped_category,
                    detection_source="YOLO",
                    gemini_feedback=f"Valid e-waste, using mapped category. Gemini reasoning: {reasoning}"
                )
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse Gemini JSON response: {response_text[:200]}...")
            logger.warning(f"JSON parse error: {str(e)}")
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
        mapped_category: str = None,
        extra_image_path: str = None,
        prompt_context: dict = None
    ) -> Dict[str, Any]:
        """
        Process all Gemini operations in parallel for maximum speed:
        - Validation, Description, Suggestions, and Damage Analysis
        """
        if not self.is_available or self.model is None:
            return {
                "validation": ValidationResult(
                    is_valid=True,
                    final_category=mapped_category or category,
                    detection_source="YOLO",
                    gemini_feedback="Gemini not available"
                ),
                "description": f"Perangkat elektronik {category.lower()}",
                "suggestions": [
                    "Periksa panduan manufacturer",
                    "Pisahkan komponen berbahaya", 
                    "Bawa ke pusat daur ulang e-waste"
                ],
                "damage_level": None,
                "damage_analysis": "Damage analysis unavailable"
            }
        
        try:
            # Create all tasks concurrently
            tasks = []
            
            # Validation task
            if yolo_prediction and mapped_category:
                tasks.append(("validation", self.validate_detection(
                    image_path, yolo_prediction, mapped_category, 
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
                            final_category=mapped_category or category,
                            detection_source="YOLO",
                            gemini_feedback=f"Validation failed: {str(result)}"
                        )
                    elif task_name == "description":
                        batch_result["description"] = f"{category} elektronik terdeteksi"
                    elif task_name == "suggestions":
                        batch_result["suggestions"] = [
                            "Periksa panduan manufacturer",
                            "Pisahkan komponen berbahaya",
                            "Bawa ke pusat daur ulang e-waste"
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
                            "Periksa panduan manufacturer",
                            "Pisahkan komponen berbahaya",
                            "Bawa ke pusat daur ulang e-waste"
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
                    final_category=mapped_category or category,
                    detection_source="YOLO",
                    gemini_feedback=f"Batch analysis error: {str(e)}"
                ),
                "description": f"Perangkat elektronik {category.lower()}",
                "suggestions": [
                    "Periksa panduan manufacturer",
                    "Pisahkan komponen berbahaya",
                    "Bawa ke pusat daur ulang e-waste"
                ],
                "damage_level": None,
                "damage_analysis": f"Batch analysis error: {str(e)}"
            }

    async def cross_validate_category(self, description: str, yolo_category: str, mapped_category: str) -> str:
        """
        Cross-validate category using the generated description to catch obvious mismatches
        """
        if not description or len(description) < 5:
            return mapped_category
        
        # Simple keyword matching for common mismatches
        description_lower = description.lower()
        
        # Common mismatch patterns
        category_keywords = {
            "Mesin Cuci": ["cuci", "washing", "mesin cuci"],
            "TV": ["tv", "televisi", "television", "layar besar"],
            "Laptop": ["laptop", "notebook", "komputer"],
            "Handphone": ["hp", "handphone", "phone", "smartphone"],
            "Printer": ["printer", "cetak", "print"],
            "Monitor": ["monitor", "layar komputer"],
            "Speaker": ["speaker", "audio", "suara"],
            "Microwave": ["microwave", "oven", "panggang"],
            "AC": ["ac", "air conditioner", "pendingin"],
            "Kipas": ["kipas", "fan", "angin"]
        }
        
        # Check if description suggests a different category
        for category, keywords in category_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                if category != yolo_category and category in PRICE_CATEGORIES:
                    logger.info(f"Cross-validation suggests category change: {yolo_category} -> {category} based on description: '{description}'")
                    return category
        
        return mapped_category
