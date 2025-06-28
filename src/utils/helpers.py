"""
Helper functions for E-waste detection system
"""

import uuid
import logging
from typing import List, Optional, Any, Callable, TypeVar
from functools import wraps
from src.config.settings import LOW_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

T = TypeVar('T')

def generate_unique_id() -> str:
    """Generate unique ID for detections"""
    return str(uuid.uuid4())


def safe_execute(func: Callable[..., T], default_value: T, error_msg: str = "", *args, **kwargs) -> T:
    """
    Safely execute a function with error handling and default fallback
    
    Args:
        func: Function to execute
        default_value: Value to return if function fails
        error_msg: Custom error message prefix
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        Function result or default_value if failed
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        full_msg = f"{error_msg}: {str(e)}" if error_msg else f"Function {func.__name__} failed: {str(e)}"
        logger.error(full_msg)
        return default_value


def log_execution_time(func_name: str = ""):
    """Decorator to log function execution time"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"{func_name or func.__name__} completed in {execution_time:.2f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func_name or func.__name__} failed after {execution_time:.2f}s: {str(e)}")
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"{func_name or func.__name__} completed in {execution_time:.2f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func_name or func.__name__} failed after {execution_time:.2f}s: {str(e)}")
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def create_fallback_prediction(
    category: str, 
    confidence: float, 
    bbox: List[float], 
    price: Optional[int] = None,
    detection_source: str = "Fallback"
) -> dict:
    """
    Create a standardized fallback prediction object
    
    Args:
        category: Detection category
        confidence: Detection confidence
        bbox: Bounding box coordinates
        price: Optional price value
        detection_source: Source identifier
        
    Returns:
        Dictionary with fallback prediction data
    """
    from src.models.response_models import FullPrediction
    
    risk_level = calculate_risk_level(category, confidence)
    
    return FullPrediction(
        id=generate_unique_id(),
        category=category,
        confidence=confidence,
        regression_result=price,
        description=f"Perangkat elektronik {category.lower()}",
        bbox=bbox,
        suggestion=[
            "Periksa panduan manufacturer",
            "Pisahkan komponen berbahaya",
            "Bawa ke pusat daur ulang e-waste"
        ],
        risk_lvl=risk_level,
        damage_level=None,
        detection_source=detection_source
    )


def calculate_risk_level(category: str, confidence: float) -> int:
    """
    Calculate risk level 1-10 based on YOLO category and confidence
    Higher risk = more dangerous to environment/health
    """
    # Risk levels based on YOLO class names (1-5 base scale)
    # Categories with high environmental/health risks
    high_risk_categories = {
        "Television", "Fridge", "Microwave", "Washing Machine", 
        "Rice Cooker", "Iron"
    }
    
    # Categories with medium-high risks (batteries, screens, complex electronics)
    medium_high_risk_categories = {
        "Laptop", "Phone", "Monitor", "Battery", "Powerbank",
        "GPU", "Motherboard", "PC Case", "CPU Component"
    }
    
    # Categories with medium risks (general electronics)
    medium_risk_categories = {
        "Printer", "Speaker", "Router", "Solar Panel", "DVD Player",
        "Radio", "Microphone", "Harddisk", "Stick Ps"
    }
    
    # Categories with lower risks (peripherals, small devices)
    low_risk_categories = {
        "Keyboard", "Mouse", "Charger", "Electronic Socket", "Cables",
        "Calculator", "Clock", "Walkie Talkie", "Body Weight Scale", "Remote"
    }
    
    # Categories with minimal risks (simple devices)
    minimal_risk_categories = {
        "Fan", "Lamp", "Flashlight"
    }
    
    # Determine base risk level
    if category in high_risk_categories:
        base_risk = 5
    elif category in medium_high_risk_categories:
        base_risk = 4
    elif category in medium_risk_categories:
        base_risk = 3
    elif category in low_risk_categories:
        base_risk = 2
    elif category in minimal_risk_categories:
        base_risk = 1
    else:
        # Unknown category - assign medium risk
        base_risk = 3
    
    # Adjust based on confidence level
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        # Low confidence increases risk (uncertainty is risky)
        base_risk = min(5, base_risk + 1)
    elif confidence > 0.9:
        # Very high confidence slightly reduces risk
        base_risk = max(1, base_risk - 1)
    
    # Scale to 1-10 range
    scaled_risk = base_risk * 2
    return min(10, max(1, scaled_risk))



