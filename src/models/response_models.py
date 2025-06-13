"""
Pydantic response models for API endpoints
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class Detection(BaseModel):
    """Single object detection result"""
    id: str = Field(..., description="Unique detection ID")
    category: str = Field(..., description="Detected category")
    confidence: float = Field(..., description="Detection confidence")
    bbox: List[float] = Field(..., description="Bounding box coordinates")


class PriceResponse(BaseModel):
    """Price prediction response"""
    category: str = Field(..., description="Category name")
    price: int = Field(..., description="Estimated price in IDR")


class ObjectResponse(BaseModel):
    """YOLO detection response"""
    detections: List[Detection] = Field(..., description="List of detected objects")


class FullPrediction(BaseModel):
    """Complete prediction with all analysis"""
    id: str = Field(..., description="Unique prediction ID")
    category: str = Field(..., description="Detected category")
    confidence: float = Field(..., description="Detection confidence")
    regression_result: Optional[int] = Field(None, description="Price prediction")
    description: str = Field(..., description="Visual description of e-waste item (max 20 words)")
    bbox: List[float] = Field(..., description="Bounding box coordinates")
    suggestion: List[str] = Field(..., description="Disposal suggestions")
    risk_lvl: int = Field(..., description="Risk level 1-5")
    damage_level: int = Field(..., description="Damage level 1-5 (1=Excellent, 5=Severe)")
    detection_source: str = Field(..., description="YOLO, Gemini Interfered, or Rejected")


class FullResponse(BaseModel):
    """Complete analysis response"""
    predictions: List[FullPrediction] = Field(..., description="Complete predictions")


class ValidationResult(BaseModel):
    """Gemini validation result"""
    is_valid: bool = Field(..., description="Whether detection is valid")
    final_category: Optional[str] = Field(None, description="Final category after validation")
    detection_source: str = Field(..., description="Source of final decision")
    gemini_feedback: str = Field(..., description="Gemini reasoning or error message")
