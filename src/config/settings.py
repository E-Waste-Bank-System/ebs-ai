"""
Configuration settings for the E-Waste Detection System

Pipeline Flow: YOLO Detection → Gemini Validation → YOLO-to-Price Mapping → Price Prediction
- YOLO: 37 categories for object detection
- Price Model: 33 categories for price prediction
- Gemini: Validates YOLO predictions before mapping
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Model paths
YOLO_MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(MODELS_DIR, "best.pt"))
REG_MODEL_PATH = os.environ.get('REG_MODEL_PATH', os.path.join(MODELS_DIR, "regresih.joblib"))

# API Configuration
API_TITLE = "E-Waste Detection API"
API_DESCRIPTION = "Production API for e-waste detection with YOLO, Gemini validation, and pricing"
API_VERSION = "3.1.0"
HOST = "0.0.0.0"
PORT = 8080

# Detection thresholds
LOW_CONFIDENCE_THRESHOLD = 0.5
MEDIUM_CONFIDENCE_THRESHOLD = 0.7

# Gemini AI Configuration
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GEMINI_MODEL = 'gemini-2.5-flash'

# Gemini generation settings
GEMINI_MAX_TOKENS = 2048
GEMINI_TEMPERATURE = 0.0
GEMINI_TOP_P = 0.8

# Gemini performance settings
GEMINI_MAX_WORKERS = 8
GEMINI_TIMEOUT = 12.0
GEMINI_REQUEST_TIMEOUT =10.0
GEMINI_MAX_CONCURRENT_REQUESTS = 4
GEMINI_BATCH_SIZE = 3

GEMINI_ENABLE_CROSS_VALIDATION = True

# Feature availability flags and initialization
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("Gemini AI configured successfully")
    else:
        GEMINI_AVAILABLE = False
        logger.warning("GEMINI_API_KEY not found. Gemini features disabled.")
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Google Generative AI not available. Install google-generativeai package.")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    logger.info("YOLO (Ultralytics) available")
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("YOLO not available. Install ultralytics package.")

# Pipeline configuration summary
PIPELINE_CONFIG = {
    "flow": "YOLO Detection → Gemini Validation → YOLO-to-Price Mapping → Price Prediction",
    "yolo_categories": 37,
    "price_categories": 33,
    "gemini_validation": GEMINI_AVAILABLE,
    "yolo_detection": YOLO_AVAILABLE
}

logger.info(f"Pipeline configuration: {PIPELINE_CONFIG}")

# Log model paths for debugging
logger.info(f"YOLO_MODEL_PATH: {YOLO_MODEL_PATH}, exists: {os.path.exists(YOLO_MODEL_PATH)}")
logger.info(f"REG_MODEL_PATH: {REG_MODEL_PATH}, exists: {os.path.exists(REG_MODEL_PATH)}")
logger.info(f"GEMINI_API_KEY: {'set' if GEMINI_API_KEY else 'not set'}")
