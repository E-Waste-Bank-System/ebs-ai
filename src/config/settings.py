"""
Configuration settings for the E-Waste Detection System
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
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "v5.pt")
KNR_MODEL_PATH = os.path.join(MODELS_DIR, "model_knr_best.joblib")
ENCODER_PATH = os.path.join(MODELS_DIR, "encoder_target.joblib")

# API Configuration
API_TITLE = "E-Waste Detection API"
API_DESCRIPTION = "Production API for e-waste detection with YOLO, pricing"
API_VERSION = "3.0.1"  # Updated version to reflect optimizations
HOST = "0.0.0.0"
PORT = 8080

# Detection thresholds
LOW_CONFIDENCE_THRESHOLD = 0.5
MEDIUM_CONFIDENCE_THRESHOLD = 0.7

# Gemini AI Configuration
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GEMINI_MODEL = 'gemini-2.5-flash'

# Gemini generation settings (optimized for speed)
GEMINI_MAX_TOKENS = 2048  # Reduced from 2048 for faster responses
GEMINI_TEMPERATURE = 0.3
GEMINI_TOP_P = 0.8

# Gemini performance settings
GEMINI_MAX_WORKERS = int(os.environ.get('GEMINI_MAX_WORKERS', '8'))  # Reduced default
GEMINI_TIMEOUT = float(os.environ.get('GEMINI_TIMEOUT', '12.0'))  # Reduced timeout
GEMINI_REQUEST_TIMEOUT = float(os.environ.get('GEMINI_REQUEST_TIMEOUT', '10.0'))  # Reduced timeout
GEMINI_MAX_CONCURRENT_REQUESTS = int(os.environ.get('GEMINI_MAX_CONCURRENT_REQUESTS', '4'))  # Reduced default
GEMINI_BATCH_SIZE = int(os.environ.get('GEMINI_BATCH_SIZE', '3'))

# Gemini feature flags (optimized defaults)
GEMINI_ENABLE_CROSS_VALIDATION = os.environ.get('GEMINI_ENABLE_CROSS_VALIDATION', 'false').lower() == 'true'  # Disabled by default for speed

# Feature availability flags
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
    else:
        GEMINI_AVAILABLE = False
        logger.warning("GEMINI_API_KEY not found. Gemini features disabled.")
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Google Generative AI not available.")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("YOLO not available.")
