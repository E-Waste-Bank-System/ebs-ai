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
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "v32.pt")
KNR_MODEL_PATH = os.path.join(MODELS_DIR, "model_knr_best.joblib")
ENCODER_PATH = os.path.join(MODELS_DIR, "encoder_target.joblib")

# API Configuration
API_TITLE = "E-Waste Detection API"
API_DESCRIPTION = "Production API for e-waste detection with YOLO, pricing"
API_VERSION = "1.0.0"
HOST = "0.0.0.0"
PORT = 8080

# Gemini Configuration
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GEMINI_MODEL = 'gemini-1.5-flash'
GEMINI_MAX_TOKENS = 2048
GEMINI_TEMPERATURE = 0.1
GEMINI_TOP_P = 0.8
GEMINI_MAX_WORKERS = int(os.environ.get('GEMINI_MAX_WORKERS', '5'))
GEMINI_TIMEOUT = float(os.environ.get('GEMINI_TIMEOUT', '15.0'))
GEMINI_REQUEST_TIMEOUT = float(os.environ.get('GEMINI_REQUEST_TIMEOUT', '10.0'))

# Gemini feature flags for performance tuning
GEMINI_SKIP_VALIDATION = os.environ.get('GEMINI_SKIP_VALIDATION', 'false').lower() == 'true'
GEMINI_SKIP_DESCRIPTION = os.environ.get('GEMINI_SKIP_DESCRIPTION', 'false').lower() == 'true'
GEMINI_SKIP_DAMAGE_ANALYSIS = os.environ.get('GEMINI_SKIP_DAMAGE_ANALYSIS', 'false').lower() == 'true'

# Detection thresholds
LOW_CONFIDENCE_THRESHOLD = 0.5
MEDIUM_CONFIDENCE_THRESHOLD = 0.7

# Feature flags
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
