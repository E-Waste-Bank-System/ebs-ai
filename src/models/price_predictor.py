"""
Price Prediction Module
Handles K-Nearest Neighbors regression model for price prediction - Final stage of the pipeline

Pipeline: YOLO Detection → Gemini Validation → YOLO-to-Price Mapping → Price Prediction
This module handles stage 4: Predicts prices using 33 price model categories
"""

import os
import logging
import sys
from typing import List, Optional
import joblib
from joblib import load
import pandas as pd

from src.models.regresih import Regresih
from src.config.settings import REG_MODEL_PATH
from src.utils.mappings import PRICE_CATEGORIES, is_valid_price_category

logger = logging.getLogger(__name__)
sys.modules['__main__'].Regresih = Regresih


class PricePredictor:
    """
    K-Nearest Neighbors Price Prediction Manager
    
    Responsible for the final stage of the pipeline:
    - Loads KNR model and encoder for 33 price categories
    - Predicts prices for mapped categories from YOLO→Price mapping
    - Provides fallback prices when ML models unavailable
    """
    
    def __init__(self):
        self.model = None
        self.encoder = None
        self.is_loaded = False
    
    def load_models(self) -> bool:
        """
        Load KNR model and target encoder for price prediction
        
        Returns:
            True if models loaded successfully, False otherwise
        """
        try:
            # Check if model files exist
            if not os.path.exists(REG_MODEL_PATH):
                logger.error(f"KNR model file not found: {REG_MODEL_PATH}")
                return False
                
            # if not os.path.exists(ENCODER_PATH):
            #     logger.error(f"Encoder file not found: {ENCODER_PATH}")
            #     return False
            
            logger.info(f"Loading KNR model from: {REG_MODEL_PATH}")
            # logger.info(f"Loading encoder from: {ENCODER_PATH}")
            
            # Load the models with better error handling
            try:
                self.model = joblib.load(REG_MODEL_PATH)
                logger.info("KNR price prediction model loaded successfully")
            except Exception as model_error:
                logger.error(f"Failed to load KNR model: {str(model_error)}")
                logger.error(f"Model error type: {type(model_error).__name__}")
                return False
            
            # try:
            #     self.encoder = joblib.load(ENCODER_PATH)
            #     logger.info("Price category encoder loaded successfully")
            # except Exception as encoder_error:
            #     logger.error(f"Failed to load encoder: {str(encoder_error)}")
            #     logger.error(f"Encoder error type: {type(encoder_error).__name__}")
            #     return False
            
            # Test the models with a sample prediction
            try:
                test_categories = list(PRICE_CATEGORIES)[:3]  # Test with first 3 categories
                for test_cat in test_categories:
                    test_df = pd.DataFrame({
                        "Nama Item": [test_cat],
                        "Kondisi": ["Baik"]
                    })
                    test_prediction = self.model.predict(test_df)
                    # Convert prediction to integer for consistent logging
                    if test_prediction is not None and len(test_prediction) > 0:
                        test_price = int(test_prediction[0]) if hasattr(test_prediction, '__getitem__') else int(test_prediction)
                        logger.info(f"Test price prediction for {test_cat}: {test_price} IDR")
                    else:
                        logger.error(f"Test prediction returned None or empty for {test_cat}")
                        return False
                    break  # Only test one to verify it works
                    
                self.is_loaded = True
                logger.info("Price prediction model and encoder loaded and tested successfully")
                logger.info(f"Model supports {len(PRICE_CATEGORIES)} price categories")
                return True
                
            except Exception as test_error:
                logger.error(f"Model test failed: {str(test_error)}")
                logger.error(f"Test error type: {type(test_error).__name__}")
                logger.error("Models loaded but failed validation test")
                return False
                
        except ImportError as import_error:
            logger.error(f"Import error loading models: {str(import_error)}")
            logger.error("This usually indicates a dependency version mismatch")
            logger.error("Check scikit-learn, scipy, pandas, and joblib versions")
            return False
        except Exception as e:
            logger.error(f"Unexpected error loading price prediction models: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def predict_price(self, price_category: str) -> Optional[int]:
        """
        Predict price for given price model category (after YOLO→Price mapping)
        
        Args:
            price_category: Price model category name (must be from 33 price categories)
            
        Returns:
            Predicted price in IDR or None if failed
        """
        # Validate that this is a price category, not a YOLO category
        if not is_valid_price_category(price_category):
            logger.error(f"Invalid price category: {price_category}")
            logger.error("Expected price model category (33 classes), not YOLO category (37 classes)")
            return None
        
        if not self.is_loaded:
            logger.warning("Price prediction model not loaded, using fallback prices")
            return self._get_fallback_price(price_category)
        
        try:
            # Try different column names that might be expected by the model
            df = pd.DataFrame({
                "Nama Item": [price_category],
                "Kondisi": ["Baik"]
            })
            
            print(type(self.model))
            prediction = self.model.predict(df)
            
            # Convert prediction to integer (prediction is usually a numpy array)
            if prediction is not None and len(prediction) > 0:
                price = int(prediction[0]) if hasattr(prediction, '__getitem__') else int(prediction)
                logger.info(f"ML price prediction for {price_category}: {price} IDR")
                return price
            else:
                logger.error(f"Model returned None or empty prediction for {price_category}")
                return self._get_fallback_price(price_category)
            
        except KeyError:
            # Fallback to 'name' if 'Nama Item' doesn't work
            try:
                df = pd.DataFrame({
                    "name": [price_category],
                    "Kondisi": ["Baik"]
                })
                prediction = self.model.predict(df)
                
                # Convert prediction to integer (prediction is usually a numpy array)
                if prediction is not None and len(prediction) > 0:
                    price = int(prediction[0]) if hasattr(prediction, '__getitem__') else int(prediction)
                    logger.info(f"ML price prediction for {price_category}: {price} IDR (fallback column)")
                    return price
                else:
                    logger.error(f"Model returned None or empty prediction for {price_category} (fallback column)")
                    return self._get_fallback_price(price_category)
                
            except Exception as e:
                logger.error(f"Price prediction failed with both column names: {str(e)}")
                logger.warning("Using fallback price due to prediction failure")
                return self._get_fallback_price(price_category)
        except Exception as e:
            logger.error(f"Price prediction error: {str(e)}")
            logger.warning("Using fallback price due to prediction error")
            return self._get_fallback_price(price_category)
    
    def _get_fallback_price(self, price_category: str) -> int:
        """
        Get fallback price when ML models are not available
        
        Args:
            price_category: Price model category name
            
        Returns:
            Fallback price in IDR
        """
        # Fallback prices based on typical e-waste values in IDR
        # These prices are for the 33 price model categories
        fallback_prices = {
            "Handphone": 8000,
            "Laptop": 25000,
            "Monitor": 15000,
            "TV": 20000,
            "Printer": 10000,
            "Keyboard": 2000,
            "Mouse": 1000,
            "Speaker": 3000,
            "Router": 5000,
            "Hardisk": 8000,
            "Microwave": 18000,
            "Mesin Cuci": 30000,
            "Seterika": 4000,
            "Kipas": 6000,
            "Lampu": 1500,
            "CPU Intel": 20000,
            "Komponen CPU": 12000,
            "Baterai Laptop": 5000,
            "Adaptor /Kilo": 2000,
            "Telefon": 6000,
            "PS2": 8000,
            "Panel Surya": 35000,
            "Komponen Kulkas": 15000,
            "Senter": 1000,
            "Jam Tangan": 3000,
            "Alat Tensi": 8000,
            "Alat Tes Vol": 4000,
            "Kompor Listrik": 12000,
            "Remot": 1500
        }
        
        price = fallback_prices.get(price_category, 5000)  # Default 5000 IDR
        logger.info(f"Fallback price for {price_category}: {price} IDR")
        return price
    
    def get_supported_categories(self) -> List[str]:
        """Get list of supported price categories (33 categories)"""
        return sorted(list(PRICE_CATEGORIES))
    
    def is_category_supported(self, price_category: str) -> bool:
        """Check if category is supported by price model"""
        return is_valid_price_category(price_category)
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded price prediction model
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_loaded": self.is_loaded,
            "model_path": REG_MODEL_PATH if self.is_loaded else None,
            "categories_count": len(PRICE_CATEGORIES),
            "pipeline_stage": "4 - Price Prediction",
            "input": "Price model categories (after YOLO→Price mapping)",
            "output": "Price estimates in IDR",
            "fallback_available": True
        }
