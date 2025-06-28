"""
Price Prediction Module
Handles K-Nearest Neighbors regression model for price prediction
"""

import os
import logging
from typing import List, Optional
import joblib
import pandas as pd

from src.config.settings import KNR_MODEL_PATH, ENCODER_PATH
from src.utils.mappings import PRICE_CATEGORIES, is_valid_price_category

logger = logging.getLogger(__name__)


class PricePredictor:
    """K-Nearest Neighbors Price Prediction Manager"""
    
    def __init__(self):
        self.model = None
        self.encoder = None
        self.is_loaded = False
    
    def load_models(self) -> bool:
        """Load KNR model and target encoder"""
        try:
            # Check if model files exist
            if not os.path.exists(KNR_MODEL_PATH):
                logger.error(f"KNR model file not found: {KNR_MODEL_PATH}")
                return False
                
            if not os.path.exists(ENCODER_PATH):
                logger.error(f"Encoder file not found: {ENCODER_PATH}")
                return False
            
            logger.info(f"Loading KNR model from: {KNR_MODEL_PATH}")
            logger.info(f"Loading encoder from: {ENCODER_PATH}")
            
            # Load the models with better error handling
            try:
                self.model = joblib.load(KNR_MODEL_PATH)
                logger.info("KNR model loaded successfully")
            except Exception as model_error:
                logger.error(f"Failed to load KNR model: {str(model_error)}")
                logger.error(f"Model error type: {type(model_error).__name__}")
                return False
            
            try:
                self.encoder = joblib.load(ENCODER_PATH)
                logger.info("Target encoder loaded successfully")
            except Exception as encoder_error:
                logger.error(f"Failed to load encoder: {str(encoder_error)}")
                logger.error(f"Encoder error type: {type(encoder_error).__name__}")
                return False
            
            # Test the models with a sample prediction
            try:
                test_categories = list(PRICE_CATEGORIES)[:3]  # Test with first 3 categories
                for test_cat in test_categories:
                    test_df = pd.DataFrame({'Nama Item': [test_cat]})
                    test_encoded = self.encoder.transform(test_df)
                    test_prediction = self.model.predict(test_encoded)
                    logger.info(f"Test prediction for {test_cat}: {int(test_prediction[0])}")
                    break  # Only test one to verify it works
                    
                self.is_loaded = True
                logger.info("Price prediction model and encoder loaded and tested successfully")
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
    
    def predict_price(self, category: str) -> Optional[int]:
        """
        Predict price for given category
        
        Args:
            category: Category name (must be from PRICE_CATEGORIES)
            
        Returns:
            Predicted price in IDR or None if failed
        """
        # Validate category
        if not is_valid_price_category(category):
            logger.error(f"Invalid category for price prediction: {category}")
            return None
        
        if not self.is_loaded:
            logger.warning("Price prediction model not loaded, using fallback prices")
            return self._get_fallback_price(category)
        
        try:
            # Try different column names that might be expected by the model
            df = pd.DataFrame({'Nama Item': [category]})
            encoded = self.encoder.transform(df)
            prediction = self.model.predict(encoded)
            
            price = int(prediction[0])
            logger.info(f"Price prediction for {category}: {price} IDR")
            return price
            
        except KeyError:
            # Fallback to 'name' if 'Nama Item' doesn't work
            try:
                df = pd.DataFrame({'name': [category]})
                encoded = self.encoder.transform(df)
                prediction = self.model.predict(encoded)
                
                price = int(prediction[0])
                logger.info(f"Price prediction for {category}: {price} IDR (fallback column)")
                return price
                
            except Exception as e:
                logger.error(f"Price prediction failed with both column names: {str(e)}")
                logger.warning("Using fallback price due to prediction failure")
                return self._get_fallback_price(category)
        except Exception as e:
            logger.error(f"Price prediction error: {str(e)}")
            logger.warning("Using fallback price due to prediction error")
            return self._get_fallback_price(category)
    
    def _get_fallback_price(self, category: str) -> int:
        """
        Get fallback price when ML models are not available
        
        Args:
            category: Category name
            
        Returns:
            Fallback price in IDR
        """
        # Fallback prices based on typical e-waste values in IDR
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
        
        price = fallback_prices.get(category, 5000)  # Default 5000 IDR
        logger.info(f"Fallback price for {category}: {price} IDR")
        return price
    
    def get_supported_categories(self) -> List[str]:
        """Get list of supported price categories"""
        return sorted(list(PRICE_CATEGORIES))
    
    def is_category_supported(self, category: str) -> bool:
        """Check if category is supported by price model"""
        return is_valid_price_category(category)
