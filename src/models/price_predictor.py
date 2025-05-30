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
            if os.path.exists(KNR_MODEL_PATH) and os.path.exists(ENCODER_PATH):
                self.model = joblib.load(KNR_MODEL_PATH)
                self.encoder = joblib.load(ENCODER_PATH)
                self.is_loaded = True
                logger.info("Price prediction model and encoder loaded successfully")
                return True
            else:
                logger.warning(f"Model files not found: {KNR_MODEL_PATH}, {ENCODER_PATH}")
                return False
        except Exception as e:
            logger.error(f"Error loading price prediction models: {str(e)}")
            return False
    
    def predict_price(self, category: str) -> Optional[int]:
        """
        Predict price for given category
        
        Args:
            category: Category name (must be from PRICE_CATEGORIES)
            
        Returns:
            Predicted price in IDR or None if failed
        """
        if not self.is_loaded:
            logger.error("Price prediction model not loaded")
            return None
        
        # Validate category
        if not is_valid_price_category(category):
            logger.error(f"Invalid category for price prediction: {category}")
            return None
        
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
                return None
        except Exception as e:
            logger.error(f"Price prediction error: {str(e)}")
            return None
    
    def get_supported_categories(self) -> List[str]:
        """Get list of supported price categories"""
        return sorted(list(PRICE_CATEGORIES))
    
    def is_category_supported(self, category: str) -> bool:
        """Check if category is supported by price model"""
        return is_valid_price_category(category)
