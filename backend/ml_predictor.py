"""
ML Model Predictor
Loads trained models and makes predictions for stock signals
"""

import numpy as np
import joblib
import pandas as pd
import os
import sys
import json

# Add parent directory to path to allow importing from ml_training
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_training.utils import extract_intraday_features

from config_models import (
    INTRADAY_MODEL_PATH,
    LABEL_MAP, DEFAULT_MODEL
)


class MLPredictor:
    """Machine Learning prediction interface"""
    
    def __init__(self):
        self.models = {}
        self.model_info = {}
        self.models_loaded = False
        
        self._load_models()
    
    def _load_models(self):
        """Load only the Intraday model for production (Vercel size limit)"""
        try:
            # Load only Intraday model for Vercel deployment (others available locally)
            if os.path.exists(INTRADAY_MODEL_PATH):
                self.models['INTRADAY'] = joblib.load(INTRADAY_MODEL_PATH)
                self._load_metadata('INTRADAY', INTRADAY_MODEL_PATH)
                print("✅ Intraday model loaded (production mode)")
                self.models_loaded = True
            else:
                print("⚠️  Intraday model not found")
                self.models_loaded = False
        
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.models_loaded = False
            
    def _load_metadata(self, timeframe, model_path):
        """Load feature names from model metadata"""
        try:
            # Assume metadata file is named {model_name}_info.json
            info_path = model_path.replace('.pkl', '_info.json')
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    info = json.load(f)
                    self.model_info[timeframe] = info.get('features', [])
            else:
                print(f"⚠️  Metadata not found for {timeframe}")
                self.model_info[timeframe] = []
        except Exception as e:
            print(f"⚠️  Error loading metadata for {timeframe}: {e}")
    
    def predict(self, df, model_type='ensemble', strategy_mode='INTRADAY'):
        """
        Make prediction using ML models (Intraday only in production)
        
        Args:
            df: DataFrame with technical indicators
            model_type: Unused (kept for compatibility)
            strategy_mode: Requested mode (falls back to INTRADAY if not available)
        
        Returns:
            dict with prediction, confidence, and model info
        """
        if not self.models_loaded:
            return {
                'available': False,
                'error': 'No models loaded'
            }
        
        # Fallback to INTRADAY if requested model not available (production mode)
        if strategy_mode not in self.models:
            strategy_mode = 'INTRADAY'
        
        if strategy_mode not in self.models:
            return {
                'available': False,
                'error': f'Model {strategy_mode} not available'
            }
        
        try:
            # 1. Extract Features (use Intraday for all modes in production)
            current_idx = len(df) - 1
            features_dict = extract_intraday_features(df, current_idx)

            if not features_dict:
                 return {'available': False, 'error': 'Feature extraction failed'}
            
            # 2. Convert to DataFrame with correct column order
            feature_names = self.model_info.get(strategy_mode, [])
            if not feature_names:
                # If no metadata, try to use keys from dict (risky order)
                cols = list(features_dict.keys())
            else:
                cols = feature_names
                
            # Create DataFrame ensuring all columns exist
            row_data = {}
            for col in cols:
                row_data[col] = features_dict.get(col, 0.0)
                
            X = pd.DataFrame([row_data])
            
            # 3. Predict
            model = self.models[strategy_mode]
            
            # Sklearn models (RandomForest/XGBoost) return class 0/1 (or 0/1/2?)
            # The training script used labels from 0/1 or similar.
            # 3_train_models.py uses 'label' column.
            # 2_generate_training_data used 0 (Bearish), 1 (Success? No, let's check).
            # RUN_PIPELINE.md says: Success/Failure. Binary?
            # 3_train_models: "Class distribution: {1: 6841, 0: 5183}". So it is Binary (0/1).
            # 1 = Success (Buy signal worked), 0 = Failure.
            # So prediction is Probability of Success.
            
            # Get probability of class 1
            pred_proba = model.predict_proba(X)[0]
            
            # If binary (Failure/Success)
            if len(pred_proba) == 2:
                success_prob = pred_proba[1]
                
                # We interpret High Success Prob as BULLISH (since we only look for Buy setups currently?)
                # Wait, the training data generation checks for "Target Hit".
                # If the strategy was searching for Buys, then 1=Bullish Success.
                # If we assume the input `df` is a potential setup...
                # Actually, the `strategies.py` logic finds potential setups first?
                # No, `strategies.py` calls `intraday_logic_ai` which calls `manager.predict(df)`.
                # So we run the model on *every* candle?
                # Yes.
                # If the model predicts "1" (Success), it means "If you buy now, it will likely hit target".
                # So Signal = BULLISH if conf > threshold.
                
                signal = 'BULLISH' if success_prob > 0.5 else 'NEUTRAL' # Or BEARISH? 
                # If prob is low, it just means "Not a buy". calling it NEUTRAL is safer.
                
                return {
                    'available': True,
                    'signal': signal,
                    'confidence': float(success_prob),
                    'probabilities': {
                        'BEARISH': float(pred_proba[0]),
                        'NEUTRAL': 0.0,
                        'BULLISH': float(pred_proba[1])
                    },
                    'model': f'{strategy_mode} Model'
                }
            else:
                # Multiclass (0, 1, 2)
                 return {
                    'available': False,
                    'error': f'Unexpected class count: {len(pred_proba)}'
                }

        except Exception as e:
            return {
                'available': False,
                'error': str(e)
            }

    
    def get_feature_importance(self):
        """Not implemented for multi-model setup yet"""
        return None


# Global predictor instance
_predictor = None

def get_predictor():
    """Get or create global predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = MLPredictor()
    return _predictor
