"""
LSTM Model Predictor
Loads trained LSTM model and makes predictions
"""

import numpy as np
import pandas as pd
import joblib
import os

from config_models import (
    LSTM_MODEL_PATH, SEQUENCE_SCALER_PATH,
    LSTM_PARAMS, LABEL_MAP
)

# Lazy-load TensorFlow only if model file exists and needed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
keras = None


class LSTMPredictor:
    """LSTM prediction interface"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_loaded = False
        
        self._load_model()
    
    def _load_model(self):
        """Load trained LSTM model"""
        try:
            if not os.path.exists(LSTM_MODEL_PATH):
                print("⚠️  LSTM model not found. Skipping LSTM initialization.")
                self.model_loaded = False
                return
            
            # Lazy-load TensorFlow only when model exists
            global keras
            if keras is None:
                try:
                    from tensorflow import keras
                except ImportError:
                    print("⚠️  TensorFlow not installed. LSTM unavailable (using ML models only).")
                    self.model_loaded = False
                    return
            
            self.model = keras.models.load_model(LSTM_MODEL_PATH)
            if os.path.exists(SEQUENCE_SCALER_PATH):
                self.scaler = joblib.load(SEQUENCE_SCALER_PATH)
            self.model_loaded = True
            print("✅ LSTM model loaded")
        except Exception as e:
            print(f"⚠️  LSTM initialization skipped: {e}")
            self.model_loaded = False
    
    def prepare_sequence(self, df, sequence_length=None):
        """
        Prepare sequence from dataframe for LSTM prediction
        
        Args:
            df: DataFrame with technical indicators
            sequence_length: Length of sequence (default from config)
        
        Returns:
            Scaled sequence ready for prediction
        """
        if sequence_length is None:
            sequence_length = LSTM_PARAMS['sequence_length']
        
        # Select same features as training
        feature_cols = ['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 
                       'EMA_9', 'EMA_21', 'EMA_50', 'Volume_Ratio', 'ATR']
        
        # Get last sequence_length rows
        if len(df) < sequence_length:
            raise ValueError(f"Need at least {sequence_length} data points, got {len(df)}")
        
        data = df[feature_cols].iloc[-sequence_length:].values
        
        # Scale data
        data_scaled = self.scaler.transform(data)
        
        # Reshape for LSTM: (1, sequence_length, n_features)
        sequence = data_scaled.reshape(1, sequence_length, -1)
        
        return sequence
    
    def predict(self, df):
        """
        Make prediction using LSTM model
        
        Args:
            df: DataFrame with technical indicators
        
        Returns:
            dict with prediction, confidence, and probabilities
        """
        if not self.model_loaded:
            return {
                'available': False,
                'error': 'LSTM model not trained yet'
            }
        
        try:
            # Prepare sequence
            sequence = self.prepare_sequence(df)
            
            # Get prediction
            pred_proba = self.model.predict(sequence, verbose=0)[0]
            pred_label = int(np.argmax(pred_proba))
            
            # Get confidence
            confidence = float(pred_proba[pred_label])
            
            # Convert label to signal
            signal = LABEL_MAP[pred_label]
            
            return {
                'available': True,
                'signal': signal,
                'confidence': confidence,
                'probabilities': {
                    'BEARISH': float(pred_proba[0]),
                    'NEUTRAL': float(pred_proba[1]),
                    'BULLISH': float(pred_proba[2])
                },
                'model': 'LSTM Deep Learning'
            }
        
        except Exception as e:
            return {
                'available': False,
                'error': str(e)
            }
    
    def predict_trend(self, df, horizon=5):
        """
        Predict trend over multiple time horizons
        
        Args:
            df: DataFrame with technical indicators
            horizon: Number of steps ahead to predict
        
        Returns:
            List of predictions for each time step
        """
        if not self.model_loaded:
            return None
        
        try:
            predictions = []
            sequence = self.prepare_sequence(df)
            
            for _ in range(horizon):
                pred_proba = self.model.predict(sequence, verbose=0)[0]
                predictions.append(pred_proba)
            
            return predictions
        except:
            return None


# Global predictor instance
_lstm_predictor = None

def get_lstm_predictor():
    """Get or create global LSTM predictor instance"""
    global _lstm_predictor
    if _lstm_predictor is None:
        _lstm_predictor = LSTMPredictor()
    return _lstm_predictor
