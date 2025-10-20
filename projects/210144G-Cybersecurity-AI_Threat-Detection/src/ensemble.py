"""
Multi-Model Ensemble System for Network Intrusion Detection
"""

import numpy as np
import tensorflow as tf
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import Dict, List, Tuple, Any, Optional
import warnings
import os
import logging

# Suppress all warnings and TensorFlow logging
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
tf.get_logger().setLevel(logging.ERROR)  # Only show errors

# Suppress ABSL warnings (TensorFlow warnings)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from config import MODEL_PATHS, ATTACK_CATEGORIES, ENSEMBLE_CONFIG
from data_preprocessing import NSLKDDPreprocessor


class EnsembleDetector:
    """
    Multi-model ensemble system for network intrusion detection
    """
    
    def __init__(self, models_dir: str = None):
        self.models_dir = models_dir
        self.models = {}
        self.preprocessor = None
        self.attack_types = ['DoS', 'Probe', 'R2L', 'U2R']
        
    def load_models(self) -> None:
        """Load all trained models"""
        
        # Load deep learning models
        for attack_type in self.attack_types:
            model_path = MODEL_PATHS[attack_type]
            if model_path.exists():
                try:
                    # Try different loading approaches for robustness
                    try:
                        # Try loading with custom objects for focal loss
                        from load_models import focal_loss
                        custom_objects = {'focal_loss_fixed': focal_loss()}
                        model = tf.keras.models.load_model(str(model_path), custom_objects=custom_objects)
                        self.models[f'{attack_type}_dl'] = model
                        # print(f"Loaded {attack_type} deep learning model")  # COMMENTED OUT
                    except Exception as e1:
                        try:
                            # Try loading with compile=False
                            model = tf.keras.models.load_model(str(model_path), compile=False)
                            self.models[f'{attack_type}_dl'] = model
                            # print(f"Loaded {attack_type} deep learning model (compile=False)")  # COMMENTED OUT
                        except Exception as e2:
                            try:
                                # Last resort: try loading with all custom objects disabled
                                model = tf.keras.models.load_model(str(model_path), compile=False, custom_objects={})
                                self.models[f'{attack_type}_dl'] = model
                                # print(f"Loaded {attack_type} deep learning model (minimal)")  # COMMENTED OUT
                            except Exception as e3:
                                print(f"Warning: Could not load {attack_type} model:")
                                print(f"         Primary: {e1}")
                                print(f"         Secondary: {e2}")
                                print(f"         Tertiary: {e3}")
                except Exception as e:
                    print(f"Warning: Could not load {attack_type} model: {e}")
            else:
                print(f"Warning: {attack_type} model not found at {model_path}")
        
        # Load traditional ML models for Probe
        ml_models = ['rf', 'gb', 'lr']
        for model_type in ml_models:
            model_path = MODEL_PATHS[f'probe_{model_type}']
            if model_path.exists():
                try:
                    self.models[f'probe_{model_type}'] = joblib.load(str(model_path))
                    # print(f"Loaded Probe {model_type} model")  # COMMENTED OUT
                except Exception as e:
                    print(f"Warning: Could not load Probe {model_type} model: {e}")
    
    def set_preprocessor(self, preprocessor: NSLKDDPreprocessor) -> None:
        """Set the data preprocessor"""
        self.preprocessor = preprocessor
    
    def hierarchical_detection(self, X: np.ndarray) -> Tuple[List[str], List[float]]:
        """
        Hierarchical detection approach with attack-specific thresholds:
        1. Get predictions from all attack-specific models
        2. Apply specific thresholds per attack type
        3. Use weighted ensemble for Probe attack
        """
        results = []
        confidences = []
        
        for i in range(len(X)):
            sample = X[i:i+1]  # Keep 2D shape
            predictions = {}
            probabilities = {}
            
            # Get predictions from all attack-specific models
            for attack_type in self.attack_types:
                threshold = ENSEMBLE_CONFIG['attack_thresholds'][attack_type]
                
                if attack_type == 'Probe':
                    # Special weighted ensemble for Probe
                    probe_prob = self._get_probe_ensemble_prediction(sample)
                    predictions[attack_type] = probe_prob > threshold
                    probabilities[attack_type] = float(probe_prob)
                else:
                    # Single deep learning model for other attacks
                    model_key = f'{attack_type}_dl'
                    if model_key in self.models:
                        # Reshape for CNN-LSTM input
                        sample_reshaped = sample.reshape(sample.shape[0], sample.shape[1], 1)
                        prob = self.models[model_key].predict(sample_reshaped, verbose=0)[0, 0]
                        predictions[attack_type] = prob > threshold
                        probabilities[attack_type] = float(prob)
            
            # Decision logic
            positive_predictions = [k for k, v in predictions.items() if v]
            
            if len(positive_predictions) == 0:
                results.append('Normal')
                confidences.append(0.0)
            elif len(positive_predictions) == 1:
                attack_type = positive_predictions[0]
                results.append(attack_type)
                confidences.append(probabilities[attack_type])
            else:
                # Multiple predictions - choose highest probability
                best_attack = max(positive_predictions, key=lambda x: probabilities[x])
                results.append(best_attack)
                confidences.append(probabilities[best_attack])
        
        return results, confidences
    
    def _get_probe_ensemble_prediction(self, sample: np.ndarray) -> float:
        """
        Get Probe attack prediction using weighted ensemble:
        ensemble_pred = (0.4 * cnn_lstm_pred + 0.3 * rf_pred + 0.2 * gb_pred + 0.1 * lr_pred)
        """
        weights = ENSEMBLE_CONFIG['probe_ensemble_weights']
        predictions = {}
        
        # CNN-LSTM prediction
        if 'Probe_dl' in self.models:
            sample_reshaped = sample.reshape(sample.shape[0], sample.shape[1], 1)
            cnn_lstm_pred = self.models['Probe_dl'].predict(sample_reshaped, verbose=0)[0, 0]
            predictions['cnn_lstm'] = float(cnn_lstm_pred)
        else:
            predictions['cnn_lstm'] = 0.0
        
        # Traditional ML predictions
        for model_type in ['rf', 'gb', 'lr']:
            model_key = f'probe_{model_type}'
            if model_key in self.models:
                ml_pred = self.models[model_key].predict_proba(sample)[0, 1]  # Probability of positive class
                predictions[model_type] = float(ml_pred)
            else:
                predictions[model_type] = 0.0
        
        # Weighted ensemble
        ensemble_pred = (
            weights['cnn_lstm'] * predictions['cnn_lstm'] +
            weights['rf'] * predictions['rf'] +
            weights['gb'] * predictions['gb'] +
            weights['lr'] * predictions['lr']
        )
        
        return ensemble_pred
    
    def voting_detection(self, X: np.ndarray, min_votes: int = 1) -> List[str]:
        """
        Voting-based detection approach with attack-specific thresholds
        """
        results = []
        
        for i in range(len(X)):
            sample = X[i:i+1]
            votes = {}
            
            # Collect votes from all models with specific thresholds
            for attack_type in self.attack_types:
                threshold = ENSEMBLE_CONFIG['attack_thresholds'][attack_type]
                
                if attack_type == 'Probe':
                    # Use weighted ensemble for Probe
                    probe_prob = self._get_probe_ensemble_prediction(sample)
                    if probe_prob > threshold:
                        votes[attack_type] = votes.get(attack_type, 0) + 1
                else:
                    # Single deep learning model for other attacks
                    model_key = f'{attack_type}_dl'
                    if model_key in self.models:
                        sample_reshaped = sample.reshape(sample.shape[0], sample.shape[1], 1)
                        prob = self.models[model_key].predict(sample_reshaped, verbose=0)[0, 0]
                        
                        if prob > threshold:
                            votes[attack_type] = votes.get(attack_type, 0) + 1
            
            # Decision based on votes
            if not votes:
                results.append('Normal')
            else:
                # Attack type with most votes
                best_attack = max(votes.keys(), key=lambda x: votes[x])
                if votes[best_attack] >= min_votes:
                    results.append(best_attack)
                else:
                    results.append('Normal')
        
        return results
    
    def confidence_weighted_detection(self, X: np.ndarray,
                                    confidence_threshold: float = ENSEMBLE_CONFIG['confidence_threshold'],
                                    weights: Dict[str, float] = None) -> List[str]:
        """
        Confidence-weighted ensemble detection with attack-specific thresholds
        """
        if weights is None:
            weights = ENSEMBLE_CONFIG['voting_weights']
        
        results = []
        
        for i in range(len(X)):
            sample = X[i:i+1]
            confident_predictions = {}
            
            # Get high-confidence predictions with specific thresholds
            for attack_type in self.attack_types:
                threshold = ENSEMBLE_CONFIG['attack_thresholds'][attack_type]
                
                if attack_type == 'Probe':
                    # Use weighted ensemble for Probe
                    probe_prob = self._get_probe_ensemble_prediction(sample)
                    if probe_prob > confidence_threshold:
                        confident_predictions[attack_type] = probe_prob
                else:
                    # Single deep learning model for other attacks
                    model_key = f'{attack_type}_dl'
                    if model_key in self.models:
                        sample_reshaped = sample.reshape(sample.shape[0], sample.shape[1], 1)
                        prob = self.models[model_key].predict(sample_reshaped, verbose=0)[0, 0]
                        
                        if prob > confidence_threshold:
                            confident_predictions[attack_type] = prob
            
            if not confident_predictions:
                results.append('Normal')
            else:
                # Weight the confident predictions
                weighted_scores = {}
                for attack_type, pred in confident_predictions.items():
                    weighted_scores[attack_type] = pred * weights.get(attack_type, 1.0)
                
                # Select attack type with highest weighted score
                best_attack = max(weighted_scores.keys(), key=lambda x: weighted_scores[x])
                results.append(best_attack)
        
        return results
    
    def predict(self, X: np.ndarray, method: str = 'hierarchical', **kwargs) -> Tuple[List[str], Optional[List[float]]]:
        """
        Main prediction method
        
        Args:
            X: Input features
            method: Detection method ('hierarchical', 'voting', 'confidence_weighted')
            **kwargs: Additional parameters for specific methods
        
        Returns:
            Predictions and confidences (if available)
        """
        if method == 'hierarchical':
            return self.hierarchical_detection(X, **kwargs)
        elif method == 'voting':
            predictions = self.voting_detection(X, **kwargs)
            return predictions, None
        elif method == 'confidence_weighted':
            predictions = self.confidence_weighted_detection(X, **kwargs)
            return predictions, None
        else:
            raise ValueError(f"Unknown detection method: {method}")
    
    def evaluate(self, X_test: np.ndarray, y_test: List[str], 
                 method: str = 'hierarchical', **kwargs) -> Dict[str, Any]:
        """
        Evaluate ensemble performance
        """
        predictions, confidences = self.predict(X_test, method=method, **kwargs)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        
        # Get unique labels for classification report
        labels = sorted(list(set(y_test + predictions)))
        
        # Detailed metrics
        precision = precision_score(y_test, predictions, labels=labels, average='weighted', zero_division=0)
        recall = recall_score(y_test, predictions, labels=labels, average='weighted', zero_division=0)
        f1 = f1_score(y_test, predictions, labels=labels, average='weighted', zero_division=0)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': predictions,
            'confidences': confidences,
            'classification_report': classification_report(y_test, predictions, labels=labels, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, predictions, labels=labels)
        }
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {
            'total_models': len(self.models),
            'deep_learning_models': [],
            'traditional_ml_models': [],
            'missing_models': []
        }
        
        for attack_type in self.attack_types:
            model_key = f'{attack_type}_dl'
            if model_key in self.models:
                info['deep_learning_models'].append(attack_type)
            else:
                info['missing_models'].append(f'{attack_type}_dl')
        
        # Traditional ML models
        ml_models = ['rf', 'gb', 'lr']
        for model_type in ml_models:
            model_key = f'probe_{model_type}'
            if model_key in self.models:
                info['traditional_ml_models'].append(model_key)
            else:
                info['missing_models'].append(model_key)
        
        return info
