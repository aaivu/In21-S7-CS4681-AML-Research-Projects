import sys
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add current directory to Python path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from data_preprocessing import NSLKDDPreprocessor
from ensemble import EnsembleDetector
from config import TRAIN_DATA_PATH, TEST_DATA_PATH, MODELS_DIR, ATTACK_CATEGORIES

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report


class MetaClassifierTrainer:
    """Train and manage meta-classifiers for stacking ensemble"""
    
    def __init__(self):
        self.meta_models_dir = Path(__file__).parent.parent / "meta_models"
        self.meta_models_dir.mkdir(exist_ok=True)
        
        self.meta_classifiers = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(probability=True, random_state=42, kernel='rbf')
        }
        
    def load_base_models_and_data(self, train_size: int = 10000, test_size: int = 5000):
        """Load base models and prepare training data for meta-classifier"""
        
        print("="*70)
        print("META-CLASSIFIER TRAINING PIPELINE")
        print("="*70)
        
        print("Loading base ensemble models...")
        ensemble = EnsembleDetector()
        ensemble.load_models()
        
        model_info = ensemble.get_model_info()
        print(f"Loaded {model_info['total_models']} base models")
        
        print("Loading and preprocessing data...")
        preprocessor = NSLKDDPreprocessor()
        
        # Use subset for faster meta-classifier training
        train_df = pd.read_csv(TRAIN_DATA_PATH, header=None, names=preprocessor.feature_columns or list(range(42)))
        test_df = pd.read_csv(TEST_DATA_PATH, header=None, names=preprocessor.feature_columns or list(range(42)))
        
        if train_size and len(train_df) > train_size:
            train_df = train_df.sample(n=train_size, random_state=42)
            print(f"Using {train_size} training samples for meta-classifier")
        
        if test_size and len(test_df) > test_size:
            test_df = test_df.sample(n=test_size, random_state=42)
            print(f"Using {test_size} test samples for meta-classifier")
        
        # Preprocess data
        train_processed = preprocessor.preprocess_for_evaluation_only(TRAIN_DATA_PATH, TEST_DATA_PATH)
        
        # Get subset matching our samples
        if train_size:
            train_processed = train_processed.iloc[:train_size]
        if test_size:
            test_processed = train_processed.iloc[-test_size:] if test_size < len(train_processed) else train_processed.copy()
        else:
            test_processed = train_processed.copy()
        
        return ensemble, train_processed, test_processed
    
    def generate_base_model_predictions(self, ensemble: EnsembleDetector, data_processed: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Generate predictions from all base models"""
        
        print("Generating base model predictions...")
        
        # Prepare data
        X = data_processed.drop('attack', axis=1).values
        X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)  # For CNN-LSTM models
        
        model_predictions = {}
        feature_names = []
        
        # Deep learning models
        for attack_type in ['DoS', 'Probe', 'R2L', 'U2R']:
            model_key = f'{attack_type}_dl'
            if model_key in ensemble.models:
                print(f"  Generating {attack_type} DL predictions...")
                pred = ensemble.models[model_key].predict(X_reshaped, batch_size=256, verbose=0)
                model_predictions[model_key] = pred.flatten()
                feature_names.append(model_key)
        
        # Traditional ML models for Probe
        probe_models = ['probe_rf', 'probe_gb', 'probe_lr']
        for model_key in probe_models:
            if model_key in ensemble.models:
                print(f"  Generating {model_key} predictions...")
                pred = ensemble.models[model_key].predict_proba(X)[:, 1]
                model_predictions[model_key] = pred
                feature_names.append(model_key)
        
        # Create feature matrix
        stacking_features = []
        for feature_name in feature_names:
            stacking_features.append(model_predictions[feature_name])
        
        X_meta = np.column_stack(stacking_features)
        
        print(f"Generated predictions from {len(feature_names)} base models")
        print(f"Meta-features shape: {X_meta.shape}")
        
        return X_meta, feature_names
    
    def prepare_binary_labels(self, data_processed: pd.DataFrame) -> np.ndarray:
        """Convert multi-class labels to binary (Normal vs Attack)"""
        attack_labels = data_processed['attack'].values
        binary_labels = (attack_labels != 0).astype(int)  # 0 = Normal, 1 = Attack
        
        normal_count = np.sum(binary_labels == 0)
        attack_count = np.sum(binary_labels == 1)
        
        print(f"Binary labels prepared:")
        print(f"  Normal: {normal_count} ({normal_count/len(binary_labels)*100:.1f}%)")
        print(f"  Attack: {attack_count} ({attack_count/len(binary_labels)*100:.1f}%)")
        
        return binary_labels
    
    def train_meta_classifiers(self, X_meta: np.ndarray, y_binary: np.ndarray, feature_names: List[str]):
        """Train multiple meta-classifiers and save the best ones"""
        
        print(f"\n{'-'*50}")
        print("TRAINING META-CLASSIFIERS")
        print(f"{'-'*50}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_meta, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        results = {}
        best_model = None
        best_score = 0
        best_name = None
        
        for name, classifier in self.meta_classifiers.items():
            print(f"\nTraining {name.replace('_', ' ').title()}...")
            
            try:
                # Cross-validation
                cv_scores = cross_val_score(classifier, X_train, y_train, cv=5, scoring='f1')
                print(f"  Cross-validation F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
                # Train on full training set
                classifier.fit(X_train, y_train)
                
                # Evaluate on test set
                y_pred = classifier.predict(X_test)
                y_pred_proba = classifier.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                auc = roc_auc_score(y_test, y_pred_proba)
                
                results[name] = {
                    'classifier': classifier,
                    'cv_f1_mean': cv_scores.mean(),
                    'cv_f1_std': cv_scores.std(),
                    'test_accuracy': accuracy,
                    'test_precision': precision,
                    'test_recall': recall,
                    'test_f1': f1,
                    'test_auc': auc
                }
                
                print(f"  Test Results:")
                print(f"    Accuracy:  {accuracy:.4f}")
                print(f"    Precision: {precision:.4f}")
                print(f"    Recall:    {recall:.4f}")
                print(f"    F1-Score:  {f1:.4f}")
                print(f"    ROC-AUC:   {auc:.4f}")
                
                # Track best model
                if f1 > best_score:
                    best_score = f1
                    best_model = classifier
                    best_name = name
                
                # Feature importance/coefficients
                if hasattr(classifier, 'feature_importances_'):
                    importances = classifier.feature_importances_
                    print(f"  Feature Importances:")
                    for i, (fname, importance) in enumerate(zip(feature_names, importances)):
                        print(f"    {fname}: {importance:.4f}")
                
                elif hasattr(classifier, 'coef_'):
                    coefficients = classifier.coef_[0]
                    print(f"  Feature Coefficients:")
                    for fname, coef in zip(feature_names, coefficients):
                        print(f"    {fname}: {coef:.4f}")
                
            except Exception as e:
                print(f"  Error training {name}: {e}")
                results[name] = {'error': str(e)}
        
        # Save all models
        print(f"\n{'-'*50}")
        print("SAVING META-CLASSIFIERS")
        print(f"{'-'*50}")
        
        for name, result in results.items():
            if 'error' not in result:
                model_path = self.meta_models_dir / f"anomaly_meta_{name}.pkl"
                
                # Save model with metadata
                model_data = {
                    'classifier': result['classifier'],
                    'feature_names': feature_names,
                    'metrics': {k: v for k, v in result.items() if k != 'classifier'},
                    'training_info': {
                        'train_samples': len(X_train),
                        'test_samples': len(X_test),
                        'n_features': X_meta.shape[1]
                    }
                }
                
                with open(model_path, 'wb') as f:
                    pickle.dump(model_data, f)
                
                print(f"Saved: {model_path}")
        
        # Save best model info
        if best_model is not None:
            best_model_info = {
                'best_model_name': best_name,
                'best_f1_score': best_score,
                'feature_names': feature_names,
                'all_results': {name: {k: v for k, v in result.items() if k != 'classifier'} 
                              for name, result in results.items() if 'error' not in result}
            }
            
            best_info_path = self.meta_models_dir / "best_meta_classifier_info.pkl"
            with open(best_info_path, 'wb') as f:
                pickle.dump(best_model_info, f)
            
            print(f"\nBest Model: {best_name} (F1: {best_score:.4f})")
            print(f"Best model info saved: {best_info_path}")
        
        return results


def main():
    """Main training pipeline"""
    trainer = MetaClassifierTrainer()
    
    # Load models and data
    ensemble, train_data, test_data = trainer.load_base_models_and_data(
        train_size=10000,  # Use subset for faster training
        test_size=3000
    )
    
    # Generate base model predictions
    X_meta_train, feature_names = trainer.generate_base_model_predictions(ensemble, train_data)
    y_binary_train = trainer.prepare_binary_labels(train_data)
    
    # Train meta-classifiers
    results = trainer.train_meta_classifiers(X_meta_train, y_binary_train, feature_names)
    
    print("\n" + "="*70)
    print("META-CLASSIFIER TRAINING COMPLETED")
    print("="*70)
    print(f"Models saved in: {trainer.meta_models_dir}")
    print("Ready for ensemble evaluation!")


if __name__ == "__main__":
    main()
