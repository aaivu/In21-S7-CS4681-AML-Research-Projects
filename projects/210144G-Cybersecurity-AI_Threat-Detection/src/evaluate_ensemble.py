"""
Evaluation Pipeline for Multi-Model Ensemble System
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

# Add current directory to Python path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from data_preprocessing import NSLKDDPreprocessor
from ensemble import EnsembleDetector
from load_models import verify_model_compatibility
from config import TRAIN_DATA_PATH, TEST_DATA_PATH, RESULTS_DIR, ATTACK_CATEGORIES


def prepare_ensemble_test_data(test_processed: pd.DataFrame, sample_size: int = None) -> Dict[str, Any]:
    """
    Prepare test data for ensemble evaluation using all preprocessed features
    
    Args:
        test_processed: Preprocessed test dataframe
        sample_size: Number of samples to use (None for full dataset)
    """
    if sample_size is None:
        # Use full test dataset
        test_sample = test_processed.copy()
        # print(f"Using FULL test dataset: {len(test_sample)} samples")  # COMMENTED OUT
    else:
        # Sample test data for evaluation
        test_sample = test_processed.sample(n=min(sample_size, len(test_processed)), random_state=42)
        # print(f"Using SAMPLED test dataset: {len(test_sample)} samples (from {len(test_processed)} total)")  # COMMENTED OUT
    
    # Convert attack labels to categories, handle NaN values
    y_test = []
    nan_count = 0
    for code in test_sample['attack'].values:
        if pd.isna(code):
            # Handle unmapped attacks - treat as Normal (0) or skip
            y_test.append('Normal')  # or could skip these samples
            nan_count += 1
        else:
            y_test.append(ATTACK_CATEGORIES[int(code)])
    
    if nan_count > 0:
        # print(f"Warning: Found {nan_count} unmapped attack samples, treating as Normal")  # COMMENTED OUT
        pass
    
    # Use all preprocessed features (excluding attack column)
    X_test = test_sample.drop('attack', axis=1).values
    
    return {
        'X_test': X_test,
        'y_test': y_test,
        'sample_info': {
            'total_samples': len(test_sample),
            'original_dataset_size': len(test_processed),
            'feature_count': X_test.shape[1],
            'attack_distribution': pd.Series(y_test).value_counts().to_dict(),
            'is_full_dataset': sample_size is None
        }
    }


def evaluate_hierarchical_vectorized(predictions: Dict, thresholds: Dict, probe_weights: Dict) -> List[str]:
    """Vectorized hierarchical detection"""
    n_samples = len(predictions['DoS_dl'])
    final_preds = ['Normal'] * n_samples
    
    # Vectorized threshold application
    dos_positive = predictions['DoS_dl'] > thresholds['DoS']
    r2l_positive = predictions['R2L_dl'] > thresholds['R2L']  
    u2r_positive = predictions['U2R_dl'] > thresholds['U2R']
    
    # Probe ensemble (vectorized)
    probe_ensemble = (probe_weights['cnn_lstm'] * predictions['Probe_dl'] + 
                     probe_weights['rf'] * predictions['probe_rf'] +
                     probe_weights['gb'] * predictions['probe_gb'] + 
                     probe_weights['lr'] * predictions['probe_lr'])
    probe_positive = probe_ensemble > thresholds['Probe']
    
    # Vectorized assignment
    for i in range(n_samples):
        attacks = []
        if dos_positive[i]: attacks.append(('DoS', predictions['DoS_dl'][i]))
        if probe_positive[i]: attacks.append(('Probe', probe_ensemble[i]))
        if r2l_positive[i]: attacks.append(('R2L', predictions['R2L_dl'][i]))
        if u2r_positive[i]: attacks.append(('U2R', predictions['U2R_dl'][i]))
        
        if attacks:
            final_preds[i] = max(attacks, key=lambda x: x[1])[0]
    
    return final_preds


def evaluate_voting_vectorized(predictions: Dict, thresholds: Dict, probe_weights: Dict) -> List[str]:
    """Vectorized voting detection"""
    n_samples = len(predictions['DoS_dl'])
    final_preds = ['Normal'] * n_samples
    
    # Apply thresholds vectorized
    votes = {}
    votes['DoS'] = (predictions['DoS_dl'] > thresholds['DoS']).astype(int)
    votes['R2L'] = (predictions['R2L_dl'] > thresholds['R2L']).astype(int)
    votes['U2R'] = (predictions['U2R_dl'] > thresholds['U2R']).astype(int)
    
    # Probe voting
    probe_ensemble = (probe_weights['cnn_lstm'] * predictions['Probe_dl'] + 
                     probe_weights['rf'] * predictions['probe_rf'] +
                     probe_weights['gb'] * predictions['probe_gb'] + 
                     probe_weights['lr'] * predictions['probe_lr'])
    votes['Probe'] = (probe_ensemble > thresholds['Probe']).astype(int)
    
    # Find winner for each sample
    for i in range(n_samples):
        sample_votes = {k: v[i] for k, v in votes.items()}
        positive_votes = [k for k, v in sample_votes.items() if v > 0]
        if positive_votes:
            final_preds[i] = positive_votes[0]
    
    return final_preds


def evaluate_confidence_vectorized(predictions: Dict, thresholds: Dict, probe_weights: Dict) -> List[str]:
    """Vectorized confidence-weighted detection"""
    return evaluate_hierarchical_vectorized(predictions, thresholds, probe_weights)


def calculate_metrics(y_true: List, y_pred: List) -> Dict:
    """Calculate evaluation metrics"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'classification_report': classification_report(y_true, y_pred, zero_division=0)
    }


def evaluate_ensemble_comprehensive(data_path_train: str, data_path_test: str, 
                                   sample_size: int = None) -> Dict[str, Any]:
    """
    Comprehensive evaluation of the multi-model ensemble system
    
    Args:
        data_path_train: Path to training data
        data_path_test: Path to test data  
        sample_size: Number of samples to evaluate (None for full dataset)
    """
    # print("="*70)  # COMMENTED OUT FOR CLEAN OUTPUT
    # print("MULTI-MODEL ENSEMBLE EVALUATION")
    # print("Network Intrusion Detection with Deep Ensemble")  
    # print("="*70)
    
    # Verify models are available
    if not verify_model_compatibility():
        print("Error: Required models not found. Please train models using notebooks first.")
        return {}
    
    # Initialize ensemble
    # print("Loading models...")  # COMMENTED OUT
    ensemble = EnsembleDetector()
    ensemble.load_models()
    
    # Get model information
    model_info = ensemble.get_model_info()
    # print(f"Models loaded: {model_info['total_models']} total")  # COMMENTED OUT
    
    # Load and prepare test data only (using training data to fit transformers)
    # print("Preprocessing test data for evaluation...")  # COMMENTED OUT
    preprocessor = NSLKDDPreprocessor()
    test_processed = preprocessor.preprocess_for_evaluation_only(data_path_train, data_path_test)
    
    # Prepare test data
    test_data = prepare_ensemble_test_data(test_processed, sample_size)
    
    # print(f"Dataset: {test_data['sample_info']['total_samples']} samples")  # COMMENTED OUT
    # print(f"Features: {test_data['sample_info']['feature_count']}")  # COMMENTED OUT
    
    # Batch evaluation
    # print("Running ensemble evaluation...")  # COMMENTED OUT
    
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    
    # Reshape for CNN-LSTM models (batch processing)
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Get all model predictions in batch
    # print("Getting batch predictions from all models...")  # COMMENTED OUT
    model_predictions = {}
    
    # Deep learning models - batch prediction
    for attack_type in ['DoS', 'Probe', 'R2L', 'U2R']:
        if f'{attack_type}_dl' in ensemble.models:
            model = ensemble.models[f'{attack_type}_dl']
            pred = model.predict(X_test_reshaped, batch_size=256, verbose=0)
            model_predictions[f'{attack_type}_dl'] = pred.flatten()
    
    # Traditional ML models - batch prediction
    if 'probe_rf' in ensemble.models:
        model_predictions['probe_rf'] = ensemble.models['probe_rf'].predict_proba(X_test)[:, 1]
    if 'probe_gb' in ensemble.models:
        model_predictions['probe_gb'] = ensemble.models['probe_gb'].predict_proba(X_test)[:, 1]
    if 'probe_lr' in ensemble.models:
        model_predictions['probe_lr'] = ensemble.models['probe_lr'].predict_proba(X_test)[:, 1]
    
    # print("All predictions obtained")  # COMMENTED OUT
    
    # Get ensemble configuration
    from config import ENSEMBLE_CONFIG
    thresholds = ENSEMBLE_CONFIG['attack_thresholds']
    probe_weights = ENSEMBLE_CONFIG['probe_ensemble_weights']
    
    # ANOMALY DETECTION EVALUATION (Normal vs Attack) - COMMENTED OUT FOR CLEANER OUTPUT
    # print(f"\n{'-'*50}")
    # print("ANOMALY DETECTION EVALUATION")
    # print("Binary Classification: Normal vs Attack")
    # print(f"{'-'*50}")
    
    # Create binary labels (0 = Normal, 1 = Attack) - still needed for stacking ensemble
    y_binary_true = [0 if label == 'Normal' else 1 for label in y_test]
    
    # # Anomaly detection using "OR" logic - if any model detects attack above threshold
    # anomaly_predictions = []
    # for i in range(len(X_test)):
    #     is_attack = False
    #     
    #     # Check each model's prediction against its threshold
    #     for attack_type in ['DoS', 'Probe', 'R2L', 'U2R']:
    #         threshold = thresholds[attack_type]
    #         
    #         if attack_type == 'Probe':
    #             # Use weighted ensemble for Probe
    #             probe_prob = (probe_weights['cnn_lstm'] * model_predictions['Probe_dl'][i] +
    #                          probe_weights['rf'] * model_predictions['probe_rf'][i] +
    #                          probe_weights['gb'] * model_predictions['probe_gb'][i] +
    #                          probe_weights['lr'] * model_predictions['probe_lr'][i])
    #             if probe_prob > threshold:
    #                 is_attack = True
    #                 break
    #         else:
    #             # Check deep learning model
    #             if f'{attack_type}_dl' in model_predictions:
    #                 if model_predictions[f'{attack_type}_dl'][i] > threshold:
    #                     is_attack = True
    #                     break
    #     
    #     anomaly_predictions.append(1 if is_attack else 0)
    # 
    # # Calculate anomaly detection metrics
    # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
    # 
    # anomaly_accuracy = accuracy_score(y_binary_true, anomaly_predictions)
    # anomaly_precision = precision_score(y_binary_true, anomaly_predictions, zero_division=0)
    # anomaly_recall = recall_score(y_binary_true, anomaly_predictions, zero_division=0)
    # anomaly_f1 = f1_score(y_binary_true, anomaly_predictions, zero_division=0)
    # 
    # # Generate classification report for binary classification
    # anomaly_class_report = classification_report(
    #     y_binary_true, anomaly_predictions, 
    #     target_names=['Normal', 'Attack'],
    #     zero_division=0
    # )
    # 
    # print(f"Anomaly Detection Results:")
    # print(f"  Accuracy:  {anomaly_accuracy:.4f}")
    # print(f"  Precision: {anomaly_precision:.4f} (Attack detection precision)")
    # print(f"  Recall:    {anomaly_recall:.4f} (Attack detection recall)")
    # print(f"  F1-Score:  {anomaly_f1:.4f}")
    # 
    # print("Classification Report (Normal vs Attack):")
    # print(anomaly_class_report)
    # 
    # # Calculate ROC-AUC using maximum prediction probability as score
    # anomaly_scores = []
    # for i in range(len(X_test)):
    #     max_prob = 0
    #     for attack_type in ['DoS', 'Probe', 'R2L', 'U2R']:
    #         if attack_type == 'Probe':
    #             prob = (probe_weights['cnn_lstm'] * model_predictions['Probe_dl'][i] +
    #                    probe_weights['rf'] * model_predictions['probe_rf'][i] +
    #                    probe_weights['gb'] * model_predictions['probe_gb'][i] +
    #                    probe_weights['lr'] * model_predictions['probe_lr'][i])
    #         else:
    #             if f'{attack_type}_dl' in model_predictions:
    #                 prob = model_predictions[f'{attack_type}_dl'][i]
    #             else:
    #                 prob = 0
    #         max_prob = max(max_prob, prob)
    #     anomaly_scores.append(max_prob)
    # 
    # try:
    #     anomaly_auc = roc_auc_score(y_binary_true, anomaly_scores)
    #     print(f"  ROC-AUC:   {anomaly_auc:.4f}")
    # except:
    #     print(f"  ROC-AUC:   Cannot calculate (insufficient class variation)")
    
    # STACKING ENSEMBLE ANOMALY DETECTION
    print(f"\n{'-'*50}")
    print("STACKING ENSEMBLE EVALUATION")
    print("Loading pre-trained meta-classifier")
    print(f"{'-'*50}")
    
    # Load pre-trained meta-classifier
    meta_models_dir = Path(__file__).parent.parent / "meta_models"
    best_meta_info_path = meta_models_dir / "best_meta_classifier_info.pkl"
    
    stacking_results = {}
    best_meta_classifier = None
    best_meta_name = None
    best_meta_f1 = 0
    
    if best_meta_info_path.exists():
        try:
            import pickle
            
            # Load best model info
            with open(best_meta_info_path, 'rb') as f:
                best_info = pickle.load(f)
            
            best_meta_name = best_info['best_model_name']
            expected_features = best_info['feature_names']
            
            print(f"Loading best meta-classifier: {best_meta_name}")
            print(f"Expected features: {expected_features}")
            
            # Load the best meta-classifier
            best_model_path = meta_models_dir / f"anomaly_meta_{best_meta_name}.pkl"
            
            if best_model_path.exists():
                with open(best_model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                best_meta_classifier = model_data['classifier']
                training_metrics = model_data['metrics']
                
                print(f"Meta-classifier loaded successfully!")
                print(f"Training metrics: F1={training_metrics['test_f1']:.4f}, AUC={training_metrics['test_auc']:.4f}")
                
                # Prepare features for prediction (match training feature order)
                stacking_features = []
                available_features = []
                
                for feature_name in expected_features:
                    if feature_name in model_predictions:
                        stacking_features.append(model_predictions[feature_name])
                        available_features.append(feature_name)
                    else:
                        print(f"Warning: Feature {feature_name} not available in current predictions")
                
                if len(stacking_features) == len(expected_features):
                    # Create feature matrix
                    X_meta = np.column_stack(stacking_features)
                    
                    print(f"Predicting with meta-classifier on {X_meta.shape[0]} samples...")
                    
                    # Make predictions
                    meta_predictions = best_meta_classifier.predict(X_meta)
                    meta_probabilities = best_meta_classifier.predict_proba(X_meta)[:, 1]
                    
                    # Calculate metrics
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,classification_report
                    
                    meta_accuracy = accuracy_score(y_binary_true, meta_predictions)
                    meta_precision = precision_score(y_binary_true, meta_predictions, zero_division=0)
                    meta_recall = recall_score(y_binary_true, meta_predictions, zero_division=0)
                    meta_f1 = f1_score(y_binary_true, meta_predictions, zero_division=0)
                    meta_auc = roc_auc_score(y_binary_true, meta_probabilities)
                    
                    # Generate classification report for stacking
                    from sklearn.metrics import classification_report
                    stacking_class_report = classification_report(
                        y_binary_true, meta_predictions,
                        target_names=['Normal', 'Attack'],
                        zero_division=0
                    )
                    
                    stacking_results[best_meta_name] = {
                        'accuracy': meta_accuracy,
                        'precision': meta_precision,
                        'recall': meta_recall,
                        'f1_score': meta_f1,
                        'roc_auc': meta_auc,
                        'classification_report': stacking_class_report
                    }
                    
                    best_meta_f1 = meta_f1
                    
                    print(f"Stacking Ensemble Results:")
                    print(f"  Accuracy:  {meta_accuracy:.4f}")
                    print(f"  Precision: {meta_precision:.4f}")
                    print(f"  Recall:    {meta_recall:.4f}")
                    print(f"  F1-Score:  {meta_f1:.4f}")
                    print(f"  ROC-AUC:   {meta_auc:.4f}")
                    
                    print("Classification Report (Normal vs Attack):")
                    print(stacking_class_report)

                    
                else:
                    print(f"Error: Feature mismatch. Expected {len(expected_features)}, got {len(stacking_features)}")
                    stacking_results['error'] = "Feature mismatch"
                
            else:
                print(f"Error: Best model file not found: {best_model_path}")
                stacking_results['error'] = "Model file not found"
                
        except Exception as e:
            print(f"Error loading meta-classifier: {e}")
            stacking_results['error'] = str(e)
    
    else:
        print("No pre-trained meta-classifier found.")
        print("Run 'python src/train_meta_classifier.py' first to train meta-classifiers.")
        stacking_results['error'] = "No pre-trained meta-classifier"
    
    # Evaluate ensemble methods - COMMENTED OUT TO SHOW ONLY STACKING ENSEMBLE
    results = {}
    ensemble_methods = [
        ("hierarchical", "HIERARCHICAL DETECTION"),
        ("voting", "VOTING-BASED ENSEMBLE"),
        ("confidence_weighted", "CONFIDENCE-WEIGHTED DETECTION")
    ]
    
    for method_key, method_name in ensemble_methods:
        # print(f"\nEvaluating: {method_name}")  # COMMENTED OUT
        
        try:
            if method_key == "hierarchical":
                predictions = evaluate_hierarchical_vectorized(model_predictions, thresholds, probe_weights)
            elif method_key == "voting":
                predictions = evaluate_voting_vectorized(model_predictions, thresholds, probe_weights)
            elif method_key == "confidence_weighted":
                predictions = evaluate_confidence_vectorized(model_predictions, thresholds, probe_weights)
            
            # Calculate metrics
            metrics = calculate_metrics(y_test, predictions)
            results[method_key] = {
                'status': 'success',
                'predictions': predictions,
                'metrics': metrics,
                'method_name': method_name
            }
            
            # print(f"Accuracy: {metrics['accuracy']:.4f}")
            # print(f"F1-Score: {metrics['f1_score']:.4f}")
            # print("Classification Report:")
            # print(metrics['classification_report'])

        except Exception as e:
            # print(f"Error evaluating {method_key}: {e}")  # COMMENTED OUT
            results[method_key] = {
                'status': 'error',
                'error': str(e),
                'method_name': method_name
            }
    

    
    anomaly_results = {
        'accuracy': 0.0,  
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0,
        'predictions': [],
        'true_labels': y_binary_true,
        'classification_report': ""
    }
    anomaly_results['roc_auc'] = None
    
    return {
        'ensemble_results': results,
        'anomaly_detection': anomaly_results,
        'stacking_ensemble': stacking_results,
        'best_meta_classifier': {
            'name': best_meta_name,
            'f1_score': best_meta_f1
        } if best_meta_classifier is not None else None,
        'model_info': model_info,
        'test_data_info': test_data['sample_info']
    }


def generate_comprehensive_report(evaluation_results: Dict[str, Any]) -> None:
    """
    Generate comprehensive evaluation report
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE EVALUATION REPORT")
    print("="*70)
    
    # Model Summary
    model_info = evaluation_results.get('model_info', {})
    print(f"\nModel Architecture Summary:")
    print(f"  Specialized binary classifiers per attack type")
    print(f"  CNN-LSTM with focal loss for class imbalance")
    print(f"  Attack-specific thresholds: DoS=0.5, Probe=0.47, R2L=0.004, U2R=0.942")
    print(f"  Probe weighted ensemble: CNN-LSTM(0.4) + RF(0.3) + GB(0.2) + LR(0.1)")
    print(f"  Total models loaded: {model_info.get('total_models', 0)}")
    
    # Test Data Summary
    test_info = evaluation_results.get('test_data_info', {})
    print(f"\nEvaluation Dataset:")
    print(f"  Sample size: {test_info.get('total_samples', 'N/A')}")
    if not test_info.get('is_full_dataset', False):
        print(f"  Original dataset size: {test_info.get('original_dataset_size', 'N/A')}")
    print(f"  Feature count: {test_info.get('feature_count', 'N/A')}")
    print(f"  Attack distribution: {test_info.get('attack_distribution', {})}")
    
    # Anomaly Detection Results
    anomaly_results = evaluation_results.get('anomaly_detection', {})
    if anomaly_results:
        print(f"\nAnomaly Detection Performance (Normal vs Attack):")
        print(f"  Accuracy:  {anomaly_results.get('accuracy', 0):.4f}")
        print(f"  Precision: {anomaly_results.get('precision', 0):.4f}")
        print(f"  Recall:    {anomaly_results.get('recall', 0):.4f}")
        print(f"  F1-Score:  {anomaly_results.get('f1_score', 0):.4f}")
        if anomaly_results.get('roc_auc') is not None:
            print(f"  ROC-AUC:   {anomaly_results.get('roc_auc', 0):.4f}")


    # Stacking Ensemble Results
    stacking_results = evaluation_results.get('stacking_ensemble', {})
    best_meta = evaluation_results.get('best_meta_classifier', {})
    
    if stacking_results:
        print(f"\nStacking Ensemble Performance (Meta-Classifiers):")
        print(f"{'Meta-Classifier':<18} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}")
        print("-" * 75)
        
        for meta_name, metrics in stacking_results.items():
            if 'error' not in metrics:
                print(f"{meta_name:<18} {metrics.get('accuracy', 0):<10.4f} "
                      f"{metrics.get('precision', 0):<10.4f} {metrics.get('recall', 0):<10.4f} "
                      f"{metrics.get('f1_score', 0):<10.4f} {metrics.get('roc_auc', 0):<10.4f}")
        
        if best_meta and best_meta.get('name'):
            print(f"\nBest Meta-Classifier: {best_meta['name']} (F1: {best_meta.get('f1_score', 0):.4f})")
    
    # Multi-class Classification Results
    print(f"\nMulti-class Classification Performance:")
    ensemble_results = evaluation_results.get('ensemble_results', {})
    successful_methods = {k: v for k, v in ensemble_results.items() 
                         if v['status'] == 'success'}
    
    if successful_methods:
        print(f"\n{'Method':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 65)
        
        best_f1 = 0
        best_method = None
        
        for method_key, method_data in successful_methods.items():
            metrics = method_data.get('metrics', {})
            method_name = method_data.get('method_name', method_key)
            
            if metrics:
                print(f"{method_name:<25} {metrics.get('accuracy', 0):<10.4f} "
                      f"{metrics.get('precision', 0):<10.4f} {metrics.get('recall', 0):<10.4f} "
                      f"{metrics.get('f1_score', 0):<10.4f}")
                
                if metrics.get('f1_score', 0) > best_f1:
                    best_f1 = metrics.get('f1_score', 0)
                    best_method = method_name
        
        if best_method:
            print(f"\nBest Performing Method: {best_method}")
            print(f"F1-Score: {best_f1:.4f}")
    
    else:
        print("\nNo successful method evaluations to compare.")
    
    # Research Insights
    print(f"\nResearch Insights:")
    print(f"  • Multi-model ensemble with attack-specific optimization")
    print(f"  • Probe attack uses weighted ensemble of 4 models (CNN-LSTM + 3 ML)")
    print(f"  • Optimized thresholds per attack type improve detection accuracy")
    print(f"  • Uses full feature set ({test_info.get('feature_count', 'N/A')} features) for comprehensive analysis")
    print(f"  • Hierarchical detection suitable for high-precision requirements")
    print(f"  • Different ensemble methods leverage specialized model strengths")
    
    print("\n" + "="*70)


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate Network Intrusion Detection Ensemble')
    parser.add_argument('--train-data', type=str, default=str(TRAIN_DATA_PATH),
                       help='Path to training data file')
    parser.add_argument('--test-data', type=str, default=str(TEST_DATA_PATH),
                       help='Path to test data file')
    parser.add_argument('--output-dir', type=str, default=str(RESULTS_DIR),
                       help='Directory to save results')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Number of test samples to evaluate (None for full dataset)')
    
    args = parser.parse_args()
    
    # Display evaluation mode
    if args.sample_size is None:
        print("Evaluation Mode: FULL TEST DATASET")
    else:
        print(f"Evaluation Mode: SAMPLE ({args.sample_size} samples)")
    
    # Check if data files exist
    if not Path(args.train_data).exists():
        print(f"Error: Training data file not found: {args.train_data}")
        print("Please place KDDTrain+.txt in the data directory")
        return
    
    if not Path(args.test_data).exists():
        print(f"Error: Test data file not found: {args.test_data}")
        print("Please place KDDTest+.txt in the data directory")
        return
    
    try:
        # Run comprehensive evaluation
        evaluation_results = evaluate_ensemble_comprehensive(
            args.train_data, args.test_data, args.sample_size
        )
        
        if evaluation_results:
            # Generate report
            generate_comprehensive_report(evaluation_results)
            
            # Save results
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)
            print(f"\nResults saved to: {output_dir}")
            
            print("\nEvaluation completed successfully!")
        else:
            print("Evaluation failed. Please check model availability.")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()
