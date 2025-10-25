"""
Main execution script for Network Intrusion Detection System
Ensemble evaluation pipeline
"""

import sys
import argparse
from pathlib import Path

# Add current directory to Python path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from evaluate_ensemble import evaluate_ensemble_comprehensive
from load_models import verify_model_compatibility
from config import TRAIN_DATA_PATH, TEST_DATA_PATH


def main():
    """Main execution pipeline for ensemble evaluation"""
    parser = argparse.ArgumentParser(
        description='Network Intrusion Detection - Multi-Model Ensemble Evaluation'
    )
    parser.add_argument('--train-data', type=str, default=str(TRAIN_DATA_PATH),
                       help='Path to training data file')
    parser.add_argument('--test-data', type=str, default=str(TEST_DATA_PATH),
                       help='Path to test data file')
    parser.add_argument('--test-dataset', type=str, choices=['KDDTest+', 'KDDTest-21'], default='KDDTest+',
                       help='Choose test dataset: KDDTest+ (default) or KDDTest-21 (generalizability)')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Number of test samples for evaluation (None for full dataset)')
    
    args = parser.parse_args()
    
    # Set test data path based on chosen dataset
    from config import TEST_DATASETS
    if args.test_dataset in TEST_DATASETS:
        args.test_data = str(TEST_DATASETS[args.test_dataset])
    
    # Clean, professional output - removed verbose headers and emojis
    
    # Check model availability
    if not verify_model_compatibility():
        print("Error: Required models not found!")
        print("Please train models using the individual notebooks first:")
        print("  - experiments/ndd_DoS_.ipynb")
        print("  - experiments/ndd_probe_.ipynb") 
        print("  - experiments/ndd_R2L_.ipynb")
        print("  - experiments/ndd_U2R_.ipynb")
        return
    
    # Check data availability
    if not Path(args.train_data).exists():
        print(f"Error: Training data not found: {args.train_data}")
        print("Please place KDDTrain+.txt in the data/ directory")
        return
    
    if not Path(args.test_data).exists():
        print(f"Error: Test data not found: {args.test_data}")
        print("Please place test data files in the data/ directory")
        return
    
    # Run ensemble evaluation
    try:
        evaluation_results = evaluate_ensemble_comprehensive(
            args.train_data, args.test_data, args.sample_size
        )
        
        if not evaluation_results:
            print("Error: Ensemble evaluation failed!")
            return
        
        # Save results to main results directory
        save_evaluation_results(evaluation_results, args.test_dataset, args.sample_size)
            
    except Exception as e:
        print(f"Error: Evaluation failed: {e}")
        raise


def save_evaluation_results(results, dataset_name, sample_size):
    """Save evaluation results to the main results directory"""
    import json
    from datetime import datetime
    
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_suffix = f"_sample_{sample_size}" if sample_size else "_full_dataset"
    
    # Save detailed JSON results
    results_file = results_dir / f"single_evaluation_{dataset_name.lower().replace('+', 'plus').replace('-', '_')}{sample_suffix}_{timestamp}.json"
    
    evaluation_summary = {
        'dataset': dataset_name,
        'sample_size': sample_size or 'full_dataset',
        'evaluation_timestamp': timestamp,
        'results': results
    }
    
    with open(results_file, 'w') as f:
        json.dump(evaluation_summary, f, indent=2, default=str)
    
    # Save summary text report
    report_file = results_dir / f"single_evaluation_report_{dataset_name.lower().replace('+', 'plus').replace('-', '_')}{sample_suffix}_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ENSEMBLE-BASED NETWORK INTRUSION DETECTION SYSTEM\n")
        f.write("SINGLE DATASET EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Student ID: 210144G\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Sample Size: {sample_size or 'Full dataset'}\n\n")
        
        # Stacking Ensemble Results
        stacking_data = results.get('stacking_ensemble', {})
        if stacking_data:
            f.write("STACKING ENSEMBLE RESULTS\n")
            f.write("-" * 40 + "\n")
            
            best_meta = results.get('best_meta_classifier', {})
            if best_meta and best_meta.get('name'):
                meta_name = best_meta['name']
                if meta_name in stacking_data:
                    meta_metrics = stacking_data[meta_name]
                    f.write(f"Meta-classifier: {meta_name}\n")
                    f.write(f"Accuracy: {meta_metrics.get('accuracy', 0):.4f}\n")
                    f.write(f"Precision: {meta_metrics.get('precision', 0):.4f}\n")
                    f.write(f"Recall: {meta_metrics.get('recall', 0):.4f}\n")
                    f.write(f"F1-Score: {meta_metrics.get('f1_score', 0):.4f}\n")
                    f.write(f"ROC-AUC: {meta_metrics.get('roc_auc', 0):.4f}\n\n")
                    
                    classification_report = meta_metrics.get('classification_report', 'Not available')
                    f.write("Classification Report:\n")
                    f.write(f"{classification_report}\n")
    
    print(f"Results saved to: {results_file}")
    print(f"Report saved to: {report_file}")


if __name__ == "__main__":
    main()
