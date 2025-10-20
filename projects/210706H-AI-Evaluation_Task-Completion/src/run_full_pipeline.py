"""
Full Pipeline Orchestrator
Runs the complete CAFE project from preprocessing to results generation
"""

import os
import sys
import time
import subprocess
from datetime import datetime


def print_header(text):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(f" {text}")
    print("="*80 + "\n")


def run_step(script_name, description, skip_on_error=False):
    """
    Run a Python script and handle errors.
    
    Args:
        script_name (str): Name of the script to run
        description (str): Description of this step
        skip_on_error (bool): If True, continue on error; otherwise stop
    
    Returns:
        bool: True if successful, False otherwise
    """
    print_header(f"STEP: {description}")
    print(f"Running: {script_name}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n✓ {description} completed successfully")
        print(f"  Time elapsed: {elapsed/60:.1f} minutes\n")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {description} failed")
        print(f"  Time elapsed: {elapsed/60:.1f} minutes")
        print(f"  Error: {e}\n")
        
        if skip_on_error:
            print("  Continuing to next step...\n")
            return False
        else:
            print("  Pipeline stopped due to error.\n")
            sys.exit(1)


def check_requirements():
    """Check if required packages are installed."""
    print_header("Checking Requirements")
    
    required_packages = [
        'torch', 'transformers', 'datasets', 'pandas', 
        'numpy', 'sklearn', 'matplotlib', 'seaborn', 'tqdm'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("Please install requirements: pip install -r requirements.txt")
        return False
    
    print("\n✓ All required packages are installed")
    return True


def check_gpu():
    """Check GPU availability."""
    print_header("Checking GPU")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ GPU detected: {gpu_name}")
            print(f"  Memory: {gpu_memory:.1f} GB")
            return True
        else:
            print("⚠ No GPU detected. Training will be very slow on CPU.")
            response = input("Continue anyway? (y/n): ")
            return response.lower() == 'y'
    except ImportError:
        print("✗ PyTorch not installed")
        return False


def main():
    """Run the complete pipeline."""
    
    print("="*80)
    print(" CAFE: Complete Pipeline Execution")
    print(" Debiased Toxicity Classification System")
    print("="*80)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    pipeline_start = time.time()
    
    # Pre-flight checks
    if not check_requirements():
        sys.exit(1)
    
    if not check_gpu():
        sys.exit(1)
    
    # Define pipeline steps
    steps = [
        {
            'script': 'preprocessing.py',
            'description': 'Data Preprocessing',
            'required': True
        },
        {
            'script': 'train.py',
            'description': 'Model Training (5-Fold CV)',
            'required': True
        },
        {
            'script': 'evaluate.py',
            'description': 'Model Evaluation',
            'required': True
        },
        {
            'script': 'run_rtp_experiment.py',
            'description': 'RealToxicityPrompts Experiment',
            'required': False  # Can skip if Perspective API not configured
        },
        {
            'script': 'generate_results.py',
            'description': 'Generate Figures and Tables',
            'required': False
        }
    ]
    
    # Confirm execution
    print("\n" + "-"*80)
    print("Pipeline Steps:")
    for i, step in enumerate(steps, 1):
        required = "Required" if step['required'] else "Optional"
        print(f"  {i}. {step['description']} - {step['script']} [{required}]")
    print("-"*80)
    
    response = input("\nProceed with full pipeline? (y/n): ")
    if response.lower() != 'y':
        print("Pipeline cancelled.")
        sys.exit(0)
    
    # Execute pipeline
    results = []
    
    for i, step in enumerate(steps, 1):
        success = run_step(
            step['script'],
            f"{i}/{len(steps)} - {step['description']}",
            skip_on_error=not step['required']
        )
        results.append({
            'step': step['description'],
            'success': success
        })
    
    # Summary
    pipeline_elapsed = time.time() - pipeline_start
    
    print_header("Pipeline Complete")
    print(f"Total time: {pipeline_elapsed/3600:.2f} hours\n")
    
    print("Summary:")
    for i, result in enumerate(results, 1):
        status = "✓ Success" if result['success'] else "✗ Failed"
        print(f"  {i}. {result['step']}: {status}")
    
    print("\n" + "="*80)
    print(" Next Steps:")
    print("="*80)
    print("1. Review evaluation_results.csv for bias metrics")
    print("2. Check figures: figure1_*.png, figure2_*.png")
    print("3. Read tables: table2_*.md, table3_*.csv")
    print("4. Use the trained models in models/ directory")
    
    # Check if all required steps succeeded
    required_success = all(
        r['success'] for r, s in zip(results, steps) if s['required']
    )
    
    if required_success:
        print("\n✓ All required steps completed successfully!")
    else:
        print("\n⚠ Some required steps failed. Check the logs above.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)