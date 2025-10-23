#!/usr/bin/env python3
"""
Dataset Setup Script

This script extracts and prepares all downloaded datasets for NODE baseline experiments.
"""

import os
import bz2
import gzip
import shutil
import sys

def extract_bz2_file(archive_path, output_path):
    """Extract a bz2 file"""
    print(f"Extracting {os.path.basename(archive_path)}...")
    
    try:
        with bz2.BZ2File(archive_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"✓ Extracted: {os.path.basename(output_path)}")
        return True
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False

def extract_gz_file(archive_path, output_path):
    """Extract a gz file"""
    print(f"Extracting {os.path.basename(archive_path)}...")
    
    try:
        with gzip.open(archive_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"✓ Extracted: {os.path.basename(output_path)}")
        return True
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False

def setup_epsilon():
    """Setup EPSILON dataset"""
    print("\n" + "="*50)
    print("Setting up EPSILON Dataset")
    print("="*50)
    
    data_path = './data/EPSILON'
    
    # Extract compressed files
    archives = [
        os.path.join(data_path, 'epsilon_normalized.bz2'),
        os.path.join(data_path, 'epsilon_normalized.t.bz2')
    ]
    
    outputs = [
        os.path.join(data_path, 'epsilon_normalized'),
        os.path.join(data_path, 'epsilon_normalized.t')
    ]
    
    success = True
    for archive, output in zip(archives, outputs):
        if not os.path.exists(output) or os.path.getsize(output) == 0:
            if not extract_bz2_file(archive, output):
                success = False
    
    if success:
        print("✓ EPSILON dataset ready!")
    
    return success

def setup_year():
    """Setup YEAR dataset"""
    print("\n" + "="*50)
    print("Setting up YEAR Dataset")
    print("="*50)
    
    data_path = './data/YEAR'
    
    # Rename the main file
    old_file = os.path.join(data_path, 'YearPredictionMSD.txt')
    new_file = os.path.join(data_path, 'data.csv')
    
    if os.path.exists(old_file) and not os.path.exists(new_file):
        print("Renaming YearPredictionMSD.txt to data.csv...")
        os.rename(old_file, new_file)
        print("✓ YEAR dataset ready!")
        return True
    elif os.path.exists(new_file):
        print("✓ YEAR dataset already ready!")
        return True
    else:
        print("✗ YEAR dataset file not found!")
        return False

def setup_higgs():
    """Setup HIGGS dataset"""
    print("\n" + "="*50)
    print("Setting up HIGGS Dataset")
    print("="*50)
    
    data_path = './data/HIGGS'
    
    archive_path = os.path.join(data_path, 'HIGGS.csv.gz')
    output_path = os.path.join(data_path, 'higgs.csv')
    
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        if extract_gz_file(archive_path, output_path):
            print("✓ HIGGS dataset ready!")
            return True
        else:
            return False
    else:
        print("✓ HIGGS dataset already ready!")
        return True

def setup_a9a():
    """Setup A9A dataset"""
    print("\n" + "="*50)
    print("Setting up A9A Dataset")
    print("="*50)
    
    data_path = './data/A9A'
    
    # Rename the test index file
    old_file = os.path.join(data_path, 'stratified_test_idx.txt')
    new_file = os.path.join(data_path, 'stratified_valid_idx.txt')
    
    if os.path.exists(old_file) and not os.path.exists(new_file):
        print("Renaming stratified_test_idx.txt to stratified_valid_idx.txt...")
        os.rename(old_file, new_file)
        print("✓ A9A dataset ready!")
        return True
    elif os.path.exists(new_file):
        print("✓ A9A dataset already ready!")
        return True
    else:
        print("✗ A9A dataset files not found!")
        return False

def test_dataset_loading():
    """Test if datasets can be loaded"""
    print("\n" + "="*50)
    print("Testing Dataset Loading")
    print("="*50)
    
    # Add lib to path
    sys.path.insert(0, '.')
    
    try:
        import lib
        
        datasets_to_test = ['EPSILON', 'YEAR', 'HIGGS', 'A9A']
        
        for dataset_name in datasets_to_test:
            try:
                print(f"\nTesting {dataset_name}...")
                data = lib.Dataset(dataset_name, random_state=1337, quantile_transform=True, quantile_noise=1e-3)
                
                print(f"✓ {dataset_name} loaded successfully!")
                print(f"  Features: {data.X_train.shape[1]}")
                print(f"  Train: {len(data.X_train)}")
                print(f"  Valid: {len(data.X_valid)}")
                print(f"  Test: {len(data.X_test)}")
                
            except Exception as e:
                print(f"✗ {dataset_name} failed to load: {e}")
                return False
        
        print("\n✓ All datasets loaded successfully!")
        return True
        
    except ImportError:
        print("✗ Could not import lib module. Make sure you're in the correct directory.")
        return False

def main():
    """Main function"""
    print("Dataset Setup Script")
    print("="*60)
    print("This will extract and prepare all downloaded datasets")
    print("="*60)
    
    # Setup datasets
    datasets = [
        ("EPSILON", setup_epsilon),
        ("YEAR", setup_year),
        ("HIGGS", setup_higgs),
        ("A9A", setup_a9a)
    ]
    
    success = True
    for name, setup_func in datasets:
        try:
            if not setup_func():
                print(f"✗ Failed to setup {name}")
                success = False
        except Exception as e:
            print(f"✗ Error setting up {name}: {e}")
            success = False
    
    if success:
        print("\n" + "="*60)
        print("All datasets setup successfully!")
        print("="*60)
        
        # Test loading
        if test_dataset_loading():
            print("\n🎉 Ready to run baseline experiments!")
            print("\nYou can now run:")
            print("  python quick_baseline.py epsilon shallow")
            print("  python quick_baseline.py year deep")
            print("  python baseline_evaluation.py --all")
            print("  jupyter notebook notebooks/baseline_evaluation.ipynb")
        else:
            print("\n⚠️  Datasets setup but loading test failed")
    else:
        print("\n✗ Some datasets failed to setup")
    
    return success

if __name__ == '__main__':
    main()
