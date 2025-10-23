#!/usr/bin/env python3
"""
NODE Baseline Dataset Downloader

This script downloads all required datasets for NODE baseline experiments:
- EPSILON (classification)
- YEAR (regression) 
- HIGGS (classification)
- A9A (classification)

Usage: python download_datasets.py
"""

import os
import sys
import time
import requests
import bz2
import gzip
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.datasets import load_svmlight_file

def download_file(url, filename, description="file"):
    """Download a file with progress bar and retry logic"""
    print(f"Downloading {description}: {url}")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Remove incomplete file if it exists
            if os.path.exists(filename):
                os.remove(filename)
            
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_length = int(response.headers.get('content-length', 0))
            
            with open(filename, 'wb') as f:
                if total_length > 0:
                    with tqdm(total=total_length, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                        pbar.set_description(f"Downloading {description}")
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            
            print(f"✓ Successfully downloaded: {filename}")
            return True
            
        except Exception as e:
            print(f"✗ Download attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"✗ Failed to download {url} after {max_retries} attempts")
                return False
    
    return False

def extract_bz2(archive_path, output_path):
    """Extract a bz2 file"""
    print(f"Extracting {archive_path} -> {output_path}")
    
    try:
        with bz2.BZ2File(archive_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"✓ Extracted: {output_path}")
        return True
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False

def extract_gz(archive_path, output_path):
    """Extract a gz file"""
    print(f"Extracting {archive_path} -> {output_path}")
    
    try:
        with gzip.open(archive_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"✓ Extracted: {output_path}")
        return True
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False

def download_epsilon():
    """Download EPSILON dataset"""
    print("\n" + "="*50)
    print("Downloading EPSILON Dataset")
    print("="*50)
    
    data_path = './data/EPSILON'
    os.makedirs(data_path, exist_ok=True)
    
    # Check if already downloaded and extracted
    train_path = os.path.join(data_path, 'epsilon_normalized')
    test_path = os.path.join(data_path, 'epsilon_normalized.t')
    
    if os.path.exists(train_path) and os.path.exists(test_path) and os.path.getsize(train_path) > 0:
        print("✓ EPSILON dataset already exists")
        return True
    
    # Download compressed files
    urls = [
        "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2",
        "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.t.bz2"
    ]
    
    archives = [
        os.path.join(data_path, 'epsilon_normalized.bz2'),
        os.path.join(data_path, 'epsilon_normalized.t.bz2')
    ]
    
    # Download archives
    for url, archive_path in zip(urls, archives):
        if not os.path.exists(archive_path) or os.path.getsize(archive_path) == 0:
            if not download_file(url, archive_path, "EPSILON archive"):
                return False
    
    # Extract archives
    outputs = [train_path, test_path]
    for archive_path, output_path in zip(archives, outputs):
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            if not extract_bz2(archive_path, output_path):
                return False
    
    # Download index files
    index_urls = [
        "https://www.dropbox.com/s/wxgm94gvm6d3xn5/stratified_train_idx.txt?dl=1",
        "https://www.dropbox.com/s/fm4llo5uucdglti/stratified_valid_idx.txt?dl=1"
    ]
    
    index_files = [
        os.path.join(data_path, 'stratified_train_idx.txt'),
        os.path.join(data_path, 'stratified_valid_idx.txt')
    ]
    
    for url, filename in zip(index_urls, index_files):
        if not os.path.exists(filename):
            download_file(url, filename, "EPSILON index file")
    
    print("✓ EPSILON dataset ready!")
    return True

def download_year():
    """Download YEAR dataset"""
    print("\n" + "="*50)
    print("Downloading YEAR Dataset")
    print("="*50)
    
    data_path = './data/YEAR'
    os.makedirs(data_path, exist_ok=True)
    
    # Check if already downloaded
    data_file = os.path.join(data_path, 'data.csv')
    if os.path.exists(data_file):
        print("✓ YEAR dataset already exists")
        return True
    
    # Download main data file
    url = "https://www.dropbox.com/s/l09pug0ywaqsy0e/YearPredictionMSD.txt?dl=1"
    if not download_file(url, data_file, "YEAR dataset"):
        return False
    
    # Download index files
    index_urls = [
        "https://www.dropbox.com/s/00u6cnj9mthvzj1/stratified_train_idx.txt?dl=1",
        "https://www.dropbox.com/s/420uhjvjab1bt7k/stratified_valid_idx.txt?dl=1"
    ]
    
    index_files = [
        os.path.join(data_path, 'stratified_train_idx.txt'),
        os.path.join(data_path, 'stratified_valid_idx.txt')
    ]
    
    for url, filename in zip(index_urls, index_files):
        if not os.path.exists(filename):
            download_file(url, filename, "YEAR index file")
    
    print("✓ YEAR dataset ready!")
    return True

def download_higgs():
    """Download HIGGS dataset"""
    print("\n" + "="*50)
    print("Downloading HIGGS Dataset")
    print("="*50)
    
    data_path = './data/HIGGS'
    os.makedirs(data_path, exist_ok=True)
    
    # Check if already downloaded
    data_file = os.path.join(data_path, 'higgs.csv')
    if os.path.exists(data_file):
        print("✓ HIGGS dataset already exists")
        return True
    
    # Download compressed file
    archive_path = os.path.join(data_path, 'HIGGS.csv.gz')
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
    
    if not download_file(url, archive_path, "HIGGS dataset"):
        return False
    
    # Extract
    if not extract_gz(archive_path, data_file):
        return False
    
    # Download index files
    index_urls = [
        "https://www.dropbox.com/s/i2uekmwqnp9r4ix/stratified_train_idx.txt?dl=1",
        "https://www.dropbox.com/s/wkbk74orytmb2su/stratified_valid_idx.txt?dl=1"
    ]
    
    index_files = [
        os.path.join(data_path, 'stratified_train_idx.txt'),
        os.path.join(data_path, 'stratified_valid_idx.txt')
    ]
    
    for url, filename in zip(index_urls, index_files):
        if not os.path.exists(filename):
            download_file(url, filename, "HIGGS index file")
    
    print("✓ HIGGS dataset ready!")
    return True

def download_a9a():
    """Download A9A dataset"""
    print("\n" + "="*50)
    print("Downloading A9A Dataset")
    print("="*50)
    
    data_path = './data/A9A'
    os.makedirs(data_path, exist_ok=True)
    
    # Check if already downloaded
    train_path = os.path.join(data_path, 'a9a')
    test_path = os.path.join(data_path, 'a9a.t')
    
    if os.path.exists(train_path) and os.path.exists(test_path):
        print("✓ A9A dataset already exists")
        return True
    
    # Download files
    urls = [
        "https://www.dropbox.com/s/9cqdx166iwonrj9/a9a?dl=1",
        "https://www.dropbox.com/s/sa0ds895c0v4xc6/a9a.t?dl=1"
    ]
    
    files = [train_path, test_path]
    
    for url, filename in zip(urls, files):
        if not os.path.exists(filename):
            if not download_file(url, filename, "A9A dataset"):
                return False
    
    # Download index files
    index_urls = [
        "https://www.dropbox.com/s/xy4wwvutwikmtha/stratified_train_idx.txt?dl=1",
        "https://www.dropbox.com/s/nthpxofymrais5s/stratified_test_idx.txt?dl=1"
    ]
    
    index_files = [
        os.path.join(data_path, 'stratified_train_idx.txt'),
        os.path.join(data_path, 'stratified_valid_idx.txt')
    ]
    
    for url, filename in zip(index_urls, index_files):
        if not os.path.exists(filename):
            download_file(url, filename, "A9A index file")
    
    print("✓ A9A dataset ready!")
    return True

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
    """Main function to download all datasets"""
    print("NODE Baseline Dataset Downloader")
    print("="*60)
    print("This will download all required datasets for NODE baseline experiments:")
    print("- EPSILON (classification, ~3GB)")
    print("- YEAR (regression, ~500MB)")
    print("- HIGGS (classification, ~2GB)")
    print("- A9A (classification, ~10MB)")
    print("="*60)
    
    # Create data directory
    os.makedirs('./data', exist_ok=True)
    
    # Download datasets
    datasets = [
        ("EPSILON", download_epsilon),
        ("YEAR", download_year),
        ("HIGGS", download_higgs),
        ("A9A", download_a9a)
    ]
    
    success = True
    for name, download_func in datasets:
        try:
            if not download_func():
                print(f"✗ Failed to download {name}")
                success = False
        except KeyboardInterrupt:
            print(f"\n✗ Download interrupted by user")
            return False
        except Exception as e:
            print(f"✗ Error downloading {name}: {e}")
            success = False
    
    if success:
        print("\n" + "="*60)
        print("All datasets downloaded successfully!")
        print("="*60)
        
        # Test loading
        if test_dataset_loading():
            print("\n🎉 Ready to run baseline experiments!")
            print("\nYou can now run:")
            print("  python quick_baseline.py epsilon shallow")
            print("  python quick_baseline.py year deep")
            print("  python baseline_evaluation.py --all")
        else:
            print("\n⚠️  Datasets downloaded but loading test failed")
    else:
        print("\n✗ Some datasets failed to download")
    
    return success

if __name__ == '__main__':
    main()
