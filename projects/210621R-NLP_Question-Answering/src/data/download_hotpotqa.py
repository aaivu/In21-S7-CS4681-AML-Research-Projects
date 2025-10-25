#!/usr/bin/env python3
"""
Script to download HotpotQA dataset files.
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm

# Dataset URLs
URLS = {
    'train': 'http://qa.cs.washington.edu/data/hotpotqa/hotpot_train_v1.1.json',
    'dev': 'http://qa.cs.washington.edu/data/hotpotqa/hotpot_dev_distractor_v1.json',
}

def download_file(url: str, output_path: Path) -> None:
    """
    Download a file with progress bar
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as file, tqdm(
        desc=output_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def main():
    # Set up paths
    current_dir = Path(__file__).resolve().parent
    data_dir = current_dir.parent.parent / 'data' / 'hotpotqa'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download files
    print("Downloading HotpotQA dataset files...")
    for split, url in URLS.items():
        output_path = data_dir / f'{split}.json'
        if not output_path.exists():
            print(f"\nDownloading {split} split...")
            download_file(url, output_path)
        else:
            print(f"\n{split} split already exists, skipping...")
    
    print("\nDownload complete!")

if __name__ == '__main__':
    main()