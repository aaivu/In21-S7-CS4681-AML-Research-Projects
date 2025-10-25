"""Data loading utilities for adversarial text data."""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads and manages adversarial text data from JSON files."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize DataLoader with data directory path."""
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {data_dir} not found")
    
    def load_json_file(self, filename: str) -> Dict[str, Any]:
        """Load a single JSON file and return its contents."""
        file_path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"File {filename} not found in {self.data_dir}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Successfully loaded {filename}")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {filename}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            raise
    
    def get_outputs_from_file(self, filename: str) -> List[str]:
        """Extract outputs array from a JSON file."""
        data = self.load_json_file(filename)
        outputs = data.get('outputs', [])
        if not outputs:
            logger.warning(f"No outputs found in {filename}")
        return outputs
    
    def load_all_adversarial_data(self) -> Dict[str, List[str]]:
        """Load all adversarial data from JSON files in the data directory."""
        adversarial_data = {}
        
        # Look for JSON files in the data directory
        json_files = list(self.data_dir.glob("*.json"))
        
        if not json_files:
            logger.warning(f"No JSON files found in {self.data_dir}")
            return adversarial_data
        
        for json_file in json_files:
            try:
                outputs = self.get_outputs_from_file(json_file.name)
                adversarial_data[json_file.stem] = outputs
                logger.info(f"Loaded {len(outputs)} outputs from {json_file.name}")
            except Exception as e:
                logger.error(f"Failed to load {json_file.name}: {e}")
                continue
        
        return adversarial_data
    
    def get_sample_data(self, dataset_name: str, sample_size: int = 10) -> List[str]:
        """Get a sample of adversarial data from a specific dataset."""
        outputs = self.get_outputs_from_file(f"{dataset_name}.json")
        return outputs[:sample_size] if outputs else []
    
    def list_available_datasets(self) -> List[str]:
        """List all available datasets in the data directory."""
        json_files = list(self.data_dir.glob("*.json"))
        return [f.stem for f in json_files]
