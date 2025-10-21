"""Tests for the data loader utility."""

import unittest
import json
import tempfile
import os
from pathlib import Path
from utils.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data = {
            "outputs": [
                "This is a test output 1",
                "This is a test output 2",
                "This is a test output 3"
            ]
        }
        
        # Create test JSON file
        self.test_file = Path(self.temp_dir) / "test_data.json"
        with open(self.test_file, 'w') as f:
            json.dump(self.test_data, f)
        
        self.data_loader = DataLoader(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test DataLoader initialization."""
        self.assertEqual(str(self.data_loader.data_dir), self.temp_dir)
        self.assertTrue(self.data_loader.data_dir.exists())
    
    def test_initialization_with_nonexistent_dir(self):
        """Test DataLoader initialization with non-existent directory."""
        with self.assertRaises(FileNotFoundError):
            DataLoader("/nonexistent/directory")
    
    def test_load_json_file(self):
        """Test loading a JSON file."""
        data = self.data_loader.load_json_file("test_data.json")
        self.assertEqual(data, self.test_data)
    
    def test_load_nonexistent_file(self):
        """Test loading a non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.data_loader.load_json_file("nonexistent.json")
    
    def test_get_outputs_from_file(self):
        """Test extracting outputs from a file."""
        outputs = self.data_loader.get_outputs_from_file("test_data.json")
        self.assertEqual(outputs, self.test_data["outputs"])
    
    def test_get_outputs_from_file_no_outputs(self):
        """Test extracting outputs from a file with no outputs."""
        # Create file without outputs
        no_outputs_file = Path(self.temp_dir) / "no_outputs.json"
        with open(no_outputs_file, 'w') as f:
            json.dump({"other_data": "value"}, f)
        
        outputs = self.data_loader.get_outputs_from_file("no_outputs.json")
        self.assertEqual(outputs, [])
    
    def test_load_all_adversarial_data(self):
        """Test loading all adversarial data."""
        # Create another test file
        test_data_2 = {"outputs": ["Output A", "Output B"]}
        test_file_2 = Path(self.temp_dir) / "test_data_2.json"
        with open(test_file_2, 'w') as f:
            json.dump(test_data_2, f)
        
        all_data = self.data_loader.load_all_adversarial_data()
        
        self.assertIn("test_data", all_data)
        self.assertIn("test_data_2", all_data)
        self.assertEqual(all_data["test_data"], self.test_data["outputs"])
        self.assertEqual(all_data["test_data_2"], test_data_2["outputs"])
    
    def test_get_sample_data(self):
        """Test getting sample data."""
        sample = self.data_loader.get_sample_data("test_data", 2)
        self.assertEqual(len(sample), 2)
        self.assertEqual(sample, self.test_data["outputs"][:2])
    
    def test_get_sample_data_larger_than_available(self):
        """Test getting sample data larger than available."""
        sample = self.data_loader.get_sample_data("test_data", 10)
        self.assertEqual(len(sample), 3)  # Only 3 outputs available
        self.assertEqual(sample, self.test_data["outputs"])
    
    def test_list_available_datasets(self):
        """Test listing available datasets."""
        datasets = self.data_loader.list_available_datasets()
        self.assertIn("test_data", datasets)
    
    def test_invalid_json_file(self):
        """Test handling invalid JSON file."""
        invalid_file = Path(self.temp_dir) / "invalid.json"
        with open(invalid_file, 'w') as f:
            f.write("invalid json content {")
        
        with self.assertRaises(json.JSONDecodeError):
            self.data_loader.load_json_file("invalid.json")


if __name__ == '__main__':
    unittest.main()
