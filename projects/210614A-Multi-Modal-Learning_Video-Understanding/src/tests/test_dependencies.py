"""
Simple dependency verification test
Tests that all required packages can be imported and basic functionality works.
"""

import pytest
import torch
import numpy as np


class TestDependencyVerification:
    """Test that all required dependencies are installed and working."""
    
    def test_torch_import_and_basic_operations(self):
        """Test PyTorch import and basic tensor operations."""
        # Test tensor creation
        x = torch.randn(2, 3, 224, 224)
        assert x.shape == (2, 3, 224, 224)
        
        # Test basic operations
        y = torch.zeros_like(x)
        assert y.shape == x.shape
        assert torch.allclose(y, torch.zeros(2, 3, 224, 224))
        
        # Test gradient computation
        x.requires_grad_(True)
        loss = x.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
    
    def test_transformers_import(self):
        """Test transformers import."""
        try:
            import transformers
            assert hasattr(transformers, '__version__')
            print(f"Transformers version: {transformers.__version__}")
        except ImportError:
            pytest.fail("transformers package not available")
    
    def test_pytest_functionality(self):
        """Test pytest basic functionality."""
        # Test assertions
        assert 1 + 1 == 2
        assert "hello" in "hello world"
        
        # Test numpy integration
        arr = np.array([1, 2, 3])
        assert arr.shape == (3,)
        assert np.sum(arr) == 6
    
    def test_additional_packages(self):
        """Test additional required packages."""
        packages_to_test = [
            'pandas',
            'numpy', 
            'tqdm',
            'matplotlib',
            'cv2',  # opencv-python
            'einops',
            'timm'
        ]
        
        for package in packages_to_test:
            try:
                if package == 'cv2':
                    import cv2
                else:
                    __import__(package)
                print(f"✓ {package} imported successfully")
            except ImportError as e:
                pytest.fail(f"Failed to import {package}: {e}")
    
    def test_torch_neural_network(self):
        """Test PyTorch neural network functionality."""
        import torch.nn as nn
        import torch.nn.functional as F
        
        # Create a simple neural network
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 5)
                self.fc2 = nn.Linear(5, 1)
            
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        # Test network creation and forward pass
        net = SimpleNet()
        x = torch.randn(4, 10)
        output = net(x)
        
        assert output.shape == (4, 1)
        
        # Test parameter access
        params = list(net.parameters())
        assert len(params) == 4  # 2 weight matrices + 2 bias vectors
    
    def test_device_compatibility(self):
        """Test device compatibility (CPU/CUDA)."""
        # Test CPU operations
        x_cpu = torch.randn(2, 3)
        assert x_cpu.device.type == 'cpu'
        
        # Test CUDA if available
        if torch.cuda.is_available():
            x_gpu = x_cpu.cuda()
            assert x_gpu.device.type == 'cuda'
            
            # Test operations on GPU
            y_gpu = x_gpu * 2
            assert y_gpu.device.type == 'cuda'
            
            # Test data transfer back to CPU
            y_cpu = y_gpu.cpu()
            assert y_cpu.device.type == 'cpu'
            print("✓ CUDA operations successful")
        else:
            print("⚠ CUDA not available, using CPU only")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])