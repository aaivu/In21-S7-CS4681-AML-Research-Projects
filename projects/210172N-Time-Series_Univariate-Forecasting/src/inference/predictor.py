"""Inference module for PatchTST models."""

import torch
import torch.nn as nn
import numpy as np
from typing import Union, Optional
from pathlib import Path

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class Predictor:
    """
    Unified predictor for PyTorch, ONNX FP32, and ONNX INT8 models.

    Args:
        model_path: Path to model (PyTorch .pth or ONNX .onnx)
        model_type: Type of model ('pytorch', 'onnx', or 'onnx_int8')
        device: Device for PyTorch models

    Examples:
        >>> predictor = Predictor('model.pth', model_type='pytorch')
        >>> predictions = predictor.predict(input_data)
    """

    def __init__(
        self,
        model_path: str,
        model_type: str = 'pytorch',
        device: str = 'cpu',
        pytorch_model: Optional[nn.Module] = None
    ):
        self.model_path = Path(model_path)
        self.model_type = model_type.lower()
        self.device = torch.device(device)

        if self.model_type == 'pytorch':
            if pytorch_model is None:
                raise ValueError("pytorch_model must be provided for model_type='pytorch'")
            self.model = pytorch_model
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.to(self.device)
            self.model.eval()

        elif self.model_type in ['onnx', 'onnx_int8']:
            if not ONNX_AVAILABLE:
                raise RuntimeError("onnxruntime not installed. Install with: pip install onnxruntime")

            # Setup ONNX Runtime
            providers = self._get_onnx_providers()
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.session = ort.InferenceSession(
                str(model_path),
                sess_options,
                providers=providers
            )
            self.input_name = self.session.get_inputs()[0].name

            # Warmup
            self._warmup_onnx()

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def _get_onnx_providers(self):
        """Get ONNX Runtime providers."""
        available_providers = ort.get_available_providers()

        if self.model_type == 'onnx_int8':
            # INT8 models run best on CPU
            return ['CPUExecutionProvider']
        elif 'CUDAExecutionProvider' in available_providers and self.device.type == 'cuda':
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            return ['CPUExecutionProvider']

    def _warmup_onnx(self, num_iterations: int = 10):
        """Warmup ONNX model for stable inference times."""
        dummy_input = np.random.randn(1, 336, 1).astype(np.float32)
        for _ in range(num_iterations):
            _ = self.session.run(None, {self.input_name: dummy_input})

    def predict(
        self,
        x: Union[torch.Tensor, np.ndarray],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Make predictions on input data.

        Args:
            x: Input data (batch, seq_len, features)
            batch_size: Batch size for processing (if None, process all at once)

        Returns:
            Predictions array
        """
        if self.model_type == 'pytorch':
            return self._predict_pytorch(x, batch_size)
        else:
            return self._predict_onnx(x, batch_size)

    def _predict_pytorch(self, x: Union[torch.Tensor, np.ndarray], batch_size: Optional[int]) -> np.ndarray:
        """PyTorch model prediction."""
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)

        x = x.to(self.device)

        if batch_size is None:
            with torch.no_grad():
                output = self.model(x)
            return output.cpu().numpy()
        else:
            # Batch processing
            outputs = []
            for i in range(0, len(x), batch_size):
                batch = x[i:i+batch_size]
                with torch.no_grad():
                    output = self.model(batch)
                outputs.append(output.cpu().numpy())
            return np.concatenate(outputs, axis=0)

    def _predict_onnx(self, x: Union[torch.Tensor, np.ndarray], batch_size: Optional[int]) -> np.ndarray:
        """ONNX model prediction."""
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()

        x = x.astype(np.float32)

        if batch_size is None:
            output = self.session.run(None, {self.input_name: x})[0]
            return output
        else:
            # Batch processing
            outputs = []
            for i in range(0, len(x), batch_size):
                batch = x[i:i+batch_size]
                output = self.session.run(None, {self.input_name: batch})[0]
                outputs.append(output)
            return np.concatenate(outputs, axis=0)
