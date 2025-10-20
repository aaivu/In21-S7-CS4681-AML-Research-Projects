"""ONNX export functionality for model deployment."""

import torch
import torch.nn as nn
import onnx
from pathlib import Path
from typing import Tuple, Optional


class ONNXExporter:
    """
    Export PyTorch models to ONNX format.

    Args:
        model: PyTorch model to export
        input_shape: Input tensor shape (batch_size, seq_len, features)
        opset_version: ONNX opset version

    Examples:
        >>> exporter = ONNXExporter(model, input_shape=(1, 336, 7))
        >>> exporter.export('model.onnx')
    """

    def __init__(
        self,
        model: nn.Module,
        input_shape: Tuple[int, int, int],
        opset_version: int = 14
    ):
        self.model = model
        self.input_shape = input_shape
        self.opset_version = opset_version

    def export(
        self,
        output_path: str,
        verify: bool = True,
        optimize: bool = True
    ) -> Path:
        """
        Export model to ONNX format.

        Args:
            output_path: Path to save ONNX model
            verify: Whether to verify the exported model
            optimize: Whether to apply ONNX optimizations

        Returns:
            Path to exported model
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.model.eval()
        dummy_input = torch.randn(*self.input_shape)

        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=self.opset_version,
            do_constant_folding=optimize,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        # Verify exported model
        if verify:
            try:
                onnx_model = onnx.load(str(output_path))
                onnx.checker.check_model(onnx_model)
                print(f"✓ ONNX model verified: {output_path}")
            except Exception as e:
                print(f"⚠ ONNX verification failed: {e}")

        # Get model size
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✓ Model exported: {output_path} ({size_mb:.2f} MB)")

        return output_path


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, int, int],
    opset_version: int = 14,
    verify: bool = True
) -> Path:
    """
    Quick export function for ONNX conversion.

    Args:
        model: PyTorch model
        output_path: Output path for ONNX file
        input_shape: Input tensor shape
        opset_version: ONNX opset version
        verify: Whether to verify exported model

    Returns:
        Path to exported model

    Examples:
        >>> export_to_onnx(
        ...     model,
        ...     'model_fp32.onnx',
        ...     input_shape=(1, 336, 7)
        ... )
    """
    exporter = ONNXExporter(model, input_shape, opset_version)
    return exporter.export(output_path, verify=verify)
