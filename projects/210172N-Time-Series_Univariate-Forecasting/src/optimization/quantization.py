"""Model quantization for deployment optimization."""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnxruntime not available. Quantization features disabled.")


def quantize_onnx_model(
    onnx_model_path: str,
    output_path: str,
    weight_type: str = 'int8'
) -> Path:
    """
    Apply post-training dynamic quantization to ONNX model.

    Args:
        onnx_model_path: Path to FP32 ONNX model
        output_path: Path to save quantized model
        weight_type: Weight quantization type ('int8' or 'uint8')

    Returns:
        Path to quantized model

    Examples:
        >>> quantize_onnx_model(
        ...     'model_fp32.onnx',
        ...     'model_int8.onnx',
        ...     weight_type='int8'
        ... )
    """
    if not ONNX_AVAILABLE:
        raise RuntimeError("onnxruntime not installed. Install with: pip install onnxruntime")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Map weight type
    weight_type_map = {
        'int8': QuantType.QInt8,
        'uint8': QuantType.QUInt8
    }

    # Quantize
    quantize_dynamic(
        model_input=onnx_model_path,
        model_output=str(output_path),
        weight_type=weight_type_map[weight_type.lower()]
    )

    # Get model size and compression ratio
    original_size = Path(onnx_model_path).stat().st_size / (1024 * 1024)
    quantized_size = output_path.stat().st_size / (1024 * 1024)
    compression = original_size / quantized_size

    print(f"✓ Model quantized: {output_path}")
    print(f"   Original: {original_size:.2f} MB → Quantized: {quantized_size:.2f} MB")
    print(f"   Compression: {compression:.2f}x")

    return output_path


def quantize_model(
    model: nn.Module,
    calibration_data: Optional[torch.utils.data.DataLoader] = None
) -> nn.Module:
    """
    Apply PyTorch dynamic quantization to model.

    Args:
        model: PyTorch model
        calibration_data: Optional calibration data for static quantization

    Returns:
        Quantized model

    Examples:
        >>> quantized_model = quantize_model(model)
    """
    model.eval()

    # Dynamic quantization (no calibration needed)
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv1d},
        dtype=torch.qint8
    )

    print("✓ Model quantized using PyTorch dynamic quantization")

    return quantized_model
