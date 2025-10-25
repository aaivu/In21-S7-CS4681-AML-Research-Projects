"""
PatchTST model wrapper for weather forecasting
"""

import torch
from model import PatchTST_backbone
import config


def create_model():
    """Create PatchTST model with config parameters"""
    model = PatchTST_backbone(
        c_in=config.ENC_IN,
        context_window=config.SEQ_LEN,
        target_window=config.PRED_LEN,
        patch_len=config.PATCH_LEN,
        stride=config.STRIDE,
        n_layers=config.E_LAYERS,
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
        fc_dropout=config.FC_DROPOUT,
        head_dropout=config.HEAD_DROPOUT,
        individual=config.INDIVIDUAL,
        revin=config.REVIN,
        affine=config.AFFINE,
        subtract_last=config.SUBTRACT_LAST,
    )
    return model


if __name__ == "__main__":
    # Test model creation
    model = create_model()
    print(f"Model created successfully")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, config.ENC_IN, config.SEQ_LEN)
    y = model(x)
    print(f"\nInput shape:  {x.shape}  # (batch, channels, seq_len)")
    print(f"Output shape: {y.shape}  # (batch, channels, pred_len)")

    expected_shape = (batch_size, config.ENC_IN, config.PRED_LEN)
    assert y.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {y.shape}"
    print("\n✓ Model test passed!")
