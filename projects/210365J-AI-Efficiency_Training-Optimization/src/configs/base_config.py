from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    # Basic training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 50
    warmup_epochs: int = 5
    
    # Enhancement methodologies
    use_mixed_precision: bool = True
    use_gradient_accumulation: bool = True
    gradient_accumulation_steps: int = 4
    use_dynamic_batching: bool = True
    use_activation_checkpointing: bool = True
    
    # Optimization
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    
    # DeepSpeed configuration
    use_deepspeed: bool = True
    deepspeed_config: Optional[dict] = None

@dataclass
class ResNetConfig(TrainingConfig):
    model_name: str = "resnet50"
    num_classes: int = 10
    input_size: int = 224

@dataclass
class GPT2Config(TrainingConfig):
    model_name: str = "gpt2"
    vocab_size: int = 50257
    max_length: int = 512
    model_type: str = "small"  # small, medium