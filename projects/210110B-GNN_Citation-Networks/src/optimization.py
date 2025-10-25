import paddle
import paddle.optimizer as optim
from paddle.optimizer.lr import LinearWarmup, CosineAnnealingDecay


def get_optimizer(parameters, 
                  learning_rate=0.001,
                  max_steps=100,
                  weight_decay=0.0001,
                  warmup_proportion=0.1,
                  clip=-1,
                  use_lr_decay=True):
    """Get optimizer and learning rate scheduler"""
    
    # Learning rate scheduler
    lr_scheduler = None
    if use_lr_decay and warmup_proportion > 0:
        warmup_steps = int(max_steps * warmup_proportion)
        cosine_decay = CosineAnnealingDecay(
            learning_rate=learning_rate,
            T_max=max_steps - warmup_steps,
            eta_min=learning_rate * 0.01
        )
        lr_scheduler = LinearWarmup(
            learning_rate=cosine_decay,
            warmup_steps=warmup_steps,
            start_lr=0.0,
            end_lr=learning_rate
        )
        lr = lr_scheduler
    else:
        lr = learning_rate
    
    # Optimizer
    if clip > 0:
        clip_norm = paddle.nn.ClipGradByNorm(clip_norm=clip)
    else:
        clip_norm = None
        
    optimizer = optim.AdamW(
        learning_rate=lr,
        parameters=parameters,
        weight_decay=weight_decay,
        grad_clip=clip_norm
    )
    
    return optimizer, lr_scheduler
