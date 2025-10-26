import json
import time
from pathlib import Path
from src.configs.base_config import ResNetConfig, GPT2Config
from src.models.resnet_trainer import ResNetTrainer
from src.models.gpt2_trainer import GPT2Trainer

def run_resnet_experiments():
    """Run ResNet-50 experiments with different configurations"""
    
    # Baseline configuration (no enhancements)
    baseline_config = ResNetConfig(
        use_mixed_precision=False,
        use_gradient_accumulation=False,
        use_dynamic_batching=False,
        use_activation_checkpointing=False,
        use_deepspeed=False,
        num_epochs=30
    )
    
    # Optimized configuration (all enhancements)
    optimized_config = ResNetConfig(
        use_mixed_precision=True,
        use_gradient_accumulation=True,
        gradient_accumulation_steps=4,
        use_dynamic_batching=True,
        use_activation_checkpointing=True,
        use_deepspeed=True,
        num_epochs=30
    )
    
    print("Running ResNet-50 Baseline...")
    baseline_trainer = ResNetTrainer(baseline_config)
    baseline_results = baseline_trainer.train()
    
    print("Running ResNet-50 Optimized...")
    optimized_trainer = ResNetTrainer(optimized_config)
    optimized_results = optimized_trainer.train()
    
    # Save results
    results = {
        'baseline': baseline_results,
        'optimized': optimized_results
    }
    
    with open('results/resnet50_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def run_gpt2_experiments():
    """Run GPT-2 experiments with different configurations"""
    
    # Baseline configuration
    baseline_config = GPT2Config(
        use_mixed_precision=False,
        use_gradient_accumulation=False,
        use_dynamic_batching=False,
        use_activation_checkpointing=False,
        use_deepspeed=False,
        num_epochs=20
    )
    
    # Optimized configuration
    optimized_config = GPT2Config(
        use_mixed_precision=True,
        use_gradient_accumulation=True,
        gradient_accumulation_steps=4,
        use_dynamic_batching=True,
        use_activation_checkpointing=True,
        use_deepspeed=True,
        num_epochs=20
    )
    
    print("Running GPT-2 Baseline...")
    baseline_trainer = GPT2Trainer(baseline_config)
    baseline_results = baseline_trainer.train()
    
    print("Running GPT-2 Optimized...")
    optimized_trainer = GPT2Trainer(optimized_config)
    optimized_results = optimized_trainer.train()
    
    # Save results
    results = {
        'baseline': baseline_results,
        'optimized': optimized_results
    }
    
    with open('results/gpt2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    # Create results directory
    Path("results").mkdir(exist_ok=True)
    
    # Run experiments
    print("Starting ResNet-50 Experiments...")
    resnet_results = run_resnet_experiments()
    
    print("Starting GPT-2 Experiments...")
    gpt2_results = run_gpt2_experiments()
    
    print("All experiments completed!")