"""
Demonstration of cosine temperature scheduling for EgoNCE loss.

This script shows how to use the new TemperatureScheduler and EgoNCEWithScheduler
classes for dynamic temperature scheduling in contrastive learning.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Import the enhanced loss functions
from model.loss import TemperatureScheduler, EgoNCE, EgoNCEWithScheduler


def demo_temperature_scheduler():
    """
    Demonstrate the TemperatureScheduler functionality.
    """
    print("Temperature Scheduler Demo")
    print("=" * 40)
    
    # Create temperature scheduler
    scheduler = TemperatureScheduler(tau_max=0.07, tau_min=0.03, total_epochs=10)
    
    print(f"Configuration:")
    print(f"  tau_max: {scheduler.tau_max}")
    print(f"  tau_min: {scheduler.tau_min}")
    print(f"  total_epochs: {scheduler.total_epochs}")
    
    # Show temperature schedule
    print(f"\nTemperature Schedule:")
    print(f"{'Epoch':<6} {'Temperature':<12} {'Progress':<10}")
    print("-" * 30)
    
    epochs_to_show = [0, 1, 2, 3, 5, 7, 9, 10]
    for epoch in epochs_to_show:
        temp = scheduler.get_temperature(epoch)
        progress = min(epoch / scheduler.total_epochs, 1.0) * 100
        print(f"{epoch:<6} {temp:<12.6f} {progress:<10.1f}%")
    
    # Show detailed progress info for middle epoch
    print(f"\nDetailed Progress Info (Epoch 5):")
    info = scheduler.get_progress_info(5)
    for key, value in info.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")


def demo_egonce_with_scheduler():
    """
    Demonstrate EgoNCE with temperature scheduling.
    """
    print("\n" + "=" * 50)
    print("EgoNCE with Temperature Scheduling Demo")
    print("=" * 50)
    
    # Create EgoNCE with scheduler
    loss_fn = EgoNCEWithScheduler(
        tau_max=0.07,
        tau_min=0.03,
        total_epochs=10,
        noun=True,
        verb=True
    )
    
    # Create mock similarity matrix and masks
    batch_size = 8
    similarity_matrix = torch.randn(batch_size, batch_size) * 0.5
    
    # Create masks (simplified for demo)
    mask_v = torch.eye(batch_size) + torch.randn(batch_size, batch_size) * 0.1
    mask_n = torch.eye(batch_size) + torch.randn(batch_size, batch_size) * 0.1
    mask_v = (mask_v > 0).float()
    mask_n = (mask_n > 0).float()
    
    print(f"Input similarity matrix shape: {similarity_matrix.shape}")
    
    # Simulate training across epochs
    print(f"\nTraining Simulation:")
    print(f"{'Epoch':<6} {'Loss':<10} {'Temperature':<12} {'Decay %':<8}")
    print("-" * 38)
    
    losses = []
    temperatures = []
    
    for epoch in range(11):  # 0 to 10
        loss, loss_info = loss_fn(similarity_matrix, mask_v, mask_n, current_epoch=epoch)
        
        losses.append(loss.item())
        temperatures.append(loss_info['current_temperature'])
        
        decay_percent = loss_info.get('decay_ratio', 0) * 100
        
        print(f"{epoch:<6} {loss.item():<10.6f} {loss_info['current_temperature']:<12.6f} {decay_percent:<8.1f}%")
    
    # Show temperature schedule preview
    print(f"\nComplete Temperature Schedule Preview:")
    schedule = loss_fn.get_temperature_schedule_preview()
    
    epochs = [item[0] for item in schedule]
    temps = [item[1] for item in schedule]
    
    print(f"Epochs: {epochs}")
    print(f"Temps:  {[f'{t:.4f}' for t in temps]}")
    
    return epochs, temps, losses


def demo_comparison_with_fixed_temperature():
    """
    Compare scheduled vs fixed temperature EgoNCE.
    """
    print("\n" + "=" * 50)
    print("Scheduled vs Fixed Temperature Comparison")
    print("=" * 50)
    
    # Create both loss functions
    scheduled_loss = EgoNCEWithScheduler(tau_max=0.07, tau_min=0.03, total_epochs=10)
    fixed_loss = EgoNCE(temperature=0.05)  # Fixed at middle value
    
    # Mock data
    batch_size = 8
    similarity_matrix = torch.randn(batch_size, batch_size) * 0.5
    mask_v = torch.eye(batch_size)
    mask_n = torch.eye(batch_size)
    
    print(f"Comparison Results:")
    print(f"{'Epoch':<6} {'Scheduled Loss':<15} {'Scheduled Temp':<15} {'Fixed Loss':<12} {'Fixed Temp':<12}")
    print("-" * 65)
    
    scheduled_losses = []
    fixed_losses = []
    scheduled_temps = []
    
    for epoch in range(11):
        # Scheduled temperature loss
        sched_loss, sched_info = scheduled_loss(similarity_matrix, mask_v, mask_n, current_epoch=epoch)
        scheduled_losses.append(sched_loss.item())
        scheduled_temps.append(sched_info['current_temperature'])
        
        # Fixed temperature loss
        fixed_loss_val = fixed_loss(similarity_matrix, mask_v, mask_n)
        fixed_losses.append(fixed_loss_val.item())
        
        print(f"{epoch:<6} {sched_loss.item():<15.6f} {sched_info['current_temperature']:<15.6f} "
              f"{fixed_loss_val.item():<12.6f} {fixed_loss.temperature:<12.6f}")
    
    return scheduled_losses, fixed_losses, scheduled_temps


def visualize_temperature_schedule():
    """
    Create visualization of temperature schedule (if matplotlib available).
    """
    print("\n" + "=" * 50)
    print("Temperature Schedule Visualization")
    print("=" * 50)
    
    try:
        # Create scheduler
        scheduler = TemperatureScheduler(tau_max=0.07, tau_min=0.03, total_epochs=10)
        
        # Generate temperature curve
        epochs = np.linspace(0, 10, 100)
        temperatures = [scheduler.get_temperature(epoch) for epoch in epochs]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, temperatures, 'b-', linewidth=2, label='Cosine Temperature Schedule')
        plt.axhline(y=0.07, color='r', linestyle='--', alpha=0.7, label='tau_max = 0.07')
        plt.axhline(y=0.03, color='g', linestyle='--', alpha=0.7, label='tau_min = 0.03')
        plt.xlabel('Epoch')
        plt.ylabel('Temperature (τ)')
        plt.title('Cosine Temperature Scheduling for Contrastive Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plt.savefig('temperature_schedule.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✓ Temperature schedule visualization saved as 'temperature_schedule.png'")
        
    except ImportError:
        print("✗ Matplotlib not available for visualization")
        print("Install matplotlib to see temperature schedule plot: pip install matplotlib")


def demo_integration_with_training_loop():
    """
    Show how to integrate temperature scheduling with actual training loop.
    """
    print("\n" + "=" * 50)
    print("Training Loop Integration Example")
    print("=" * 50)
    
    # Mock model and optimizer
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.video_proj = nn.Linear(512, 256)
            self.text_proj = nn.Linear(768, 256)
        
        def forward(self, video_features, text_features):
            v_proj = self.video_proj(video_features)
            t_proj = self.text_proj(text_features)
            
            # Normalize and compute similarity
            v_norm = torch.nn.functional.normalize(v_proj, dim=1)
            t_norm = torch.nn.functional.normalize(t_proj, dim=1)
            similarity = torch.mm(v_norm, t_norm.t())
            
            return similarity
    
    model = MockModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Create loss with temperature scheduling
    criterion = EgoNCEWithScheduler(tau_max=0.07, tau_min=0.03, total_epochs=5)
    
    print("Training Loop Simulation:")
    print("-" * 30)
    
    for epoch in range(5):
        model.train()
        epoch_losses = []
        
        # Simulate multiple batches per epoch
        for batch in range(3):
            # Mock batch data
            video_features = torch.randn(16, 512)
            text_features = torch.randn(16, 768)
            
            # Mock masks
            mask_v = torch.eye(16)
            mask_n = torch.eye(16)
            
            optimizer.zero_grad()
            
            # Forward pass
            similarity = model(video_features, text_features)
            
            # Compute loss with current epoch
            loss, loss_info = criterion(similarity, mask_v, mask_n, current_epoch=epoch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        # Epoch summary
        avg_loss = np.mean(epoch_losses)
        current_temp = criterion.temperature_scheduler.get_temperature(epoch)
        
        print(f"Epoch {epoch}: Loss = {avg_loss:.6f}, Temperature = {current_temp:.6f}")
    
    print("\n✓ Training loop integration demonstrated successfully")


if __name__ == '__main__':
    print("Cosine Temperature Scheduling for EgoNCE Loss")
    print("=" * 60)
    
    # Run all demonstrations
    demo_temperature_scheduler()
    epochs, temps, losses = demo_egonce_with_scheduler()
    scheduled_losses, fixed_losses, scheduled_temps = demo_comparison_with_fixed_temperature()
    demo_integration_with_training_loop()
    
    # Optional visualization
    try:
        visualize_temperature_schedule()
    except Exception as e:
        print(f"Visualization skipped: {e}")
    
    print("\n" + "=" * 60)
    print("TEMPERATURE SCHEDULING DEMO COMPLETED!")
    print("=" * 60)
    
    print("\nKey Benefits of Temperature Scheduling:")
    print("✓ Cosine decay from tau_max=0.07 to tau_min=0.03")
    print("✓ Smooth temperature transition over training epochs")
    print("✓ Easy integration with existing EgoNCE loss")
    print("✓ Detailed logging and progress tracking")
    print("✓ Better convergence through adaptive difficulty")
    
    print("\nFormula Used:")
    print("tau(epoch) = tau_min + (tau_max - tau_min) * 0.5 * [1 + cos(pi * epoch / total_epochs)]")
    
    print("\nUsage in Training:")
    print("1. Replace EgoNCE(temperature=0.05) with EgoNCEWithScheduler(...)")
    print("2. Call loss_fn(x, mask_v, mask_n, current_epoch=epoch)")
    print("3. Monitor temperature and loss info for debugging")
    print("4. Enjoy improved contrastive learning convergence!")