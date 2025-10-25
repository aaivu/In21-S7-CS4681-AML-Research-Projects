import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import json
from pathlib import Path
from .enhancements import TrainingEnhancements

class ResNetTrainer:
    def __init__(self, config):
        self.config = config
        self.enhancements = TrainingEnhancements(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_data()
        self.setup_model()
        
    def setup_data(self):
        """Setup CIFAR-10 dataset"""
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        self.train_dataset = torchvision.datasets.CIFAR10(
            root='./data/cifar10', train=True, download=True, transform=transform_train)
        self.test_dataset = torchvision.datasets.CIFAR10(
            root='./data/cifar10', train=False, download=True, transform=transform_test)
        
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=4)
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=4)
    
    def setup_model(self):
        """Setup ResNet-50 model"""
        self.model = torchvision.models.resnet50(pretrained=False, num_classes=10)
        self.model = self.model.to(self.device)
        
        if self.config.use_activation_checkpointing:
            self.model = self.enhancements.apply_activation_checkpointing(self.model)
            
        self.optimizer = self.enhancements.create_optimizer(self.model)
        self.criterion = nn.CrossEntropyLoss()
        
        num_training_steps = len(self.train_loader) * self.config.num_epochs
        self.scheduler = self.enhancements.create_scheduler(self.optimizer, num_training_steps)
        self.enhancements.setup_mixed_precision()
    
    def train_epoch(self, epoch):
        """Train one epoch with enhancements"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        
        # Dynamic batch scheduling
        current_batch_size = self.enhancements.dynamic_batch_schedule(epoch, self.config.num_epochs)
        
        for i, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss = loss / self.enhancements.gradient_accumulation_steps
            
            # Mixed precision backward pass
            self.enhancements.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (i + 1) % self.enhancements.gradient_accumulation_steps == 0:
                self.enhancements.scaler.step(self.optimizer)
                self.enhancements.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
            
            running_loss += loss.item() * self.enhancements.gradient_accumulation_steps
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        epoch_time = time.time() - start_time
        accuracy = 100. * correct / total
        avg_loss = running_loss / len(self.train_loader)
        
        return avg_loss, accuracy, epoch_time
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy
    
    def train(self):
        """Complete training loop"""
        results = {
            'train_loss': [], 'train_acc': [], 'test_acc': [], 
            'epoch_times': [], 'memory_usage': []
        }
        
        for epoch in range(self.config.num_epochs):
            train_loss, train_acc, epoch_time = self.train_epoch(epoch)
            test_acc = self.validate()
            
            # Record memory usage
            memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
            
            results['train_loss'].append(train_loss)
            results['train_acc'].append(train_acc)
            results['test_acc'].append(test_acc)
            results['epoch_times'].append(epoch_time)
            results['memory_usage'].append(memory_allocated)
            
            print(f'Epoch {epoch+1}/{self.config.num_epochs}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Test Acc: {test_acc:.2f}%, Time: {epoch_time:.2f}s, '
                  f'Memory: {memory_allocated:.2f}GB')
        
        return results