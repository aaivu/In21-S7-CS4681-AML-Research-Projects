import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluates the model on the test dataset and returns the accuracy and loss.
    
    Parameters:
        model (nn.Module): The model to be evaluated.
        test_loader (DataLoader): DataLoader for the test dataset.
        criterion (callable): Loss function to evaluate the model.
    
    Returns:
        float: Accuracy of the model on the test dataset.
        float: Loss of the model on the test dataset.
    """
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Handle datasets that provide sequence labels by selecting the next token
            if labels.dim() > 1:
                labels = labels[:, -1]
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(test_loader)
    
    return accuracy, avg_loss
