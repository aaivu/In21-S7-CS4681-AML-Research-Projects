import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class CAFELoss(nn.Module):
    """Multi-objective loss function for CAFE framework."""
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.3):
        """
        Initialize CAFE loss with weighting parameters.
        
        Args:
            alpha: Weight for toxicity loss
            beta: Weight for fairness loss  
            gamma: Weight for context loss
        """
        super(CAFELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse_loss = nn.MSELoss()
        
    def toxicity_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Toxicity Loss: MSE between predicted and true toxicity scores.
        
        Args:
            predicted: Predicted toxicity scores [batch_size]
            target: Target toxicity scores [batch_size]
            
        Returns:
            Toxicity loss value
        """
        return self.mse_loss(predicted, target)
    
    def fairness_loss(self, predicted: torch.Tensor, sensitive_groups: torch.Tensor) -> torch.Tensor:
        """
        Fairness Loss: Penalize disparities between demographic groups.
        
        Args:
            predicted: Predicted toxicity scores [batch_size]
            sensitive_groups: Binary group indicators [batch_size] (0 or 1)
            
        Returns:
            Fairness loss value
        """
        # Mask for each group
        group_0_mask = (sensitive_groups == 0)
        group_1_mask = (sensitive_groups == 1)
        
        # Check if both groups exist in batch
        if not (group_0_mask.any() and group_1_mask.any()):
            return torch.tensor(0.0, device=predicted.device, requires_grad=True)
        
        # Calculate mean predictions for each group
        group_0_mean = predicted[group_0_mask].mean()
        group_1_mean = predicted[group_1_mask].mean()
        
        # Return absolute difference
        return torch.abs(group_0_mean - group_1_mean)
    
    def context_loss(self, embeddings: torch.Tensor, context_labels: torch.Tensor, 
                    reference_embedding: torch.Tensor = None) -> torch.Tensor:
        """
        Context Loss: Promote context awareness through embedding similarity.
        
        Args:
            embeddings: Contextualized embeddings [batch_size, hidden_dim]
            context_labels: Context indicators [batch_size] (0=literal, 1=non-literal)
            reference_embedding: Reference embedding for non-literal contexts
            
        Returns:
            Context loss value
        """
        # If no reference embedding provided, compute from non-literal examples
        if reference_embedding is None:
            non_literal_mask = (context_labels == 1)
            if not non_literal_mask.any():
                return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
            reference_embedding = embeddings[non_literal_mask].mean(dim=0)
        
        # Calculate cosine similarities with reference
        similarities = F.cosine_similarity(embeddings, reference_embedding.unsqueeze(0), dim=1)
        
        # We want high similarity for non-literal contexts, low for literal
        # Loss = 1 - mean similarity for non-literal contexts
        non_literal_mask = (context_labels == 1)
        if non_literal_mask.any():
            non_literal_similarities = similarities[non_literal_mask]
            return 1.0 - non_literal_similarities.mean()
        
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor, 
                embeddings: torch.Tensor, sensitive_groups: torch.Tensor,
                context_labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total CAFE loss.
        
        Args:
            predicted: Predicted toxicity scores [batch_size]
            target: Target toxicity scores [batch_size] 
            embeddings: Contextualized embeddings [batch_size, hidden_dim]
            sensitive_groups: Binary group indicators [batch_size]
            context_labels: Context indicators [batch_size]
            
        Returns:
            total_loss: Combined loss value
            loss_components: Dictionary of individual loss components
        """
        # Calculate individual loss components
        tox_loss = self.toxicity_loss(predicted, target)
        fair_loss = self.fairness_loss(predicted, sensitive_groups)
        ctx_loss = self.context_loss(embeddings, context_labels)
        
        # Combine losses with weights
        total_loss = (self.alpha * tox_loss + 
                     self.beta * fair_loss + 
                     self.gamma * ctx_loss)
        
        # Return loss and components for monitoring
        loss_components = {
            'total_loss': total_loss.item(),
            'toxicity_loss': tox_loss.item(),
            'fairness_loss': fair_loss.item(), 
            'context_loss': ctx_loss.item()
        }
        
        return total_loss, loss_components

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in toxicity detection."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            predicted: Predicted probabilities [batch_size]
            target: Target labels [batch_size]
            
        Returns:
            Focal loss value
        """
        bce_loss = F.binary_cross_entropy(predicted, target, reduction='none')
        p_t = torch.where(target == 1, predicted, 1 - predicted)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * bce_loss
        return focal_loss.mean()

class ContrastiveLoss(nn.Module):
    """Contrastive loss for learning better embeddings."""
    
    def __init__(self, margin: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            embeddings1: First set of embeddings [batch_size, hidden_dim]
            embeddings2: Second set of embeddings [batch_size, hidden_dim] 
            labels: Similarity labels [batch_size] (1=similar, 0=dissimilar)
            
        Returns:
            Contrastive loss value
        """
        euclidean_distance = F.pairwise_distance(embeddings1, embeddings2)
        
        loss_contrastive = torch.mean(
            labels * torch.pow(euclidean_distance, 2) +
            (1 - labels) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        
        return loss_contrastive

if __name__ == "__main__":
    # Test loss functions
    batch_size = 16
    hidden_dim = 768
    
    # Create dummy data
    predicted = torch.rand(batch_size)
    target = torch.rand(batch_size) 
    embeddings = torch.randn(batch_size, hidden_dim)
    sensitive_groups = torch.randint(0, 2, (batch_size,))
    context_labels = torch.randint(0, 2, (batch_size,))
    
    # Test CAFE loss
    cafe_loss = CAFELoss()
    total_loss, components = cafe_loss(predicted, target, embeddings, sensitive_groups, context_labels)
    
    print(f"Total Loss: {total_loss.item():.4f}")
    for component, value in components.items():
        print(f"{component}: {value:.4f}")