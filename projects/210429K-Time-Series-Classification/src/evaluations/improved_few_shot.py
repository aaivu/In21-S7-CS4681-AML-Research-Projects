#!/usr/bin/env python3
"""
Improved Few-Shot Learning Methods for TNC Encoder
=================================================

This module implements improved few-shot learning methods specifically designed
to work well with TNC encoder features that are linearly separable.

Includes:
1. Linear Prototypical Networks
2. Learnable Distance Metrics
3. Hybrid Linear-Prototypical Approach
4. Meta-Learning Integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score


class LinearPrototypicalNetwork(nn.Module):
    """
    Prototypical Networks with learnable linear transformation
    
    Improves standard prototypical networks by learning a linear transformation
    that makes the feature space more suitable for prototype-based classification.
    """
    
    def __init__(self, input_dim, output_dim=None, temperature=1.0):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
            
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.temperature = temperature
        
        # Learnable linear transformation
        self.transform = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, support_features, support_labels, query_features, n_way, n_shot):
        """
        Forward pass for few-shot episode
        
        Args:
            support_features: (n_way * n_shot, input_dim)
            support_labels: (n_way * n_shot,)
            query_features: (n_query, input_dim)
            n_way: Number of classes
            n_shot: Number of examples per class
        """
        # Transform features to better prototype space
        support_transformed = self.transform(support_features)
        query_transformed = self.transform(query_features)
        
        # Compute prototypes for each class
        prototypes = []
        for class_idx in range(n_way):
            class_mask = support_labels == class_idx
            if class_mask.sum() > 0:
                class_features = support_transformed[class_mask]
                prototype = class_features.mean(dim=0)
                prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes)  # (n_way, output_dim)
        
        # Compute distances and convert to logits
        distances = torch.cdist(query_transformed, prototypes)  # (n_query, n_way)
        logits = -distances / self.temperature
        
        return logits


class MetricPrototypicalNetwork(nn.Module):
    """
    Prototypical Networks with learnable distance metric
    
    Instead of using Euclidean distance, learns a neural network to compute
    distances between query examples and prototypes.
    """
    
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Distance metric network
        self.distance_net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def compute_distance(self, query, prototype):
        """Compute learnable distance between query and prototype"""
        # Concatenate query and prototype
        combined = torch.cat([query, prototype], dim=-1)
        distance = self.distance_net(combined)
        return distance.squeeze(-1)
    
    def forward(self, support_features, support_labels, query_features, n_way, n_shot):
        """Forward pass with learnable distance metric"""
        # Compute prototypes
        prototypes = []
        for class_idx in range(n_way):
            class_mask = support_labels == class_idx
            if class_mask.sum() > 0:
                class_features = support_features[class_mask]
                prototype = class_features.mean(dim=0)
                prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes)  # (n_way, input_dim)
        
        # Compute learnable distances
        n_query = query_features.shape[0]
        distances = torch.zeros(n_query, n_way, device=query_features.device)
        
        for i, query in enumerate(query_features):
            for j, prototype in enumerate(prototypes):
                # Expand dimensions for batch processing
                query_expanded = query.unsqueeze(0)  # (1, input_dim)
                prototype_expanded = prototype.unsqueeze(0)  # (1, input_dim)
                distance = self.compute_distance(query_expanded, prototype_expanded)
                distances[i, j] = distance
        
        # Convert distances to logits (lower distance = higher logit)
        logits = -distances * 10  # Scale factor for stability
        
        return logits


class HybridFewShotClassifier(nn.Module):
    """
    Hybrid approach combining Linear classifier and Prototypical Networks
    
    Uses a learnable combination of both methods to get the best of both worlds.
    """
    
    def __init__(self, input_dim, n_classes=4, hidden_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        
        # Linear classifier head
        self.linear_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes)
        )
        
        # Prototypical network component
        self.proto_transform = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Fusion weights (learnable combination)
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, support_features, support_labels, query_features, n_way, n_shot):
        """Hybrid forward pass"""
        # 1. Linear classifier branch
        # Train linear classifier on support set
        linear_logits = self._linear_branch(support_features, support_labels, query_features)
        
        # 2. Prototypical network branch
        proto_logits = self._prototypical_branch(support_features, support_labels, query_features, n_way)
        
        # 3. Learnable fusion
        fusion_weight = torch.sigmoid(self.fusion_weight)  # Ensure [0,1]
        combined_logits = (fusion_weight * proto_logits + 
                          (1 - fusion_weight) * linear_logits)
        
        return combined_logits, linear_logits, proto_logits, fusion_weight
    
    def _linear_branch(self, support_features, support_labels, query_features):
        """Linear classifier branch"""
        # Simple forward pass through linear head
        query_logits = self.linear_head(query_features)
        return query_logits
    
    def _prototypical_branch(self, support_features, support_labels, query_features, n_way):
        """Prototypical network branch"""
        # Transform features
        support_transformed = self.proto_transform(support_features)
        query_transformed = self.proto_transform(query_features)
        
        # Compute prototypes
        prototypes = []
        for class_idx in range(n_way):
            class_mask = support_labels == class_idx
            if class_mask.sum() > 0:
                class_features = support_transformed[class_mask]
                prototype = class_features.mean(dim=0)
                prototypes.append(prototype)
        
        if len(prototypes) == 0:
            return torch.zeros(query_features.shape[0], n_way, device=query_features.device)
            
        prototypes = torch.stack(prototypes)
        
        # Compute distances and convert to logits
        distances = torch.cdist(query_transformed, prototypes)
        logits = -distances / torch.abs(self.temperature)
        
        return logits


class AdaptivePrototypicalNetwork(nn.Module):
    """
    Adaptive Prototypical Networks that adjust based on shot number
    
    Uses different strategies for different numbers of shots to optimize performance.
    """
    
    def __init__(self, input_dim, max_shots=20):
        super().__init__()
        self.input_dim = input_dim
        self.max_shots = max_shots
        
        # Different transformations for different shot regimes
        self.few_shot_transform = nn.Sequential(  # For 1-3 shots
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim)
        )
        
        self.medium_shot_transform = nn.Sequential(  # For 4-10 shots
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, input_dim)
        )
        
        self.many_shot_transform = nn.Sequential(  # For 10+ shots
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim)
        )
        
        # Shot-dependent temperature parameters
        self.temperatures = nn.Parameter(torch.tensor([2.0, 1.5, 1.0]))  # [few, medium, many]
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, support_features, support_labels, query_features, n_way, n_shot):
        """Adaptive forward pass based on shot number"""
        # Choose transformation and temperature based on shot number
        if n_shot <= 3:
            transform = self.few_shot_transform
            temperature = torch.abs(self.temperatures[0])
        elif n_shot <= 10:
            transform = self.medium_shot_transform
            temperature = torch.abs(self.temperatures[1])
        else:
            transform = self.many_shot_transform
            temperature = torch.abs(self.temperatures[2])
        
        # Transform features
        support_transformed = transform(support_features)
        query_transformed = transform(query_features)
        
        # Compute prototypes with shot-aware aggregation
        prototypes = []
        for class_idx in range(n_way):
            class_mask = support_labels == class_idx
            if class_mask.sum() > 0:
                class_features = support_transformed[class_mask]
                
                # Different aggregation strategies
                if n_shot <= 3:
                    # Simple mean for few shots
                    prototype = class_features.mean(dim=0)
                else:
                    # Weighted mean (reduce influence of outliers)
                    weights = F.softmax(torch.norm(class_features, dim=1), dim=0)
                    prototype = (class_features * weights.unsqueeze(1)).sum(dim=0)
                
                prototypes.append(prototype)
        
        if len(prototypes) == 0:
            return torch.zeros(query_features.shape[0], n_way, device=query_features.device)
            
        prototypes = torch.stack(prototypes)
        
        # Compute distances and logits
        distances = torch.cdist(query_transformed, prototypes)
        logits = -distances / temperature
        
        return logits


class ImprovedFewShotEvaluator:
    """
    Evaluator for improved few-shot learning methods
    """
    
    def __init__(self, feature_extractor, device='cpu'):
        self.feature_extractor = feature_extractor
        self.device = device
        
        # Initialize improved models
        self.models = {
            'linear_prototypical': LinearPrototypicalNetwork(input_dim=10, output_dim=16).to(device),
            'metric_prototypical': MetricPrototypicalNetwork(input_dim=10).to(device),
            'hybrid': HybridFewShotClassifier(input_dim=10, n_classes=4).to(device),
            'adaptive': AdaptivePrototypicalNetwork(input_dim=10).to(device)
        }
    
    def evaluate_method(self, model_name, X_train, y_train, X_test, y_test, 
                       n_way=4, n_shot=5, n_episodes=50):
        """Evaluate a specific improved method"""
        model = self.models[model_name]
        model.eval()
        
        accuracies = []
        
        for episode in range(n_episodes):
            # Sample episode
            support_x, support_y, query_x, query_y = self._sample_episode(
                X_train, y_train, n_way, n_shot, n_query=15
            )
            
            if len(support_x) == 0 or len(query_x) == 0:
                continue
            
            # Convert to tensors
            support_features = torch.tensor(support_x, dtype=torch.float32).to(self.device)
            support_labels = torch.tensor(support_y, dtype=torch.long).to(self.device)
            query_features = torch.tensor(query_x, dtype=torch.float32).to(self.device)
            query_labels = torch.tensor(query_y, dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                if model_name == 'hybrid':
                    logits, _, _, _ = model(support_features, support_labels, 
                                         query_features, n_way, n_shot)
                else:
                    logits = model(support_features, support_labels, 
                                 query_features, n_way, n_shot)
                
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == query_labels).float().mean().item()
                accuracies.append(accuracy)
        
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        return mean_acc, std_acc
    
    def _sample_episode(self, X_train, y_train, n_way, n_shot, n_query):
        """Sample a few-shot learning episode"""
        available_classes = np.unique(y_train)
        if len(available_classes) < n_way:
            return [], [], [], []
        
        episode_classes = np.random.choice(available_classes, n_way, replace=False)
        
        support_x, support_y, query_x, query_y = [], [], [], []
        
        for class_idx, class_label in enumerate(episode_classes):
            class_indices = np.where(y_train == class_label)[0]
            
            if len(class_indices) < n_shot + n_query:
                continue
            
            selected_indices = np.random.choice(
                class_indices, n_shot + n_query, replace=False
            )
            
            # Support set
            support_indices = selected_indices[:n_shot]
            support_x.extend(X_train[support_indices])
            support_y.extend([class_idx] * n_shot)
            
            # Query set
            query_indices = selected_indices[n_shot:n_shot + n_query]
            query_x.extend(X_train[query_indices])
            query_y.extend([class_idx] * n_query)
        
        return np.array(support_x), np.array(support_y), np.array(query_x), np.array(query_y)
    
    def train_method_few_shot(self, model_name, X_train, y_train, n_way=4, 
                             n_shot=5, n_episodes=100, lr=0.001):
        """Train an improved method using few-shot episodes"""
        model = self.models[model_name]
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        print(f"🔄 Training {model_name}...")
        
        for episode in range(n_episodes):
            # Sample training episode
            support_x, support_y, query_x, query_y = self._sample_episode(
                X_train, y_train, n_way, n_shot, n_query=10
            )
            
            if len(support_x) == 0 or len(query_x) == 0:
                continue
            
            # Convert to tensors
            support_features = torch.tensor(support_x, dtype=torch.float32).to(self.device)
            support_labels = torch.tensor(support_y, dtype=torch.long).to(self.device)
            query_features = torch.tensor(query_x, dtype=torch.float32).to(self.device)
            query_labels = torch.tensor(query_y, dtype=torch.long).to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            
            if model_name == 'hybrid':
                logits, _, _, _ = model(support_features, support_labels, 
                                     query_features, n_way, n_shot)
            else:
                logits = model(support_features, support_labels, 
                             query_features, n_way, n_shot)
            
            loss = criterion(logits, query_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if episode % 20 == 0:
                print(f"   Episode {episode}/{n_episodes}, Loss: {loss.item():.4f}")
        
        print(f"✅ Finished training {model_name}")
    
    def comprehensive_evaluation(self, X_train, y_train, X_test, y_test, 
                                shot_numbers=[1, 3, 5, 10], n_episodes=50):
        """Run comprehensive evaluation of all improved methods"""
        print("🚀 Starting Improved Few-Shot Learning Evaluation")
        print("=" * 60)
        
        results = {}
        
        for model_name in self.models.keys():
            print(f"\n🎯 Evaluating {model_name}...")
            results[model_name] = {'shots': [], 'mean': [], 'std': []}
            
            # Train the model first
            self.train_method_few_shot(model_name, X_train, y_train, n_episodes=100)
            
            # Evaluate on different shot numbers
            for n_shot in shot_numbers:
                mean_acc, std_acc = self.evaluate_method(
                    model_name, X_train, y_train, X_test, y_test,
                    n_shot=n_shot, n_episodes=n_episodes
                )
                
                results[model_name]['shots'].append(n_shot)
                results[model_name]['mean'].append(mean_acc)
                results[model_name]['std'].append(std_acc)
                
                print(f"   {n_shot}-shot: {mean_acc:.3f} ± {std_acc:.3f}")
        
        return results


def create_improved_evaluator(encoder_path, data_path, device='cpu'):
    """Create an improved few-shot evaluator with trained TNC encoder"""
    from tnc.models import RnnEncoder
    import pickle
    import os
    
    # Load pre-trained encoder
    checkpoint = torch.load(encoder_path, map_location=device)
    encoder = RnnEncoder(hidden_size=100, in_channel=3, encoding_size=10, device=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.eval()
    encoder.to(device)
    
    # Create evaluator
    evaluator = ImprovedFewShotEvaluator(encoder, device)
    
    return evaluator, encoder


if __name__ == "__main__":
    # Example usage
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create evaluator
    evaluator, encoder = create_improved_evaluator(
        './ckpt/simulation/checkpoint_0.pth.tar',
        './data/simulated_data/',
        device
    )
    
    print("✅ Improved few-shot learning methods ready for evaluation!")
    print("🔧 Available methods:")
    for model_name in evaluator.models.keys():
        print(f"   - {model_name}")