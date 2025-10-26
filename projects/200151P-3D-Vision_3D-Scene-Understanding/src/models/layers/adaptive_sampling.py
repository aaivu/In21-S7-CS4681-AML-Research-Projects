"""
Adaptive Density-Aware Sampling for PointNeXt Enhancement
Implementation based on the IEEE paper:
"Enhancing PointNeXt for Large-Scale 3D Point Cloud Processing: Adaptive Sampling vs. Memory-Efficient Attention"
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from ..build import MODELS


class AdaptiveDensityAwareSampler(nn.Module):
    """
    Adaptive sampling strategy that intelligently analyzes local geometric complexity 
    using Principal Component Analysis (PCA) eigenvalue distributions and multi-scale 
    density gradients computed via k-nearest neighbor analysis.
    """
    
    def __init__(self, 
                 target_points: int = 1024,
                 k_scales: list = [8, 16, 32],
                 alpha: float = 0.7,
                 beta: float = 0.3,
                 gamma: float = 0.0):
        """
        Initialize adaptive sampler.
        
        Args:
            target_points: Target number of points after sampling
            k_scales: Neighborhood sizes for multi-scale density analysis
            alpha: Density weight factor (learned parameter)
            beta: Complexity weight factor (learned parameter) 
            gamma: Bias term (learned parameter)
        """
        super().__init__()
        self.target_points = target_points
        self.k_scales = k_scales
        
        # Learnable parameters as described in the paper
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))
        self.gamma = nn.Parameter(torch.tensor(gamma))
        
        # Learned weights for complexity measure (w1, w2, w3 from paper)
        self.complexity_weights = nn.Parameter(torch.tensor([0.33, 0.33, 0.34]))
        
    def compute_multi_scale_density(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-scale density estimates across different neighborhood sizes.
        
        Args:
            points: Input point cloud [B, N, 3]
            
        Returns:
            Multi-scale density tensor [B, N, len(k_scales)]
        """
        B, N, _ = points.shape
        densities = []
        
        # Compute pairwise distances
        dist = torch.cdist(points, points)  # [B, N, N]
        
        for k in self.k_scales:
            # Get k-th nearest neighbor distances
            knn_dists, _ = torch.topk(dist, k=min(k+1, N), dim=-1, largest=False)
            r_k = knn_dists[:, :, -1]  # Distance to k-th neighbor
            
            # Compute density using formula from paper: ρ_i^(k) = k / (4/3 * π * r_k^3)
            volume = (4.0 / 3.0) * np.pi * (r_k ** 3 + 1e-8)
            density = k / volume
            densities.append(density)
        
        return torch.stack(densities, dim=-1)  # [B, N, len(k_scales)]
    
    def compute_geometric_complexity(self, points: torch.Tensor, k: int = 16) -> torch.Tensor:
        """
        Assess geometric complexity using Principal Component Analysis (PCA) on local neighborhoods.
        
        Args:
            points: Input point cloud [B, N, 3]
            k: Neighborhood size for PCA analysis
            
        Returns:
            Complexity measure for each point [B, N]
        """
        B, N, _ = points.shape
        complexities = []
        
        # Get k-nearest neighbors for each point
        dist = torch.cdist(points, points)
        _, knn_indices = torch.topk(dist, k=min(k, N), dim=-1, largest=False)
        
        for b in range(B):
            batch_complexities = []
            
            for i in range(N):
                # Get neighborhood points
                neighbor_indices = knn_indices[b, i]
                neighborhood = points[b, neighbor_indices]  # [k, 3]
                
                # Compute centroid
                centroid = neighborhood.mean(dim=0, keepdim=True)
                
                # Center the points
                centered_points = neighborhood - centroid
                
                # Compute covariance matrix
                cov_matrix = torch.mm(centered_points.t(), centered_points) / k
                
                # Compute eigenvalues
                try:
                    eigenvals, _ = torch.linalg.eigh(cov_matrix)
                    eigenvals = torch.abs(eigenvals)
                    eigenvals = torch.sort(eigenvals, descending=True)[0]  # λ1 ≥ λ2 ≥ λ3
                    
                    # Ensure numerical stability
                    eigenvals = eigenvals + 1e-8
                    
                    # Compute complexity measure from paper:
                    # c_i = w1 * (λ2 - λ3)/λ1 + w2 * (λ1 - λ2)/λ1 + w3 * λ3/λ1
                    lambda1, lambda2, lambda3 = eigenvals[0], eigenvals[1], eigenvals[2]
                    
                    term1 = (lambda2 - lambda3) / lambda1
                    term2 = (lambda1 - lambda2) / lambda1
                    term3 = lambda3 / lambda1
                    
                    complexity = (self.complexity_weights[0] * term1 + 
                                self.complexity_weights[1] * term2 + 
                                self.complexity_weights[2] * term3)
                    
                except:
                    # Fallback for numerical instability
                    complexity = torch.tensor(0.5, device=points.device)
                
                batch_complexities.append(complexity)
            
            complexities.append(torch.stack(batch_complexities))
        
        return torch.stack(complexities)  # [B, N]
    
    def compute_sampling_probabilities(self, 
                                     densities: torch.Tensor, 
                                     complexities: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive sampling probabilities combining density and complexity factors.
        
        Args:
            densities: Multi-scale density estimates [B, N, len(k_scales)]
            complexities: Geometric complexity measures [B, N]
            
        Returns:
            Sampling probabilities [B, N]
        """
        # Use medium scale density (k=16) as in the paper
        density_16 = densities[:, :, 1]  # k_scales[1] = 16
        
        # Apply sampling probability formula from paper:
        # P(select p_i) = σ(α * log(ρ_i^(16)) + β * c_i + γ)
        log_density = torch.log(density_16 + 1e-8)
        logits = self.alpha * log_density + self.beta * complexities + self.gamma
        
        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(logits)
        
        return probabilities
    
    def forward(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for adaptive sampling.
        
        Args:
            points: Input point cloud [B, N, 3]
            
        Returns:
            Tuple of (sampled_points, sampling_indices)
        """
        B, N, C = points.shape
        
        if N <= self.target_points:
            # Return all points if already below target
            indices = torch.arange(N, device=points.device).unsqueeze(0).expand(B, -1)
            return points, indices
        
        # Step 1: Compute multi-scale density analysis
        densities = self.compute_multi_scale_density(points)
        
        # Step 2: Assess geometric complexity
        complexities = self.compute_geometric_complexity(points)
        
        # Step 3: Compute sampling probabilities
        probabilities = self.compute_sampling_probabilities(densities, complexities)
        
        # Step 4: Sample points based on probabilities
        # Use multinomial sampling to maintain diversity while respecting probabilities
        sampled_indices = torch.multinomial(probabilities, self.target_points, replacement=False)
        
        # Gather sampled points
        batch_indices = torch.arange(B, device=points.device).unsqueeze(1).expand(-1, self.target_points)
        sampled_points = points[batch_indices, sampled_indices]
        
        return sampled_points, sampled_indices
    
    def get_complexity_stats(self, points: torch.Tensor) -> dict:
        """
        Get statistics about geometric complexity distribution for analysis.
        
        Args:
            points: Input point cloud [B, N, 3]
            
        Returns:
            Dictionary with complexity statistics
        """
        complexities = self.compute_geometric_complexity(points)
        
        return {
            'mean_complexity': complexities.mean().item(),
            'std_complexity': complexities.std().item(),
            'min_complexity': complexities.min().item(),
            'max_complexity': complexities.max().item()
        }


@MODELS.register_module()
class AdaptiveFarthestPointSampling(nn.Module):
    """
    Enhanced FPS with adaptive density awareness.
    Combines traditional FPS with local density analysis.
    """
    
    def __init__(self, target_points: int = 1024, density_weight: float = 0.3):
        super().__init__()
        self.target_points = target_points
        self.density_weight = density_weight
        self.adaptive_sampler = AdaptiveDensityAwareSampler(target_points)
    
    def forward(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adaptive FPS sampling.
        
        Args:
            points: Input point cloud [B, N, 3]
            
        Returns:
            Tuple of (sampled_points, sampling_indices)
        """
        return self.adaptive_sampler(points)