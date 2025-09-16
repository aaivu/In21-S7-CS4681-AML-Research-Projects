import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List

# Import for Integrated Gradients
try:
    from captum.attr import IntegratedGradients
except ImportError:
    print("Warning: captum is not installed. Feature attribution will not be available.")
    print("Please install it using: pip install captum")
    IntegratedGradients = None

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class AdaptiveMovingAvg(nn.Module):
    """
    Adaptive Moving Average block with learnable kernel and improved padding strategy
    """
    def __init__(self, kernel_size: int, stride: int = 1, learnable: bool = True):
        super(AdaptiveMovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.learnable = learnable
        
        if learnable:
            # Learnable weights initialized to uniform average
            self.weights = nn.Parameter(torch.ones(kernel_size) / kernel_size)
        else:
            # Standard average pooling
            self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Seq_len, Channels]
        if self.learnable:
            # Use learnable convolution for moving average
            batch_size, seq_len, channels = x.shape
            
            # Replicate padding (more appropriate for time series)
            pad_left = (self.kernel_size - 1) // 2
            pad_right = self.kernel_size - 1 - pad_left
            
            # Replicate boundary values
            front = x[:, :1, :].repeat(1, pad_left, 1)
            end = x[:, -1:, :].repeat(1, pad_right, 1)
            x_padded = torch.cat([front, x, end], dim=1)
            
            # Apply learnable convolution
            x_padded = x_padded.permute(0, 2, 1)  # [B, C, L]
            weights = self.weights.view(1, 1, -1).expand(channels, 1, -1)
            result = F.conv1d(x_padded, weights, groups=channels, stride=self.stride)
            return result.permute(0, 2, 1)  # [B, L, C]
        else:
            # Standard implementation with improved padding
            pad_left = (self.kernel_size - 1) // 2
            pad_right = self.kernel_size - 1 - pad_left
            
            front = x[:, :1, :].repeat(1, pad_left, 1)
            end = x[:, -1:, :].repeat(1, pad_right, 1)
            x = torch.cat([front, x, end], dim=1)
            x = self.avg(x.permute(0, 2, 1))
            return x.permute(0, 2, 1)

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size, is_adaptive: bool = False):
        super(series_decomp, self).__init__()
        if is_adaptive:
            self.moving_avg = AdaptiveMovingAvg(kernel_size, stride=1, learnable=True)
        else:
            self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
    
class MultiScaleDecomposition(nn.Module):
    """
    Multi-scale series decomposition with multiple kernel sizes
    """
    def __init__(self, kernel_sizes: list = [9, 25, 49], learnable: bool = False, is_adaptive: bool = False):
        super(MultiScaleDecomposition, self).__init__()
        self.kernel_sizes = kernel_sizes
        if is_adaptive:
            self.moving_avgs = nn.ModuleList([
                AdaptiveMovingAvg(k, stride=1) 
                for k in kernel_sizes
            ])
        else:
            self.moving_avgs = nn.ModuleList([
                moving_avg(k, stride=1) 
                for k in kernel_sizes
            ])
        
        # Learnable weights for combining different scales
        self.scale_weights = nn.Parameter(torch.ones(len(kernel_sizes)) / len(kernel_sizes))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute trends at different scales
        trends = []
        for moving_avg in self.moving_avgs:
            trend = moving_avg(x)
            trends.append(trend)
        
        # Weighted combination of trends
        trend = sum(w * t for w, t in zip(F.softmax(self.scale_weights, dim=0), trends))
        seasonal = x - trend
        
        return seasonal, trend

class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        is_multi_scale = configs.multi_scale
        is_adaptive = configs.adaptive

        # Decompsition Kernel Size
        kernel_size = 25
        if is_multi_scale:
            self.decompsition = MultiScaleDecomposition(kernel_sizes=[9, 25, 49], learnable=True, is_adaptive=is_adaptive)
        else:
            self.decompsition = series_decomp(kernel_size, is_adaptive=is_adaptive)
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
            
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0,2,1) # to [Batch, Output length, Channel]

    def feature_attribution(self, x_batch: torch.Tensor, y_batch: torch.Tensor, feature_names: List[str], target_feature_idx: int = 7) -> Dict:
        """
        Performs feature attribution analysis using Integrated Gradients and Permutation Importance.

        Args:
            x_batch (torch.Tensor): Input batch of shape [Batch, Seq_len, Channel].
            y_batch (torch.Tensor): True values batch of shape [Batch, Pred_len, Channel].
            feature_names (List[str]): List of feature names.
            target_feature_idx (int): Index of the target feature to explain ('OT').

        Returns:
            Dict: A dictionary containing attribution scores for each method.
        """
        self.eval()  # Ensure the model is in evaluation mode
        
        results = {
            'integrated_gradients': {},
            'permutation_importance': {}
        }

        # --- 1. Integrated Gradients ---
        # Explains a single prediction instance. We'll analyze the first item in the batch.
        if IntegratedGradients:
            try:
                ig = IntegratedGradients(self)
                # We want to explain the prediction for the target feature at the first prediction step (t=0)
                # The target is (batch_index, prediction_step, feature_index)
                target_tuple = (0, target_feature_idx)
                
                # Calculate attributions for the first instance in the batch
                attributions_ig = ig.attribute(x_batch, target=target_tuple, internal_batch_size=x_batch.size(0))
                
                # To get a single score per feature, we sum the attributions over the sequence length
                # This shows the total contribution of each feature over the entire look-back window
                feature_importance_ig = attributions_ig.sum(dim=1).mean(dim=0).cpu().numpy()
                
                results['integrated_gradients'] = {name: score for name, score in zip(feature_names, feature_importance_ig)}
            except Exception as e:
                print(f"Warning: Integrated Gradients calculation failed. {e}")
                results['integrated_gradients'] = {name: 0.0 for name in feature_names}
        
        # --- 2. Permutation Importance ---
        # Measures the drop in performance when a feature is shuffled.
        try:
            criterion = nn.MSELoss()
            baseline_preds = self(x_batch)
            baseline_loss = criterion(baseline_preds, y_batch)
            
            perm_importance_scores = {}
            for i, name in enumerate(feature_names):
                original_col = x_batch[:, :, i].clone()
                
                # Permute (shuffle) the current feature column
                perm_indices = torch.randperm(x_batch.size(0))
                x_batch_perm = x_batch.clone()
                x_batch_perm[:, :, i] = x_batch_perm[perm_indices, :, i]
                
                # Get prediction with the permuted feature and calculate loss
                perm_preds = self(x_batch_perm)
                perm_loss = criterion(perm_preds, y_batch)
                
                # The importance is the increase in loss
                importance = (perm_loss - baseline_loss).item()
                perm_importance_scores[name] = importance
            
            results['permutation_importance'] = perm_importance_scores
        except Exception as e:
            print(f"Warning: Permutation Importance calculation failed. {e}")
            results['permutation_importance'] = {name: 0.0 for name in feature_names}
            
        return results