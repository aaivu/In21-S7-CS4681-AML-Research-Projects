import pdb
import torch
import torch.nn.functional as F
from torch import nn
import pickle
import math


class TemperatureScheduler:
    """
    Cosine temperature scheduler for contrastive loss.
    
    Implements cosine decay from tau_max to tau_min using the formula:
    tau(epoch) = tau_min + (tau_max - tau_min) * 0.5 * [1 + cos(pi * epoch / total_epochs)]
    
    This provides smooth temperature decay that starts high for easier learning
    and gradually decreases to force the model to make harder distinctions.
    """
    
    def __init__(self, tau_max=0.07, tau_min=0.03, total_epochs=10):
        """
        Initialize temperature scheduler.
        
        Args:
            tau_max: Maximum temperature at the start of training
            tau_min: Minimum temperature at the end of training
            total_epochs: Total number of training epochs
        """
        self.tau_max = tau_max
        self.tau_min = tau_min
        self.total_epochs = total_epochs
        
        # Validate parameters
        assert tau_max > tau_min > 0, f"tau_max ({tau_max}) must be > tau_min ({tau_min}) > 0"
        assert total_epochs > 0, f"total_epochs ({total_epochs}) must be > 0"
        
    def get_temperature(self, current_epoch):
        """
        Compute current temperature based on epoch number.
        
        Args:
            current_epoch: Current training epoch (0-indexed)
            
        Returns:
            temperature: Current temperature value for contrastive loss
        """
        # Clamp epoch to valid range
        epoch = max(0, min(current_epoch, self.total_epochs))
        
        # Cosine decay formula
        progress = epoch / self.total_epochs
        cosine_term = 0.5 * (1 + math.cos(math.pi * progress))
        temperature = self.tau_min + (self.tau_max - self.tau_min) * cosine_term
        
        return temperature
    
    def get_progress_info(self, current_epoch):
        """
        Get detailed progress information for logging.
        
        Args:
            current_epoch: Current training epoch
            
        Returns:
            Dictionary with temperature schedule information
        """
        temperature = self.get_temperature(current_epoch)
        progress = min(current_epoch / self.total_epochs, 1.0)
        
        return {
            'current_temperature': temperature,
            'epoch': current_epoch,
            'total_epochs': self.total_epochs,
            'progress': progress,
            'tau_max': self.tau_max,
            'tau_min': self.tau_min,
            'decay_ratio': (self.tau_max - temperature) / (self.tau_max - self.tau_min)
        }


class NormSoftmaxLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()

        self.temperature = temperature

    def forward(self, x):
        "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
        i_logsm = F.log_softmax(x/self.temperature, dim=1)
        j_logsm = F.log_softmax(x.t()/self.temperature, dim=1)

        # sum over positives
        idiag = torch.diag(i_logsm)
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.diag(j_logsm)
        loss_j = jdiag.sum() / len(jdiag)

        return - loss_i - loss_j

class EgoNCE(nn.Module):
    def __init__(self, temperature=0.05, noun=True, verb=True, temperature_scheduler=None):
        super().__init__()
        self.noun = noun
        self.verb = verb
        self.temperature = temperature
        self.temperature_scheduler = temperature_scheduler
        self._current_epoch = 0  # Track current epoch for scheduling

    def set_epoch(self, epoch):
        """Set current epoch for temperature scheduling."""
        self._current_epoch = epoch

    def get_current_temperature(self):
        """Get current temperature based on scheduler or fixed value."""
        if self.temperature_scheduler is not None:
            return self.temperature_scheduler.get_temperature(self._current_epoch)
        else:
            return self.temperature

    def forward(self, x, mask_v, mask_n, current_epoch=None):
        """
        Forward pass with optional epoch-based temperature scheduling.
        
        Args:
            x: Similarity matrix
            mask_v: Verb mask
            mask_n: Noun mask  
            current_epoch: Optional current epoch for temperature scheduling
        """
        # Update epoch if provided
        if current_epoch is not None:
            self.set_epoch(current_epoch)
        
        # Get current temperature
        current_temp = self.get_current_temperature()
        
        mask_diag = torch.eye(x.shape[0]).to(x.device)
        if self.noun and self.verb:
            mask = mask_v * mask_n + mask_diag
        elif self.noun:
            mask = mask_n + mask_diag
        else:
            mask = mask_v + mask_diag

        # Use current temperature in softmax computation
        i_sm = F.softmax(x / current_temp, dim=1)
        j_sm = F.softmax(x.t() / current_temp, dim=1)

        mask_bool = mask > 0
        idiag = torch.log(torch.sum(i_sm * mask_bool, dim=1))
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.log(torch.sum(j_sm * mask_bool, dim=1) )
        loss_j = jdiag.sum() / len(jdiag)
        return - loss_i - loss_j

    def get_loss_info(self):
        """Get detailed loss information for logging."""
        info = {
            'current_temperature': self.get_current_temperature(),
            'base_temperature': self.temperature,
            'current_epoch': self._current_epoch,
            'using_scheduler': self.temperature_scheduler is not None
        }
        
        if self.temperature_scheduler is not None:
            scheduler_info = self.temperature_scheduler.get_progress_info(self._current_epoch)
            info.update(scheduler_info)
        
        return info


class EgoNCEWithScheduler(nn.Module):
    """
    Enhanced EgoNCE with built-in temperature scheduling and detailed logging.
    
    This class provides a convenient interface for EgoNCE with cosine temperature
    scheduling, including detailed logging and progress tracking.
    """
    
    def __init__(self, tau_max=0.07, tau_min=0.03, total_epochs=10, noun=True, verb=True):
        super().__init__()
        self.noun = noun
        self.verb = verb
        
        # Create temperature scheduler
        self.temperature_scheduler = TemperatureScheduler(tau_max, tau_min, total_epochs)
        
        # Create EgoNCE loss with scheduler
        self.egonce = EgoNCE(
            temperature=tau_max,  # Initial temperature
            noun=noun,
            verb=verb,
            temperature_scheduler=self.temperature_scheduler
        )
        
        self.current_epoch = 0
        
    def set_epoch(self, epoch):
        """Set current epoch for temperature scheduling."""
        self.current_epoch = epoch
        self.egonce.set_epoch(epoch)
        
    def forward(self, x, mask_v, mask_n, current_epoch=None):
        """
        Forward pass with automatic temperature scheduling.
        
        Returns:
            Tuple: (loss_value, loss_info_dict)
        """
        if current_epoch is not None:
            self.set_epoch(current_epoch)
            
        # Compute loss
        loss = self.egonce(x, mask_v, mask_n, current_epoch=self.current_epoch)
        
        # Get detailed information
        loss_info = self.egonce.get_loss_info()
        loss_info['loss_value'] = loss.item() if hasattr(loss, 'item') else float(loss)
        
        return loss, loss_info
    
    def get_temperature_schedule_preview(self, epochs=None):
        """
        Get preview of temperature schedule for visualization.
        
        Args:
            epochs: List of epochs to preview, or None for all epochs
            
        Returns:
            List of (epoch, temperature) tuples
        """
        if epochs is None:
            epochs = list(range(self.temperature_scheduler.total_epochs + 1))
            
        schedule = []
        for epoch in epochs:
            temp = self.temperature_scheduler.get_temperature(epoch)
            schedule.append((epoch, temp))
            
        return schedule


class MaxMarginRankingLoss(nn.Module):

    def __init__(self, margin=0.2, fix_norm=True):
        super().__init__()
        self.fix_norm = fix_norm
        self.loss = nn.MarginRankingLoss(margin)
        self.margin = margin

    def forward(self, x, weight=None):
        n = x.size()[0]

        x1 = torch.diag(x)
        x1 = x1.unsqueeze(1)
        x1 = x1.expand(n, n)
        x1 = x1.contiguous().view(-1, 1)
        x1 = torch.cat((x1, x1), 0)

        x2 = x.view(-1, 1)
        x3 = x.transpose(0, 1).contiguous().view(-1, 1)

        x2 = torch.cat((x2, x3), 0)
        max_margin = F.relu(self.margin - (x1 - x2))

        if self.fix_norm:
            # remove the elements from the diagonal
            keep = torch.ones(x.shape) - torch.eye(x.shape[0])  # 128 x 128
            keep1 = keep.view(-1, 1)
            keep2 = keep.transpose(0, 1).contiguous().view(-1, 1)
            keep_idx = torch.nonzero(torch.cat((keep1, keep2), 0).flatten()).flatten()
            if x1.is_cuda:
                keep_idx = keep_idx.cuda()
            x1_ = torch.index_select(x1, dim=0, index=keep_idx)
            x2_ = torch.index_select(x2, dim=0, index=keep_idx)
            max_margin = F.relu(self.margin - (x1_ - x2_))

        return max_margin.mean()

class AdaptiveMaxMarginRankingLoss(nn.Module):

    def __init__(self, margin=0.4, fix_norm=True):
        super().__init__()
        self.fix_norm = fix_norm
        self.loss = nn.MarginRankingLoss(margin)
        self.margin = margin

    def forward(self, x, weight=None):
        n = x.size()[0]

        x1 = torch.diag(x)
        x1 = x1.unsqueeze(1)
        x1 = x1.expand(n, n)
        x1 = x1.contiguous().view(-1, 1)
        x1 = torch.cat((x1, x1), 0)

        w1 = weight.unsqueeze(1)
        w1 = w1.expand(n, n)
        w1 = w1.contiguous().view(-1, 1)
        w1 = torch.cat((w1, w1), 0)

        x2 = x.view(-1, 1)
        x3 = x.transpose(0, 1).contiguous().view(-1, 1)

        x2 = torch.cat((x2, x3), 0)
        max_margin = F.relu(  w1 * self.margin - (x1 - x2))

        if self.fix_norm:
            # remove the elements from the diagonal
            keep = torch.ones(x.shape) - torch.eye(x.shape[0])  # 128 x 128
            keep1 = keep.view(-1, 1)
            keep2 = keep.transpose(0, 1).contiguous().view(-1, 1)
            keep_idx = torch.nonzero(torch.cat((keep1, keep2), 0).flatten()).flatten()
            if x1.is_cuda:
                keep_idx = keep_idx.cuda()
            x1_ = torch.index_select(x1, dim=0, index=keep_idx)
            w1_ = torch.index_select(w1, dim=0, index=keep_idx)
            x2_ = torch.index_select(x2, dim=0, index=keep_idx)
            max_margin =  F.relu( w1_ * self.margin - (x1_ - x2_))

        return max_margin.mean()

class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.loss(output, target)

class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency loss between adjacent clips at different scales.
    Encourages similar representations for overlapping temporal segments.
    """
    def __init__(self, temperature=0.1, loss_type='cosine'):
        super().__init__()
        self.temperature = temperature
        self.loss_type = loss_type
        
    def forward(self, scale_embeddings_1, scale_embeddings_2):
        """
        Compute temporal consistency loss between two sets of multi-scale embeddings.
        
        Args:
            scale_embeddings_1: List of embeddings from first clip at different scales
            scale_embeddings_2: List of embeddings from second clip at different scales
        
        Returns:
            consistency_loss: Scalar loss value
        """
        if len(scale_embeddings_1) != len(scale_embeddings_2):
            raise ValueError("Number of scales must match between clips")
        
        total_loss = 0.0
        num_scales = len(scale_embeddings_1)
        
        for i in range(num_scales):
            emb1 = scale_embeddings_1[i]  # [batch_size, projection_dim]
            emb2 = scale_embeddings_2[i]  # [batch_size, projection_dim]
            
            if self.loss_type == 'cosine':
                # Cosine similarity loss
                cosine_sim = F.cosine_similarity(emb1, emb2, dim=1)  # [batch_size]
                scale_loss = (1.0 - cosine_sim).mean()
                
            elif self.loss_type == 'mse':
                # Mean squared error loss
                scale_loss = F.mse_loss(emb1, emb2)
                
            elif self.loss_type == 'contrastive':
                # Contrastive loss - maximize similarity between corresponding clips
                # and minimize similarity with negative samples
                batch_size = emb1.shape[0]
                
                # Normalize embeddings
                emb1_norm = F.normalize(emb1, dim=1)
                emb2_norm = F.normalize(emb2, dim=1)
                
                # Positive pairs (diagonal)
                pos_sim = torch.sum(emb1_norm * emb2_norm, dim=1) / self.temperature  # [batch_size]
                
                # All pairs similarity matrix
                all_sim = torch.mm(emb1_norm, emb2_norm.t()) / self.temperature  # [batch_size, batch_size]
                
                # Contrastive loss
                pos_loss = -torch.log(torch.exp(pos_sim) / torch.sum(torch.exp(all_sim), dim=1))
                scale_loss = pos_loss.mean()
            
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
            
            total_loss += scale_loss
        
        return total_loss / num_scales

class CosineTemperatureScheduler:
    """
    Cosine temperature scheduler that adjusts temperature during training.
    Temperature starts high and decreases following a cosine schedule.
    """
    def __init__(self, initial_temp=0.1, final_temp=0.01, total_steps=10000):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.total_steps = total_steps
        
    def get_temperature(self, step):
        """
        Get temperature at given step.
        
        Args:
            step: Current training step
            
        Returns:
            temperature: Current temperature value
        """
        if step >= self.total_steps:
            return self.final_temp
        
        # Cosine decay
        progress = step / self.total_steps
        temperature = self.final_temp + 0.5 * (self.initial_temp - self.final_temp) * \
                     (1 + torch.cos(torch.tensor(progress * torch.pi)))
        
        return float(temperature)

class MultiScaleEgoNCE(nn.Module):
    """
    Enhanced EgoNCE loss with multi-scale temporal consistency.
    Combines original EgoNCE loss with temporal consistency regularization.
    """
    def __init__(self, temperature=0.05, noun=True, verb=True, 
                 consistency_weight=0.1, temp_scheduler=None):
        super().__init__()
        self.noun = noun
        self.verb = verb
        self.temperature = temperature
        self.consistency_weight = consistency_weight
        self.temp_scheduler = temp_scheduler
        
        # Original EgoNCE loss
        self.egonce_loss = EgoNCE(temperature=temperature, noun=noun, verb=verb)
        
        # Temporal consistency loss
        self.consistency_loss = TemporalConsistencyLoss(temperature=0.1, loss_type='contrastive')
        
    def forward(self, x, mask_v, mask_n, scale_embeddings_1=None, scale_embeddings_2=None, 
                step=None):
        """
        Forward pass combining EgoNCE and temporal consistency losses.
        
        Args:
            x: Similarity matrix between text and video embeddings
            mask_v: Verb mask for EgoNCE
            mask_n: Noun mask for EgoNCE
            scale_embeddings_1: Multi-scale embeddings for first set of clips
            scale_embeddings_2: Multi-scale embeddings for second set of clips (adjacent)
            step: Current training step for temperature scheduling
            
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        # Get current temperature if scheduler is provided
        current_temp = self.temperature
        if self.temp_scheduler is not None and step is not None:
            current_temp = self.temp_scheduler.get_temperature(step)
            # Update EgoNCE temperature
            self.egonce_loss.temperature = current_temp
        
        # Original EgoNCE loss
        egonce_loss = self.egonce_loss(x, mask_v, mask_n)
        
        # Temporal consistency loss
        consistency_loss = 0.0
        if scale_embeddings_1 is not None and scale_embeddings_2 is not None:
            consistency_loss = self.consistency_loss(scale_embeddings_1, scale_embeddings_2)
        
        # Combined loss
        total_loss = egonce_loss + self.consistency_weight * consistency_loss
        
        loss_dict = {
            'egonce_loss': egonce_loss,
            'consistency_loss': consistency_loss,
            'total_loss': total_loss,
            'temperature': current_temp
        }
        
        return total_loss, loss_dict