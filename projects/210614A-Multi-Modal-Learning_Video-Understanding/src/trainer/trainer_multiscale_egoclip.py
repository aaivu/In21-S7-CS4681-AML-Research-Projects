import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm
import torch.distributed as dist

from base import Multi_BaseTrainer_dist
from model.model import sim_matrix
from utils import inf_loop
from trainer.trainer_egoclip import AllGather_multi, Multi_Trainer_dist


class MultiScale_Trainer_dist(Multi_Trainer_dist):
    """
    Enhanced trainer for multi-scale EgoVLP model with temporal consistency loss
    and temperature scheduling.
    
    Extends the base Multi_Trainer_dist to handle:
    - Multi-scale video features
    - Temporal consistency loss between adjacent clips
    - Dynamic temperature scheduling
    """

    def __init__(self, args, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, writer=None,
                 visualizer=None, tokenizer=None, max_samples_per_epoch=50000, 
                 temp_scheduler=None):
        super().__init__(args, model, loss, metrics, optimizer, config, data_loader,
                        valid_data_loader, lr_scheduler, len_epoch, writer,
                        visualizer, tokenizer, max_samples_per_epoch)
        
        self.temp_scheduler = temp_scheduler
        self.consistency_loss_weight = getattr(config['loss']['args'], 'consistency_weight', 0.1)
        self.global_step = 0
        
    def _train_epoch(self, epoch):
        """
        Enhanced training epoch with multi-scale processing and temperature scheduling.
        """
        self.model.train()
        total_loss = [0] * len(self.data_loader)
        total_metrics = [np.zeros(len(self.metrics))] * len(self.data_loader)
        
        # Loss tracking
        egonce_losses = []
        consistency_losses = []
        temperatures = []

        for loader_idx, data_loader in enumerate(self.data_loader):
            for batch_idx, data in enumerate(tqdm(data_loader)):
                if (self.tokenizer is not None) and ('text' in data.keys()):
                    data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True,
                                                truncation=True)
                if (self.tokenizer is not None) and ('text_neg' in data.keys()):
                    data['text_neg'] = self.tokenizer(data['text_neg'], return_tensors='pt', padding=True,
                                                    truncation=True)

                data = self._move_to_device(data)
                
                self.optimizer.zero_grad()
                
                # Multi-scale forward pass
                if hasattr(self.model, 'module'):
                    model = self.model.module
                else:
                    model = self.model
                
                # Forward pass with multi-scale features
                if isinstance(model, model.__class__.__bases__[0]):  # Check if MultiScaleFrozenInTime
                    text_embeds, video_embeds, scale_embeds = model(data, return_multi_scale=True)
                else:
                    # Fallback to regular forward pass
                    text_embeds, video_embeds = model(data)
                    scale_embeds = None

                # Apply distributed gathering
                if self.args.world_size > 1:
                    text_embeds = AllGather_multi.apply(text_embeds, self.n_gpu, self.args)
                    video_embeds = AllGather_multi.apply(video_embeds, self.n_gpu, self.args)
                    if scale_embeds is not None:
                        scale_embeds = [AllGather_multi.apply(emb, self.n_gpu, self.args) 
                                      for emb in scale_embeds]

                # Compute similarity matrix
                output = sim_matrix(text_embeds, video_embeds)
                
                # Prepare masks for EgoNCE loss
                if 'noun_vec' in data and 'verb_vec' in data:
                    noun_mask = data['noun_vec']
                    verb_mask = data['verb_vec']
                    
                    if self.args.world_size > 1:
                        noun_mask = AllGather_multi.apply(noun_mask, self.n_gpu, self.args)
                        verb_mask = AllGather_multi.apply(verb_mask, self.n_gpu, self.args)
                else:
                    # Create identity masks if not available
                    batch_size = output.shape[0]
                    noun_mask = torch.eye(batch_size).to(output.device)
                    verb_mask = torch.eye(batch_size).to(output.device)

                # Compute loss
                if hasattr(self.loss, 'forward') and len(self.loss.forward.__code__.co_varnames) > 5:
                    # Enhanced loss with temporal consistency
                    loss_result, loss_dict = self.loss(
                        output, verb_mask, noun_mask,
                        scale_embeddings_1=scale_embeds,
                        scale_embeddings_2=scale_embeds,  # For now, use same clip (can extend to adjacent clips)
                        step=self.global_step
                    )
                    
                    # Track individual loss components
                    if isinstance(loss_dict, dict):
                        egonce_losses.append(loss_dict.get('egonce_loss', 0))
                        consistency_losses.append(loss_dict.get('consistency_loss', 0))
                        temperatures.append(loss_dict.get('temperature', 0.05))
                        
                        # Log to writer
                        if self.writer is not None and self.args.rank == 0:
                            self.writer.add_scalar('Loss/EgoNCE', loss_dict.get('egonce_loss', 0), self.global_step)
                            self.writer.add_scalar('Loss/Consistency', loss_dict.get('consistency_loss', 0), self.global_step)
                            self.writer.add_scalar('Loss/Temperature', loss_dict.get('temperature', 0.05), self.global_step)
                else:
                    # Fallback to original loss
                    loss_result = self.loss(output, verb_mask, noun_mask)

                loss_result.backward()

                self.optimizer.step()
                
                if self.writer is not None and self.args.rank == 0:
                    self.writer.add_scalar('Loss/Total', loss_result.item(), self.global_step)

                total_loss[loader_idx] += loss_result.item()
                
                # Compute metrics
                for i, metric in enumerate(self.metrics):
                    if 'accuracy' in metric.__name__:
                        total_metrics[loader_idx][i] += metric(output, noun_mask, verb_mask)
                    else:
                        total_metrics[loader_idx][i] += metric(output)
                
                self.global_step += 1

                if batch_idx % self.log_step == 0 and self.args.rank == 0:
                    self.logger.info('Train Epoch: {} dl{} {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                        epoch,
                        loader_idx,
                        self._progress(batch_idx, loader_idx),
                        batch_idx * self.data_loader[loader_idx].batch_size,
                        self.data_loader[loader_idx].n_samples,
                        100.0 * batch_idx / len(self.data_loader[loader_idx]),
                        loss_result.item()))

                if batch_idx == self.len_epoch:
                    break

        # Log epoch summaries
        if self.args.rank == 0:
            log_dict = {}
            for loader_idx in range(len(self.data_loader)):
                log_dict.update({
                    f'loss_{loader_idx}': total_loss[loader_idx] / self.len_epoch,
                })
                for i, metric in enumerate(self.metrics):
                    log_dict.update({
                        f'{metric.__name__}_{loader_idx}': total_metrics[loader_idx][i] / self.len_epoch,
                    })
            
            # Add multi-scale specific logs
            if egonce_losses:
                log_dict['avg_egonce_loss'] = np.mean(egonce_losses)
            if consistency_losses:
                log_dict['avg_consistency_loss'] = np.mean(consistency_losses)
            if temperatures:
                log_dict['avg_temperature'] = np.mean(temperatures)
            
            if self.writer is not None:
                for key, value in log_dict.items():
                    self.writer.add_scalar(f'Epoch/{key}', value, epoch)
            
            self.logger.info(f'Epoch {epoch} Summary: ' + 
                           ', '.join([f'{k}: {v:.6f}' for k, v in log_dict.items()]))

        log_dict = {}
        for loader_idx in range(len(self.data_loader)):
            log_dict.update({
                f'loss_{loader_idx}': total_loss[loader_idx] / self.len_epoch,
            })
            for i, metric in enumerate(self.metrics):
                log_dict.update({
                    f'{metric.__name__}_{loader_idx}': total_metrics[loader_idx][i] / self.len_epoch,
                })

        return log_dict

    def _move_to_device(self, data):
        """
        Move data to device, handling multi-scale video data structure.
        """
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                if key == 'video' and isinstance(value, dict):
                    # Handle multi-scale video data
                    result[key] = {k: v.to(self.device) for k, v in value.items()}
                elif isinstance(value, torch.Tensor):
                    result[key] = value.to(self.device)
                elif isinstance(value, dict):
                    result[key] = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                 for k, v in value.items()}
                else:
                    result[key] = value
            return result
        elif isinstance(data, torch.Tensor):
            return data.to(self.device)
        else:
            return data

    def _valid_epoch(self, epoch):
        """
        Enhanced validation epoch for multi-scale model.
        """
        self.model.eval()
        total_val_loss = [0] * len(self.valid_data_loader)
        total_val_metrics = [np.zeros(len(self.metrics))] * len(self.valid_data_loader)
        
        with torch.no_grad():
            for loader_idx, valid_data_loader in enumerate(self.valid_data_loader):
                for batch_idx, data in enumerate(tqdm(valid_data_loader)):
                    if (self.tokenizer is not None) and ('text' in data.keys()):
                        data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True,
                                                    truncation=True)
                    
                    data = self._move_to_device(data)
                    
                    # Forward pass
                    if hasattr(self.model, 'module'):
                        model = self.model.module
                    else:
                        model = self.model
                    
                    if hasattr(model, 'compute_video') and 'video' in data and isinstance(data['video'], dict):
                        # Multi-scale model
                        text_embeds, video_embeds = model(data, return_embeds=True)
                    else:
                        # Regular model
                        text_embeds, video_embeds = model(data)
                    
                    output = sim_matrix(text_embeds, video_embeds)
                    
                    # Create dummy masks for validation
                    batch_size = output.shape[0]
                    noun_mask = torch.eye(batch_size).to(output.device)
                    verb_mask = torch.eye(batch_size).to(output.device)
                    
                    if hasattr(self.loss, 'egonce_loss'):
                        # Use just the EgoNCE component for validation
                        loss = self.loss.egonce_loss(output, verb_mask, noun_mask)
                    else:
                        loss = self.loss(output, verb_mask, noun_mask)
                    
                    total_val_loss[loader_idx] += loss.item()
                    
                    for i, metric in enumerate(self.metrics):
                        if 'accuracy' in metric.__name__:
                            total_val_metrics[loader_idx][i] += metric(output, noun_mask, verb_mask)
                        else:
                            total_val_metrics[loader_idx][i] += metric(output)

        # Return validation log
        val_log = {}
        for loader_idx in range(len(self.valid_data_loader)):
            val_log.update({
                f'val_loss_{loader_idx}': total_val_loss[loader_idx] / len(self.valid_data_loader[loader_idx]),
            })
            for i, metric in enumerate(self.metrics):
                val_log.update({
                    f'val_{metric.__name__}_{loader_idx}': 
                    total_val_metrics[loader_idx][i] / len(self.valid_data_loader[loader_idx]),
                })

        return val_log