import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm
import torch.distributed as dist
import time

from base import Multi_BaseTrainer_dist
from model.model import sim_matrix
from utils import inf_loop
from logger.comprehensive_logger import create_comprehensive_logger

# Import enhanced components
from model.model import MultiScaleVideoEncoder
from model.loss import EgoNCEWithScheduler
from model.temporal_loss import TemporalConsistencyLoss


class AllGather_multi(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, n_gpu, args):
        output = [torch.empty_like(tensor) for _ in range(args.world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = args.rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None, None,
        )


class Enhanced_Multi_Trainer_dist(Multi_BaseTrainer_dist):
    """
    Enhanced Trainer class with multi-scale video processing, temporal consistency,
    and temperature scheduling.
    
    Note:
        Inherited from Multi_BaseTrainer_dist with enhancements.
    """

    def __init__(self, args, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, writer=None,
                 visualizer=None, tokenizer=None, max_samples_per_epoch=50000):
        super().__init__(args, model, loss, metrics, optimizer, config, writer)
        self.config = config
        self.args = args
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            # take the min
            self.len_epoch = min([len(x) for x in data_loader])
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(self.data_loader[0].batch_size))
        self.visualizer = visualizer
        self.tokenizer = tokenizer
        self.max_samples_per_epoch = max_samples_per_epoch

        self.n_gpu = args.world_size
        self.allgather = AllGather_multi.apply
        
        # Enhanced loss components
        self.setup_enhanced_losses()
        
        # Replace video encoder with multi-scale version
        self.setup_multiscale_encoder()
        
        # Comprehensive logging setup
        self.comprehensive_logger = create_comprehensive_logger(
            writer=writer,
            logger=self.logger,
            log_dir=str(config.log_dir) if hasattr(config, 'log_dir') else None,
            rank=args.rank
        )
        
        # Training metrics tracking
        self.epoch_losses = []
        self.epoch_temporal_losses = []
        self.epoch_temperatures = []
        self.samples_processed_epoch = 0

    def setup_enhanced_losses(self):
        """Setup enhanced loss functions with temperature scheduling and temporal consistency."""
        total_epochs = self.config.get('trainer', {}).get('epochs', 20)
        
        # Enhanced EgoNCE with temperature scheduling
        if self.config['loss']['type'] == 'EgoNCE':
            self.enhanced_egonce = EgoNCEWithScheduler(
                tau_max=self.config['loss'].get('tau_max', 0.07),
                tau_min=self.config['loss'].get('tau_min', 0.03),
                total_epochs=total_epochs,
                noun=True,
                verb=True
            )
            self.logger.info(f"✓ Enhanced EgoNCE with temperature scheduling: {0.07} → {0.03}")
        
        # Temporal consistency loss
        temporal_lambda = self.config.get('temporal_lambda', 0.1)
        self.temporal_loss = TemporalConsistencyLoss(lambda_temp=temporal_lambda)
        self.logger.info(f"✓ Temporal consistency loss with λ = {temporal_lambda}")

    def setup_multiscale_encoder(self):
        """Replace the video encoder with multi-scale version."""
        try:
            if hasattr(self.model, 'video_encoder'):
                # Store original encoder for initialization
                original_encoder = self.model.video_encoder
                
                # Create multi-scale encoder
                scales = self.config.get('multiscale_scales', [4, 8, 16])
                self.multiscale_encoder = MultiScaleVideoEncoder(
                    scales=scales,
                    base_encoder=original_encoder,
                    fusion_type='weighted',
                    device=self.device
                )
                
                # Replace in model
                self.model.video_encoder = self.multiscale_encoder
                
                self.logger.info(f"✓ Multi-scale video encoder initialized with scales: {scales}")
            else:
                self.logger.warning("⚠ Model doesn't have video_encoder attribute, skipping multi-scale setup")
                self.multiscale_encoder = None
                
        except Exception as e:
            self.logger.error(f"✗ Failed to setup multi-scale encoder: {e}")
            self.multiscale_encoder = None

    def _train_epoch(self, epoch):
        """
        Enhanced training epoch with comprehensive logging for multi-scale processing, 
        temporal consistency, and temperature scheduling.
        """
        self.model.train()
        
        # Start epoch timing and GPU monitoring
        self.comprehensive_logger.start_epoch_timing()
        
        # Initialize epoch metrics
        total_loss = [0] * len(self.data_loader)
        total_contrastive_loss = [0] * len(self.data_loader)
        total_temporal_loss = [0] * len(self.data_loader)
        self.samples_processed_epoch = 0
        
        # Get current temperature for this epoch
        current_temp = 0.05  # Default
        if hasattr(self, 'enhanced_egonce'):
            current_temp = self.enhanced_egonce.temperature_scheduler.get_temperature(epoch)
            self.logger.info(f"Epoch {epoch} - Current temperature: {current_temp:.6f}")
        
        for dl_idx, data_loader in enumerate(self.data_loader):
            pbar = tqdm(enumerate(data_loader), disable=self.args.rank != 0)
            for batch_idx, data in pbar:
                
                # Enhanced data processing - unpack multi-scale frames
                data = self.process_enhanced_batch(data)
                
                # Move data to device
                if 'video_neg' in data.keys():  # w/ negative sampling
                    data['text'] = data['text'] + data['text_neg']
                    data['video'] = torch.cat((data['video'], data['video_neg']), axis=0)
                
                # Tokenize text
                if isinstance(data['text'], list):
                    data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, 
                                                 truncation=True)
                data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
                data['video'] = data['video'].to(self.device)
                
                # Get embeddings for loss computation
                n_embeds = data['noun_vec'].to(self.device)
                v_embeds = data['verb_vec'].to(self.device)

                self.optimizer.zero_grad()
                
                with torch.set_grad_enabled(True):
                    # Forward pass with multi-scale processing
                    text_embeds, video_embeds = self.model(data)
                    
                    # Distributed gathering
                    video_embeds = self.allgather(video_embeds, self.n_gpu, self.args)
                    text_embeds = self.allgather(text_embeds, self.n_gpu, self.args)
                    n_embeds = self.allgather(n_embeds, self.n_gpu, self.args)
                    v_embeds = self.allgather(v_embeds, self.n_gpu, self.args)
                    
                    # Compute similarity matrices
                    output = sim_matrix(text_embeds, video_embeds)
                    
                    # Enhanced contrastive loss with temperature scheduling
                    contrastive_loss = 0
                    egonce_v2t = 0
                    egonce_t2v = 0
                    loss_info = {'current_temperature': current_temp}
                    
                    if self.config['loss']['type'] == 'EgoNCE' and hasattr(self, 'enhanced_egonce'):
                        sim_v = sim_matrix(v_embeds, v_embeds)
                        sim_n = sim_matrix(n_embeds, n_embeds)
                        contrastive_loss, loss_info = self.enhanced_egonce(
                            output, sim_v, sim_n, current_epoch=epoch
                        )
                        
                        # Extract V2T and T2V components if available
                        if isinstance(loss_info, dict):
                            egonce_v2t = loss_info.get('v2t_loss', contrastive_loss / 2)
                            egonce_t2v = loss_info.get('t2v_loss', contrastive_loss / 2)
                    else:
                        contrastive_loss = self.loss(output)
                        egonce_v2t = contrastive_loss / 2  # Approximate split
                        egonce_t2v = contrastive_loss / 2
                    
                    # Temporal consistency loss
                    temporal_loss_val = 0
                    temporal_pairs = data.get('temporal_pairs', None)
                    if temporal_pairs is not None and len(temporal_pairs) > 0:
                        temporal_loss_val = self.temporal_loss(video_embeds, temporal_pairs)
                    
                    # Get lambda weight for temporal loss
                    lambda_weight = self.temporal_loss.lambda_temp if hasattr(self.temporal_loss, 'lambda_temp') else 0.1
                    
                    # Combine losses
                    total_batch_loss = contrastive_loss + temporal_loss_val
                
                # Backward pass
                total_batch_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.optimizer.step()
                
                # Sample GPU memory after forward/backward pass
                self.comprehensive_logger.gpu_memory_tracker.sample()

                # Comprehensive iteration logging
                batch_size = getattr(self.data_loader[dl_idx], 'batch_size', 32)
                self.samples_processed_epoch += batch_size
                
                losses_dict = {
                    'total_loss': total_batch_loss.detach().item(),
                    'egonce_loss': contrastive_loss.detach().item(),
                    'egonce_v2t': egonce_v2t.detach().item() if isinstance(egonce_v2t, torch.Tensor) else egonce_v2t,
                    'egonce_t2v': egonce_t2v.detach().item() if isinstance(egonce_t2v, torch.Tensor) else egonce_t2v,
                    'temporal_loss': temporal_loss_val.detach().item() if isinstance(temporal_loss_val, torch.Tensor) else temporal_loss_val,
                    'lambda_weight': lambda_weight,
                    'temperature': loss_info['current_temperature']
                }
                
                # Log with gradient analysis every 100 iterations
                log_gradients = (batch_idx % 100 == 0)
                self.comprehensive_logger.log_iteration_losses(
                    epoch=epoch, 
                    batch_idx=batch_idx, 
                    dl_idx=dl_idx, 
                    losses_dict=losses_dict,
                    model=self.model,
                    log_gradients=log_gradients
                )

                # Update totals
                total_loss[dl_idx] += total_batch_loss.detach().item()
                total_contrastive_loss[dl_idx] += contrastive_loss.detach().item()
                if isinstance(temporal_loss_val, torch.Tensor):
                    total_temporal_loss[dl_idx] += temporal_loss_val.detach().item()

                self.optimizer.zero_grad()
                
                if batch_idx == self.len_epoch:
                    break

        # Epoch summary logging
        log = {}
        for dl_idx in range(len(self.data_loader)):
            avg_total = total_loss[dl_idx] / self.len_epoch
            avg_contrastive = total_contrastive_loss[dl_idx] / self.len_epoch  
            avg_temporal = total_temporal_loss[dl_idx] / self.len_epoch
            
            log[f'loss_{dl_idx}'] = avg_total
            log[f'contrastive_loss_{dl_idx}'] = avg_contrastive
            log[f'temporal_loss_{dl_idx}'] = avg_temporal
            
            # Store epoch metrics
            self.epoch_losses.append(avg_total)
            self.epoch_temporal_losses.append(avg_temporal)
            if hasattr(self, 'enhanced_egonce'):
                self.epoch_temperatures.append(current_temp)

        # Comprehensive epoch-level logging
        epoch_metrics = {
            'egomcq_intra_accuracy': 0.0,  # Will be updated from validation
            'egomcq_inter_accuracy': 0.0,  # Will be updated from validation
            'temperature': current_temp,
            'samples_processed': self.samples_processed_epoch
        }
        
        self.comprehensive_logger.log_epoch_metrics(
            epoch=epoch,
            metrics_dict=epoch_metrics,
            model=self.model
        )
        
        # Save detailed logs
        self.comprehensive_logger.save_detailed_logs(epoch)

        # Validation with EgoMCQ metrics logging
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            if self.args.rank == 0:
                log.update(val_log)
                
                # Update epoch metrics with validation results
                if 'egomcq_intra_accuracy' in val_log:
                    epoch_metrics['egomcq_intra_accuracy'] = val_log['egomcq_intra_accuracy']
                if 'egomcq_inter_accuracy' in val_log:
                    epoch_metrics['egomcq_inter_accuracy'] = val_log['egomcq_inter_accuracy']
                
                # Re-log epoch metrics with validation results
                self.comprehensive_logger.log_epoch_metrics(
                    epoch=epoch,
                    metrics_dict=epoch_metrics,
                    model=self.model
                )

        # Learning rate scheduling
        self._adjust_learning_rate(self.optimizer, epoch, self.args)

        return log

    def process_enhanced_batch(self, data):
        """Process batch to handle multi-scale video data."""
        # Check if multi-scale frames are available
        if 'frames_4' in data and 'frames_8' in data and 'frames_16' in data:
            # Multi-scale data available - combine into single video tensor
            # The MultiScaleVideoEncoder will handle the different scales
            data['video'] = data.get('video', data['frames_16'])  # Use 16-frame as default
            data['multiscale_frames'] = {
                'frames_4': data['frames_4'],
                'frames_8': data['frames_8'], 
                'frames_16': data['frames_16']
            }
        
        return data

    def _progress(self, batch_idx, dl_idx):
        """Format training progress string."""
        try:
            total = len(self.data_loader[dl_idx])
            current = batch_idx
            return f'[{current:3d}/{total:3d}] ({100.0 * current / total:.0f}%)'
        except:
            return f'[{batch_idx:3d}/?]'

    def _valid_epoch(self, epoch):
        """
        Enhanced validation with EgoMCQ accuracy computation and comprehensive logging.
        """
        self.model.eval()
        total_val_loss = [0] * len(self.valid_data_loader)
        
        # EgoMCQ accuracy tracking
        all_preds = []
        all_labels = []
        all_types = []

        with torch.no_grad():
            for dl_idx, data_loader in enumerate(self.valid_data_loader):
                for batch_idx, data in enumerate(data_loader):
                    # Process data
                    if isinstance(data['text'], list):
                        data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, 
                                                     truncation=True)
                    data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
                    data['video'] = data['video'].to(self.device)

                    # Forward pass
                    text_embeds, video_embeds = self.model(data)
                    output = sim_matrix(text_embeds, video_embeds)
                    
                    # Compute loss
                    if self.config['loss']['type'] == 'EgoNCE':
                        n_embeds = data['noun_vec'].to(self.device)
                        v_embeds = data['verb_vec'].to(self.device)
                        sim_v = sim_matrix(v_embeds, v_embeds)
                        sim_n = sim_matrix(n_embeds, n_embeds)
                        loss = self.loss(output, sim_v, sim_n)
                    else:
                        loss = self.loss(output)

                    total_val_loss[dl_idx] += loss.detach().item()
                    
                    # Collect EgoMCQ data if available
                    if 'answer' in data and 'type' in data:
                        # Get predictions from similarity matrix
                        predictions = torch.softmax(output, dim=-1)
                        all_preds.append(predictions.cpu())
                        all_labels.append(data['answer'].cpu())
                        all_types.append(data['type'].cpu())

        # Calculate EgoMCQ accuracy metrics
        egomcq_intra = 0.0
        egomcq_inter = 0.0
        
        if all_preds and all_labels and all_types:
            try:
                from model.metric import egomcq_accuracy_metrics
                
                # Concatenate all predictions
                preds_tensor = torch.cat(all_preds, dim=0)
                labels_tensor = torch.cat(all_labels, dim=0)
                types_tensor = torch.cat(all_types, dim=0)
                
                # Calculate EgoMCQ metrics
                egomcq_metrics = egomcq_accuracy_metrics(preds_tensor, labels_tensor, types_tensor)
                egomcq_intra = egomcq_metrics.get('Intra-video', 0.0)
                egomcq_inter = egomcq_metrics.get('Inter-video', 0.0)
                
                # Log EgoMCQ metrics
                if self.args.rank == 0 and self.logger:
                    self.logger.info(f'Validation Epoch {epoch} - EgoMCQ Accuracy: '
                                   f'Intra-video: {egomcq_intra:.2f}%, Inter-video: {egomcq_inter:.2f}%')
                
            except Exception as e:
                if self.args.rank == 0 and self.logger:
                    self.logger.warning(f"Could not calculate EgoMCQ metrics: {e}")

        # Prepare validation log
        log = {f'val_loss_{dl_idx}': total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx]) 
               for dl_idx in range(len(self.valid_data_loader))}
        
        # Add EgoMCQ metrics to log
        log['egomcq_intra_accuracy'] = egomcq_intra
        log['egomcq_inter_accuracy'] = egomcq_inter
        log['egomcq_avg_accuracy'] = (egomcq_intra + egomcq_inter) / 2
        
        # TensorBoard logging for validation
        if self.writer and self.args.rank == 0:
            for dl_idx in range(len(self.valid_data_loader)):
                self.writer.add_scalar(f'Val_loss/loss_{dl_idx}', log[f'val_loss_{dl_idx}'], epoch)
            
            self.writer.add_scalar('Val_metrics/egomcq_intra_accuracy', egomcq_intra, epoch)
            self.writer.add_scalar('Val_metrics/egomcq_inter_accuracy', egomcq_inter, epoch)
            self.writer.add_scalar('Val_metrics/egomcq_avg_accuracy', log['egomcq_avg_accuracy'], epoch)
        
        return log

    def _adjust_learning_rate(self, optimizer, epoch, args):
        """Adjust learning rate (using original logic)"""
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()