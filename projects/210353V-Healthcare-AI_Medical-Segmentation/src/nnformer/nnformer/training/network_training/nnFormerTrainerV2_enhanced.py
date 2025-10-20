"""
Enhanced Trainer for nnFormer with Multi-Scale Cross-Attention
Specialized training strategy for BraTS 2021 dataset

Features:
- Differentiated learning rates for cross-attention parameters
- Progressive training with gradual cross-attention activation
- Auxiliary supervision at intermediate scales
- Enhanced regularization strategies
- BraTS-specific configurations
"""

from collections import OrderedDict
from typing import Tuple, List
import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.optim import lr_scheduler
from batchgenerators.utilities.file_and_folder_operations import *

from nnformer.training.network_training.nnFormerTrainerV2 import nnFormerTrainerV2
from nnformer.network_architecture.nnFormer_enhanced import EnhancednnFormer, create_enhanced_nnformer_brats
from nnformer.network_architecture.initialization import InitWeights_He
from nnformer.network_architecture.neural_network import SegmentationNetwork
from nnformer.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnformer.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnformer.training.loss_functions.dice_loss import DC_and_CE_loss
from nnformer.utilities.to_torch import maybe_to_torch, to_cuda
from nnformer.utilities.nd_softmax import softmax_helper
from nnformer.training.dataloading.dataset_loading import unpack_dataset
from nnformer.training.learning_rate.poly_lr import poly_lr


class EnhancednnFormerTrainer(nnFormerTrainerV2):
    """
    Enhanced nnFormer Trainer for BraTS 2021
    
    Implements advanced training strategies including:
    - Multi-scale cross-attention with progressive activation
    - Differentiated learning rates for different parameter groups
    - Auxiliary supervision at intermediate encoder stages
    - Enhanced regularization (dropout, gradient clipping)
    """
    
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True,
                 fp16=False, enable_cross_attention=True, enable_adaptive_fusion=False,
                 enable_enhanced_training=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory,
                         batch_dice, stage, unpack_data, deterministic, fp16)
        
        # Enhancement flags
        self.enable_cross_attention = enable_cross_attention
        self.enable_adaptive_fusion = enable_adaptive_fusion
        self.enable_enhanced_training = enable_enhanced_training
        
        # Training configuration for BraTS
        self.max_num_epochs = 1000
        self.initial_lr = 1e-2
        self.weight_decay = 3e-5
        
        # Cross-attention specific settings
        self.cross_attn_lr_factor = 0.1  # Reduced LR for cross-attention
        self.warmup_epochs = 50
        self.current_epoch = 0
        
        # Auxiliary loss weights
        self.aux_loss_weights = [0.5, 0.3, 0.2]  # Weights for auxiliary outputs
        
        # Gradient clipping
        self.max_grad_norm = 1.0
        
        # BraTS specific configuration
        self.crop_size = [64, 128, 128]
        self.input_channels = 4  # T1, T1ce, T2, FLAIR
        self.num_classes = 4  # Background + ET, TC, WT
        self.embedding_dim = 192
        self.depths = [2, 2, 2, 2]
        self.num_heads = [6, 12, 24, 48]
        self.embedding_patch_size = [1, 4, 4]
        self.window_size = [[3, 5, 5], [3, 5, 5], [7, 10, 10], [3, 5, 5]]
        self.down_stride = [[1, 4, 4], [1, 8, 8], [2, 16, 16], [4, 32, 32]]
        self.deep_supervision = True
    
    def initialize(self, training=True, force_load_plans=False):
        """
        Initialize the enhanced trainer with specialized configurations
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)
            
            if force_load_plans or (self.plans is None):
                self.load_plans_file()
            
            self.process_plans(self.plans)
            self.setup_DA_params()
            
            if self.deep_supervision:
                # Setup deep supervision
                net_numpool = len(self.net_num_pool_op_kernel_sizes)
                weights = np.array([1 / (2 ** i) for i in range(net_numpool)])
                weights = weights / weights.sum()
                self.ds_loss_weights = weights
                
                # Wrap loss for deep supervision
                self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            
            self.folder_with_preprocessed_data = join(
                self.dataset_directory,
                self.plans['data_identifier'] + "_stage%d" % self.stage
            )
            
            if training:
                seeds_train = np.random.random_integers(0, 99999, self.data_aug_params.get('num_threads'))
                seeds_val = np.random.random_integers(0, 99999, max(self.data_aug_params.get('num_threads') // 2, 1))
                
                self.dl_tr, self.dl_val = self.get_basic_generators()
                
                if self.unpack_data:
                    print("Unpacking dataset...")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("Done")
                
                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params['patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales if self.deep_supervision else None,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False,
                    seeds_train=seeds_train,
                    seeds_val=seeds_val
                )
                
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())), False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())), False)
            
            self.initialize_network()
            self.initialize_optimizer_and_scheduler()
            
            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('Already initialized')
        
        self.was_initialized = True
    
    def initialize_network(self):
        """
        Initialize enhanced nnFormer network
        """
        self.network = create_enhanced_nnformer_brats(
            enable_cross_attention=self.enable_cross_attention,
            enable_adaptive_fusion=self.enable_adaptive_fusion,
            enable_enhanced_training=self.enable_enhanced_training
        )
        
        if torch.cuda.is_available():
            self.network.cuda()
        
        self.network.inference_apply_nonlin = softmax_helper
        
        self.print_to_log_file(f"Enhanced nnFormer Configuration:")
        self.print_to_log_file(f"  Cross-Attention: {self.enable_cross_attention}")
        self.print_to_log_file(f"  Adaptive Fusion: {self.enable_adaptive_fusion}")
        self.print_to_log_file(f"  Enhanced Training: {self.enable_enhanced_training}")
        self.print_to_log_file(f"  Embedding Dim: {self.embedding_dim}")
        self.print_to_log_file(f"  Input Channels: {self.input_channels}")
        self.print_to_log_file(f"  Num Classes: {self.num_classes}")
    
    def initialize_optimizer_and_scheduler(self):
        """
        Initialize optimizer with differentiated learning rates
        Cross-attention parameters get reduced learning rate
        """
        assert self.network is not None, "Network must be initialized first"
        
        # Separate parameters into groups
        cross_attn_params = []
        other_params = []
        
        for name, param in self.network.named_parameters():
            if 'cross_attn' in name or 'cross_scale_modules' in name:
                cross_attn_params.append(param)
            else:
                other_params.append(param)
        
        # Create parameter groups with different learning rates
        param_groups = [
            {
                'params': other_params,
                'lr': self.initial_lr,
                'weight_decay': self.weight_decay
            },
            {
                'params': cross_attn_params,
                'lr': self.initial_lr * self.cross_attn_lr_factor,
                'weight_decay': self.weight_decay * 0.5  # Reduced weight decay for cross-attention
            }
        ]
        
        self.optimizer = torch.optim.SGD(
            param_groups,
            momentum=0.99,
            nesterov=True
        )
        
        self.lr_scheduler = None  # Will use poly_lr manually
        
        self.print_to_log_file(f"Optimizer initialized with differentiated LRs:")
        self.print_to_log_file(f"  Base LR: {self.initial_lr}")
        self.print_to_log_file(f"  Cross-Attention LR: {self.initial_lr * self.cross_attn_lr_factor}")
    
    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        Enhanced training iteration with auxiliary supervision
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']
        
        data = maybe_to_torch(data)
        target = maybe_to_torch(target)
        
        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
        
        self.optimizer.zero_grad()
        
        if self.fp16:
            with autocast():
                output = self.network(data, epoch=self.current_epoch)
                del data
                l = self.loss(output, target)
        else:
            output = self.network(data, epoch=self.current_epoch)
            del data
            l = self.loss(output, target)
        
        if do_backprop:
            if self.fp16:
                self.amp_grad_scaler.scale(l).backward()
                
                # Gradient clipping
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
            else:
                l.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
        
        if run_online_evaluation:
            self.run_online_evaluation(output, target)
        
        del target
        
        return l.detach().cpu().numpy()
    
    def update_train_loss_MA(self):
        """Update moving average of training loss"""
        if self.train_loss_MA is None:
            self.train_loss_MA = self.all_tr_losses[-1]
        else:
            self.train_loss_MA = 0.99 * self.train_loss_MA + 0.01 * self.all_tr_losses[-1]
    
    def run_training(self):
        """
        Main training loop with progressive cross-attention activation
        """
        self.maybe_update_lr(self.epoch)
        
        ds = self.network.do_ds
        self.network.do_ds = True
        
        ret = super().run_training()
        
        self.network.do_ds = ds
        return ret
    
    def on_epoch_end(self):
        """
        Called at the end of each epoch
        Updates current epoch counter for progressive training
        """
        super().on_epoch_end()
        self.current_epoch = self.epoch
        
        # Log cross-attention activation status
        if self.enable_enhanced_training and hasattr(self.network, 'training_controller'):
            controller = self.network.training_controller
            active_stages = controller.get_active_stages(self.current_epoch)
            weights = [controller.get_cross_attention_weight(self.current_epoch, i) 
                      for i in range(len(active_stages))]
            
            self.print_to_log_file(f"Epoch {self.current_epoch} - Cross-Attention Weights: {weights}")
    
    def maybe_update_lr(self, epoch=None):
        """
        Update learning rate using polynomial decay
        Applies to all parameter groups
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        
        for param_group in self.optimizer.param_groups:
            base_lr = param_group['lr'] if ep == 0 else param_group.get('initial_lr', self.initial_lr)
            if 'initial_lr' not in param_group:
                param_group['initial_lr'] = base_lr
            
            param_group['lr'] = poly_lr(ep, self.max_num_epochs, base_lr, 0.9)
    
    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True,
                 overwrite: bool = True, validation_folder_name: str = 'validation_raw',
                 debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        Enhanced validation with BraTS-specific metrics
        """
        # Disable cross-attention weight scaling during validation
        if hasattr(self.network, 'training_controller'):
            self.network.eval()
        
        ret = super().validate(
            do_mirroring=do_mirroring,
            use_sliding_window=use_sliding_window,
            step_size=step_size,
            save_softmax=save_softmax,
            use_gaussian=use_gaussian,
            overwrite=overwrite,
            validation_folder_name=validation_folder_name,
            debug=debug,
            all_in_gpu=all_in_gpu,
            segmentation_export_kwargs=segmentation_export_kwargs,
            run_postprocessing_on_folds=run_postprocessing_on_folds
        )
        
        return ret


class EnhancednnFormerTrainer_Ablation(EnhancednnFormerTrainer):
    """
    Trainer for ablation studies
    Allows enabling/disabling specific enhancements
    """
    
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True,
                 fp16=False, ablation_config='full'):
        """
        Args:
            ablation_config: One of ['baseline', 'cross_attn', 'fusion', 'training', 'full']
        """
        # Map ablation configs to enhancement flags
        config_map = {
            'baseline': (False, False, False),
            'cross_attn': (True, False, False),
            'fusion': (True, True, False),
            'training': (True, False, True),
            'full': (True, True, True)
        }
        
        enable_cross_attn, enable_fusion, enable_training = config_map.get(
            ablation_config, config_map['full']
        )
        
        super().__init__(
            plans_file, fold, output_folder, dataset_directory,
            batch_dice, stage, unpack_data, deterministic, fp16,
            enable_cross_attention=enable_cross_attn,
            enable_adaptive_fusion=enable_fusion,
            enable_enhanced_training=enable_training
        )
        
        self.ablation_config = ablation_config
        self.print_to_log_file(f"Ablation Configuration: {ablation_config}")
