"""
Example usage of MultiScaleVideoEncoder with temporal consistency loss.

This script demonstrates how to integrate the new multi-scale encoder with
temporal consistency loss in the EgoVLP training pipeline.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import the new components
from model.model import MultiScaleVideoEncoder
from model.temporal_loss import TemporalConsistencyLoss, TemporalBatchSampler, EnhancedEgoNCEWithTemporal
from data_loader.EgoClip_EgoMCQ_dataset import EgoClip_EgoMCQ


class MultiScaleEgoVLP(nn.Module):
    """
    Complete multi-scale EgoVLP model using the new MultiScaleVideoEncoder.
    """
    
    def __init__(self, video_params, text_params, projection_dim=768):
        super().__init__()
        
        # Multi-scale video encoder
        self.video_encoder = MultiScaleVideoEncoder(video_params, projection_dim)
        
        # Text encoder (same as original)
        from transformers import AutoModel
        self.text_model = AutoModel.from_pretrained(text_params['model'])
        
        # Text projection
        self.text_projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.text_model.config.hidden_size, projection_dim)
        )
        
    def forward(self, data):
        """
        Forward pass with multi-scale video processing.
        
        Args:
            data: Dict containing 'video' (multi-scale clips) and 'text' data
            
        Returns:
            text_embeddings: [batch_size, projection_dim]
            video_embeddings: [batch_size, projection_dim] 
        """
        # Process text
        text_data = data['text']
        text_features = self.text_model(**text_data).last_hidden_state[:, 0, :]
        text_embeddings = self.text_projection(text_features)
        
        # Process multi-scale video
        video_clips = data['video']  # Dict with 'frames_4', 'frames_8', 'frames_16'
        video_embeddings = self.video_encoder(video_clips)
        
        return text_embeddings, video_embeddings


def create_enhanced_dataset(data_dir, meta_dir, video_params, text_params):
    """
    Create EgoClip dataset with multi-scale support.
    
    This function modifies the existing dataset to use the new _get_video_multiscale method.
    """
    
    class MultiScaleEgoClip(EgoClip_EgoMCQ):
        def _get_train_item(self, item):
            item = item % len(self.metadata)
            sample = self.metadata.iloc[item]
            video_fp, video_sec, bound_sec = self._get_video_path(sample)
            caption, noun_vec, verb_vec = self._get_caption(sample)
            
            # Use multi-scale video loading
            multi_scale_frames = self._get_video_multiscale(video_fp, video_sec, bound_sec)
            
            meta_arr = {'raw_captions': caption, 'paths': video_fp, 'dataset': self.dataset_name}
            
            return {
                'video': multi_scale_frames,  # Dict with frames_4, frames_8, frames_16
                'text': caption,
                'meta': meta_arr,
                'noun_vec': noun_vec, 
                'verb_vec': verb_vec
            }
    
    dataset = MultiScaleEgoClip(
        dataset_name='EgoClip',
        text_params=text_params,
        video_params=video_params,
        data_dir=data_dir,
        meta_dir=meta_dir,
        split='train'
    )
    
    return dataset


def train_multiscale_model():
    """
    Example training loop with multi-scale encoder and temporal consistency.
    """
    
    # Configuration
    video_params = {
        'model': 'SpaceTimeTransformer',
        'arch_config': 'base_patch16_224',
        'num_frames': 16,  # Max frames for compatibility
        'input_res': 224,
        'pretrained': True,
        'time_init': 'zeros'
    }
    
    text_params = {
        'model': 'distilbert-base-uncased',
        'pretrained': True,
        'input': 'text'
    }
    
    # Create model
    model = MultiScaleEgoVLP(video_params, text_params, projection_dim=768)
    
    # Create dataset with multi-scale support
    dataset = create_enhanced_dataset(
        data_dir='dataset/ego4d_256/data_chunked/',
        meta_dir='dataset/ego4d_toolbox/0_metadata/egovlp',
        video_params=video_params,
        text_params=text_params
    )
    
    # Create temporal batch sampler
    batch_sampler = TemporalBatchSampler(dataset, batch_size=16, temporal_pair_ratio=0.3)
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=4)
    
    # Create loss function with temporal consistency
    criterion = EnhancedEgoNCEWithTemporal(
        temperature=0.05,
        temporal_lambda_start=0.1,
        temporal_lambda_end=0.3
    )
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    # Training loop
    model.train()
    total_epochs = 10
    
    for epoch in range(total_epochs):
        epoch_losses = {'total': 0, 'egonce': 0, 'temporal': 0}
        num_batches = 0
        
        for batch_data, temporal_pairs in dataloader:
            optimizer.zero_grad()
            
            # Move to device (assuming single GPU for simplicity)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Move video data (multi-scale dict) to device
            batch_data['video'] = {
                k: v.to(device) for k, v in batch_data['video'].items()
            }
            
            # Tokenize and move text data
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(text_params['model'])
            batch_data['text'] = tokenizer(
                batch_data['text'], 
                return_tensors='pt', 
                padding=True, 
                truncation=True
            ).to(device)
            
            # Forward pass
            text_embeds, video_embeds = model(batch_data)
            
            # Compute similarity matrix
            similarity_matrix = torch.mm(
                torch.nn.functional.normalize(text_embeds, dim=1),
                torch.nn.functional.normalize(video_embeds, dim=1).t()
            )
            
            # Create masks (simplified - use identity for demo)
            batch_size = similarity_matrix.shape[0]
            mask_n = torch.eye(batch_size).to(device)
            mask_v = torch.eye(batch_size).to(device)
            
            # Compute loss with temporal consistency
            total_loss, loss_dict = criterion(
                similarity_matrix, mask_v, mask_n,
                video_features=video_embeds,
                temporal_pairs=temporal_pairs,
                current_epoch=epoch,
                total_epochs=total_epochs
            )
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Track losses
            epoch_losses['total'] += total_loss.item()
            epoch_losses['egonce'] += loss_dict['egonce_loss'].item()
            epoch_losses['temporal'] += loss_dict['temporal_loss'].item()
            num_batches += 1
            
            # Log fusion weights periodically
            if num_batches % 100 == 0:
                fusion_weights = model.video_encoder.get_fusion_weights()
                print(f"Epoch {epoch}, Batch {num_batches}")
                print(f"Fusion weights: {fusion_weights.detach().cpu().numpy()}")
                print(f"Losses - Total: {total_loss.item():.4f}, "
                      f"EgoNCE: {loss_dict['egonce_loss'].item():.4f}, "
                      f"Temporal: {loss_dict['temporal_loss'].item():.4f}")
        
        # Epoch summary
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        print(f"Epoch {epoch} - Avg Losses: {avg_losses}")
        
        # Log final fusion weights for epoch
        final_weights = model.video_encoder.get_fusion_weights()
        print(f"Epoch {epoch} final fusion weights: {final_weights.detach().cpu().numpy()}")


if __name__ == '__main__':
    print("Multi-Scale EgoVLP Training Example")
    print("===================================")
    
    # Check if required files exist
    import os
    
    required_files = [
        "pretrained/jx_vit_base_p16_224-80ecf9dd.pth",
        "dataset/ego4d_256/data_chunked/",
        "dataset/ego4d_toolbox/0_metadata/egovlp"
    ]
    
    print("Checking required files...")
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (required for training)")
    
    print("\nTo run the training:")
    print("1. Ensure all required files are available")
    print("2. Install dependencies: torch, transformers, etc.")
    print("3. Run: python examples/multiscale_training_example.py")
    print("\nThe model will:")
    print("- Process videos at 3 scales (4, 8, 16 frames)")
    print("- Learn fusion weights (initialized as [0.33, 0.33, 0.33])")
    print("- Apply temporal consistency loss between adjacent clips")
    print("- Return fused features of shape [batch_size, 768]")