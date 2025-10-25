"""
Integration example: Using TemporalPairBatchSampler with actual EgoClip dataset.

This script shows how to integrate the custom batch sampler with the existing
EgoVLP training pipeline for temporal consistency learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import transformers

# Import EgoVLP components
from model.model import MultiScaleVideoEncoder
from model.temporal_loss import TemporalPairBatchSampler, TemporalConsistencyLoss, EnhancedEgoNCEWithTemporal
from data_loader.EgoClip_EgoMCQ_dataset import EgoClip_EgoMCQ


class MultiScaleEgoVLPWithTemporal(nn.Module):
    """
    Complete EgoVLP model with multi-scale processing and temporal consistency.
    
    Combines MultiScaleVideoEncoder with text processing and supports
    temporal consistency loss through the custom batch sampler.
    """
    
    def __init__(self, video_params, text_params, projection_dim=768):
        super().__init__()
        
        # Multi-scale video encoder
        self.video_encoder = MultiScaleVideoEncoder(video_params, projection_dim)
        
        # Text encoder
        self.text_model = transformers.AutoModel.from_pretrained(text_params['model'])
        
        # Text projection
        self.text_projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.text_model.config.hidden_size, projection_dim)
        )
        
    def forward(self, batch_data, return_individual_features=False):
        """
        Forward pass with multi-scale video processing.
        
        Args:
            batch_data: Batch containing 'video' (multi-scale) and 'text' data
            return_individual_features: If True, return individual video features for temporal loss
            
        Returns:
            text_embeddings: [batch_size, projection_dim]
            video_embeddings: [batch_size, projection_dim]
            individual_features: Optional individual scale features for temporal consistency
        """
        # Process text
        text_data = batch_data['text']
        if isinstance(text_data, list):
            # Tokenize text if needed
            tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased')
            text_data = tokenizer(text_data, return_tensors='pt', padding=True, truncation=True)
            text_data = {k: v.to(next(self.parameters()).device) for k, v in text_data.items()}
        
        text_features = self.text_model(**text_data).last_hidden_state[:, 0, :]
        text_embeddings = self.text_projection(text_features)
        
        # Process multi-scale video
        video_clips = batch_data['video']
        video_embeddings = self.video_encoder(video_clips)
        
        if return_individual_features:
            # For temporal consistency, we can use the fused features
            # In a more sophisticated implementation, you might want individual scale features
            return text_embeddings, video_embeddings, video_embeddings
        
        return text_embeddings, video_embeddings


class TemporalDataLoader:
    """
    Custom DataLoader wrapper that works with TemporalPairBatchSampler.
    
    The standard DataLoader doesn't work directly with our custom sampler that returns
    metadata, so we create a wrapper that handles the temporal pair information.
    """
    
    def __init__(self, dataset, batch_sampler, num_workers=0):
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.num_workers = num_workers
        
    def __iter__(self):
        """
        Iterate through batches with temporal pair metadata.
        
        Yields:
            Tuple: (batch_data, temporal_pairs_metadata)
        """
        for batch_indices, temporal_pairs_metadata in self.batch_sampler:
            # Collect batch data
            batch_items = []
            for idx in batch_indices:
                item = self.dataset[idx]
                batch_items.append(item)
            
            # Collate batch
            batch_data = self._collate_fn(batch_items)
            
            yield batch_data, temporal_pairs_metadata
    
    def _collate_fn(self, batch_items):
        """
        Collate function to combine individual items into a batch.
        
        Args:
            batch_items: List of individual dataset items
            
        Returns:
            Batch dictionary with stacked tensors
        """
        # Separate video and text data
        video_data = {}
        texts = []
        noun_vecs = []
        verb_vecs = []
        
        for item in batch_items:
            texts.append(item['text'])
            
            if 'noun_vec' in item:
                noun_vecs.append(item['noun_vec'])
            if 'verb_vec' in item:
                verb_vecs.append(item['verb_vec'])
            
            # Handle multi-scale video data
            if isinstance(item['video'], dict):
                # Multi-scale format
                for key, tensor in item['video'].items():
                    if key not in video_data:
                        video_data[key] = []
                    video_data[key].append(tensor)
            else:
                # Single-scale format - convert to multi-scale
                single_video = item['video']
                if 'frames_4' not in video_data:
                    video_data['frames_4'] = []
                    video_data['frames_8'] = []
                    video_data['frames_16'] = []
                
                # Temporal subsampling for different scales
                T = single_video.shape[0]
                
                # Fine scale (4 frames)
                if T >= 4:
                    indices_4 = torch.linspace(0, T-1, 4).long()
                    video_data['frames_4'].append(single_video[indices_4])
                else:
                    # Pad if necessary
                    padded = torch.zeros(4, *single_video.shape[1:])
                    padded[:T] = single_video
                    video_data['frames_4'].append(padded)
                
                # Medium scale (8 frames)
                if T >= 8:
                    indices_8 = torch.linspace(0, T-1, 8).long()
                    video_data['frames_8'].append(single_video[indices_8])
                else:
                    padded = torch.zeros(8, *single_video.shape[1:])
                    padded[:T] = single_video
                    if T > 0:
                        # Repeat last frame
                        for i in range(T, 8):
                            padded[i] = single_video[-1]
                    video_data['frames_8'].append(padded)
                
                # Coarse scale (16 frames)
                if T >= 16:
                    indices_16 = torch.linspace(0, T-1, 16).long()
                    video_data['frames_16'].append(single_video[indices_16])
                else:
                    padded = torch.zeros(16, *single_video.shape[1:])
                    padded[:T] = single_video
                    if T > 0:
                        # Repeat last frame
                        for i in range(T, 16):
                            padded[i] = single_video[-1]
                    video_data['frames_16'].append(padded)
        
        # Stack tensors
        for key in video_data:
            video_data[key] = torch.stack(video_data[key])
        
        batch_dict = {
            'video': video_data,
            'text': texts
        }
        
        if noun_vecs:
            batch_dict['noun_vec'] = torch.stack(noun_vecs)
        if verb_vecs:
            batch_dict['verb_vec'] = torch.stack(verb_vecs)
        
        return batch_dict
    
    def __len__(self):
        return len(self.batch_sampler)


def create_temporal_training_setup(config):
    """
    Create complete training setup with temporal consistency.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (model, dataset, temporal_dataloader, loss_fn, optimizer)
    """
    # Configuration
    video_params = config['video_params']
    text_params = config['text_params']
    data_config = config['data']
    training_config = config['training']
    
    # Create dataset
    dataset = EgoClip_EgoMCQ(
        dataset_name='EgoClip',
        text_params=text_params,
        video_params=video_params,
        data_dir=data_config['data_dir'],
        meta_dir=data_config['meta_dir'],
        split='train'
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Create temporal pair batch sampler
    batch_sampler = TemporalPairBatchSampler(
        dataset=dataset,
        batch_size=training_config['batch_size'],
        temporal_pair_ratio=training_config['temporal_pair_ratio'],
        max_temporal_gap=training_config['max_temporal_gap']
    )
    
    # Create temporal data loader
    temporal_dataloader = TemporalDataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        num_workers=training_config.get('num_workers', 0)
    )
    
    # Create model
    model = MultiScaleEgoVLPWithTemporal(
        video_params=video_params,
        text_params=text_params,
        projection_dim=training_config['projection_dim']
    )
    
    # Create enhanced loss function with temporal consistency
    loss_fn = EnhancedEgoNCEWithTemporal(
        temperature=training_config['temperature'],
        temporal_lambda_start=training_config['temporal_lambda_start'],
        temporal_lambda_end=training_config['temporal_lambda_end']
    )
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay']
    )
    
    return model, dataset, temporal_dataloader, loss_fn, optimizer


def temporal_training_step(model, batch_data, temporal_pairs_metadata, loss_fn, 
                          current_epoch, total_epochs, device):
    """
    Single training step with temporal consistency.
    
    Args:
        model: MultiScaleEgoVLPWithTemporal model
        batch_data: Batch data from temporal dataloader
        temporal_pairs_metadata: Temporal pair information
        loss_fn: EnhancedEgoNCEWithTemporal loss function
        current_epoch: Current training epoch
        total_epochs: Total training epochs
        device: Training device
        
    Returns:
        Dictionary with loss components and metrics
    """
    # Move data to device
    for key in batch_data['video']:
        batch_data['video'][key] = batch_data['video'][key].to(device)
    
    # Forward pass
    text_embeds, video_embeds, individual_features = model(
        batch_data, return_individual_features=True
    )
    
    # Compute similarity matrix
    text_embeds_norm = torch.nn.functional.normalize(text_embeds, dim=1)
    video_embeds_norm = torch.nn.functional.normalize(video_embeds, dim=1)
    similarity_matrix = torch.mm(text_embeds_norm, video_embeds_norm.t())
    
    # Create masks (simplified for demo - use identity matrices)
    batch_size = similarity_matrix.shape[0]
    mask_n = torch.eye(batch_size).to(device)
    mask_v = torch.eye(batch_size).to(device)
    
    # Apply loss with temporal consistency
    total_loss, loss_dict = loss_fn(
        similarity_matrix=similarity_matrix,
        mask_v=mask_v,
        mask_n=mask_n,
        video_features=individual_features,
        temporal_pairs=temporal_pairs_metadata,
        current_epoch=current_epoch,
        total_epochs=total_epochs
    )
    
    return {
        'total_loss': total_loss,
        'loss_components': loss_dict,
        'similarity_matrix': similarity_matrix,
        'batch_size': batch_size,
        'num_temporal_pairs': len(temporal_pairs_metadata)
    }


def demo_temporal_training():
    """
    Demonstrate temporal consistency training with real EgoClip integration.
    """
    print("Temporal Consistency Training Integration Demo")
    print("=" * 50)
    
    # Configuration
    config = {
        'video_params': {
            'model': 'SpaceTimeTransformer',
            'arch_config': 'base_patch16_224',
            'pretrained': True,
            'time_init': 'zeros'
        },
        'text_params': {
            'model': 'distilbert-base-uncased',
            'pretrained': True
        },
        'data': {
            'data_dir': 'dataset/ego4d_256/data_chunked/',
            'meta_dir': 'dataset/ego4d_toolbox/0_metadata/egovlp'
        },
        'training': {
            'batch_size': 8,
            'temporal_pair_ratio': 0.3,
            'max_temporal_gap': 2.0,
            'projection_dim': 768,
            'temperature': 0.05,
            'temporal_lambda_start': 0.1,
            'temporal_lambda_end': 0.3,
            'learning_rate': 2e-5,
            'weight_decay': 0.01,
            'num_workers': 0
        }
    }
    
    try:
        # Create training setup
        model, dataset, temporal_dataloader, loss_fn, optimizer = create_temporal_training_setup(config)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        print(f"✓ Training setup created successfully")
        print(f"✓ Device: {device}")
        print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Simulate training steps
        model.train()
        total_epochs = 10
        current_epoch = 0
        
        print(f"\nSimulating training steps...")
        
        for step, (batch_data, temporal_pairs_metadata) in enumerate(temporal_dataloader):
            if step >= 3:  # Just demo first 3 steps
                break
                
            optimizer.zero_grad()
            
            # Training step with temporal consistency
            step_results = temporal_training_step(
                model=model,
                batch_data=batch_data,
                temporal_pairs_metadata=temporal_pairs_metadata,
                loss_fn=loss_fn,
                current_epoch=current_epoch,
                total_epochs=total_epochs,
                device=device
            )
            
            # Backward pass
            step_results['total_loss'].backward()
            optimizer.step()
            
            print(f"\nStep {step + 1}:")
            print(f"  Batch size: {step_results['batch_size']}")
            print(f"  Temporal pairs: {step_results['num_temporal_pairs']}")
            print(f"  Total loss: {step_results['total_loss'].item():.6f}")
            
            if 'loss_components' in step_results:
                components = step_results['loss_components']
                for key, value in components.items():
                    if hasattr(value, 'item'):
                        print(f"  {key}: {value.item():.6f}")
            
            # Show fusion weights
            fusion_weights = model.video_encoder.get_fusion_weights()
            print(f"  Fusion weights: {fusion_weights.detach().cpu().numpy()}")
        
        print("\n✓ Training steps completed successfully")
        
    except Exception as e:
        print(f"✗ Training demo failed: {e}")
        print("Note: This demo requires actual EgoClip dataset files")
        print("Make sure the data directories exist and contain the dataset")


if __name__ == '__main__':
    print("EgoVLP Temporal Consistency Integration")
    print("=" * 60)
    
    # Run the demo
    demo_temporal_training()
    
    print("\n" + "=" * 60)
    print("INTEGRATION DEMO COMPLETED!")
    print("=" * 60)
    
    print("\nKey Integration Features:")
    print("✓ TemporalPairBatchSampler with actual EgoClip dataset")
    print("✓ MultiScaleVideoEncoder with temporal consistency")
    print("✓ Custom TemporalDataLoader handling metadata")
    print("✓ EnhancedEgoNCEWithTemporal loss function")
    print("✓ Complete training step with backward pass")
    print("✓ Fusion weight monitoring during training")
    
    print("\nTo use in your training:")
    print("1. Replace config paths with your actual dataset paths")
    print("2. Adjust batch_size and temporal_pair_ratio as needed")
    print("3. Integrate with your existing training loop")
    print("4. Monitor temporal consistency loss and fusion weights")
    print("5. Evaluate on downstream tasks to measure improvements")