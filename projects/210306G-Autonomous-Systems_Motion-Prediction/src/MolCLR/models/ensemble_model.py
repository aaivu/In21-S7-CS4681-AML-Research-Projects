import torch
from torch import nn

class EnsembleModel(nn.Module):
    """
    Wraps GIN and GCN models to ensemble them at the feature level
    by concatenating their graph embeddings.
    """
    def __init__(self, model_gin, model_gcn, config):
        super(EnsembleModel, self).__init__()
        self.gin = model_gin
        self.gcn = model_gcn
        self.task = config['dataset']['task']
        
        # Get feature and embedding dimensions from config
        feat_dim = config['model']['feat_dim']
        
        # The combined embedding dim will be feat_dim (from GIN) + feat_dim (from GCN)
        combined_feat_dim = feat_dim * 2 
        
        if self.task == 'classification':
            output_dim = 2
        else: # Regression
            output_dim = 1
            
        # --- Create a NEW prediction head for the combined embeddings ---
        # We use a 2-layer head just like the individual models
        self.ensemble_pred_head = nn.Sequential(
            nn.Linear(combined_feat_dim, feat_dim), # e.g., 1024 -> 512
            nn.Softplus(),
            nn.Linear(feat_dim, output_dim) # e.g., 512 -> 1 or 2
        )

    def forward(self, data):
        # 1. Get graph embeddings from both models
        #    The models return (embedding, prediction). We only want the embedding.
        #    We use .clone() on data to prevent in-place modification issues
        #    if the models modify the data object (like adding self-loops).
        gin_emb, __ = self.gin(data.clone())
        gcn_emb, __ = self.gcn(data.clone())
        
        # 2. Concatenate the embeddings
        combined_emb = torch.cat([gin_emb, gcn_emb], dim=1)
        
        # 3. Pass the combined embedding through the new head
        pred = self.ensemble_pred_head(combined_emb)
        
        # 4. Return in the same (embedding, prediction) format
        return combined_emb, pred