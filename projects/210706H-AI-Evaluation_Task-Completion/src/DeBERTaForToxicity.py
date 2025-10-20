"""
DeBERTa-based Multi-Task Model for Toxicity Classification
"""

import torch
import torch.nn as nn
from transformers import DebertaV2Model, DebertaV2Config


class DeBERTaForToxicity(nn.Module):
    """
    DeBERTa-v3-large based model for multi-task toxicity classification.
    
    Outputs:
        - Primary target: overall toxicity
        - 6 Auxiliary targets: severe_toxicity, obscene, identity_attack, 
          insult, threat, sexual_explicit
    """
    
    def __init__(self, model_name='microsoft/deberta-v3-large', dropout_rate=0.1):
        """
        Initialize the DeBERTa toxicity model.
        
        Args:
            model_name (str): HuggingFace model identifier
            dropout_rate (float): Dropout probability for classifier head
        """
        super(DeBERTaForToxicity, self).__init__()
        
        # Load pretrained DeBERTa model
        self.deberta = DebertaV2Model.from_pretrained(model_name)
        
        # Get hidden size from model config
        self.hidden_size = self.deberta.config.hidden_size
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Multi-task classification heads
        # 7 outputs total: 1 primary (toxicity) + 6 auxiliary
        self.classifier = nn.Linear(self.hidden_size, 7)
        
        # Initialize classifier weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the classification head weights."""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs [batch_size, seq_len]
            attention_mask (torch.Tensor): Attention mask [batch_size, seq_len]
            
        Returns:
            dict: Dictionary containing logits for all 7 tasks
                - 'toxicity': primary target logits
                - 'severe_toxicity', 'obscene', 'identity_attack', 
                  'insult', 'threat', 'sexual_explicit': auxiliary logits
        """
        # Get DeBERTa outputs
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Get logits for all tasks
        logits = self.classifier(pooled_output)
        
        # Split logits into individual tasks
        return {
            'toxicity': logits[:, 0],  # Primary target
            'severe_toxicity': logits[:, 1],
            'obscene': logits[:, 2],
            'identity_attack': logits[:, 3],
            'insult': logits[:, 4],
            'threat': logits[:, 5],
            'sexual_explicit': logits[:, 6]
        }
    
    def freeze_backbone(self):
        """Freeze DeBERTa parameters for feature extraction only."""
        for param in self.deberta.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze DeBERTa parameters for fine-tuning."""
        for param in self.deberta.parameters():
            param.requires_grad = True
    
    def get_trainable_parameters(self):
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)