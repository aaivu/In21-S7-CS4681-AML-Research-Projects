import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
from typing import Dict, List
import numpy as np

class RevisedCAFEModel(nn.Module):
    """
    CAFE model with separate prompt/continuation encoding.
    Predicts toxicity of CONTINUATION given prompt context.
    """
    
    def __init__(self, 
                 model_name: str = "roberta-base",
                 hidden_dim: int = 768,
                 dropout_rate: float = 0.1,
                 max_length: int = 128):
        super(RevisedCAFEModel, self).__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        
        # Load pre-trained RoBERTa
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        
        # Toxicity prediction head
        self.toxicity_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def tokenize_prompt_continuation(self, 
                                     prompts: List[str], 
                                     continuations: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize prompts and continuations separately with clear delimiter.
        Format: [CLS] prompt [SEP] continuation [SEP]
        """
        # Use tokenizer's pair encoding which handles delimiters
        encoded = self.tokenizer(
            prompts,
            continuations,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return encoded
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                return_embeddings: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass through model."""
        
        # Get RoBERTa outputs
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Extract [CLS] embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Predict continuation toxicity
        toxicity_scores = self.toxicity_head(cls_embedding).squeeze(-1)
        
        result = {'toxicity_scores': toxicity_scores}
        
        if return_embeddings:
            result['embeddings'] = cls_embedding
            
        return result
    
    def predict(self, prompts: List[str], continuations: List[str]) -> np.ndarray:
        """Predict toxicity scores for prompt-continuation pairs."""
        self.eval()
        
        inputs = self.tokenize_prompt_continuation(prompts, continuations)
        device = next(self.parameters()).device
        
        # Move to device
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            scores = outputs['toxicity_scores'].cpu().numpy()
        
        return scores