import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
from typing import Dict
import numpy as np

class CAFEModel(nn.Module):
    """CAFE: Context-Aware Fairness-Weighted Toxicity Evaluator."""
    
    def __init__(self, 
                 model_name: str = "roberta-base",
                 num_toxicity_classes: int = 1,
                 hidden_dim: int = 768,
                 dropout_rate: float = 0.1,
                 max_length: int = 128):
        """
        Initialize CAFE model.
        
        Args:
            model_name: Pre-trained model name
            num_toxicity_classes: Number of toxicity prediction outputs
            hidden_dim: Hidden dimension size
            dropout_rate: Dropout rate for regularization
            max_length: Maximum input sequence length
        """
        super(CAFEModel, self).__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        
        # Load pre-trained RoBERTa
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        
        # Classification head for toxicity prediction
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_toxicity_classes),
            nn.Sigmoid()  # Output probabilities
        )
        
        # Additional heads for multi-task learning
        self.context_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 2),  # Binary: literal vs non-literal
            nn.Softmax(dim=1)
        )
        
        # Embedding projection for contrastive learning
        self.embedding_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
    def tokenize_inputs(self, texts: list) -> Dict[str, torch.Tensor]:
        """
        Tokenize input texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Dictionary of tokenized inputs
        """
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                return_embeddings: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through CAFE model.
        
        Args:
            input_ids: Tokenized input IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_embeddings: Whether to return contextualized embeddings
            
        Returns:
            Dictionary containing model outputs
        """
        # Get RoBERTa outputs
        roberta_outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Extract [CLS] token embedding for classification
        cls_embedding = roberta_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
        
        # Toxicity prediction
        toxicity_logits = self.classifier(cls_embedding)  # [batch_size, 1]
        toxicity_scores = toxicity_logits.squeeze(-1)  # [batch_size]
        
        # Context prediction (literal vs non-literal)
        context_logits = self.context_classifier(cls_embedding)  # [batch_size, 2]
        
        # Projected embeddings for contrastive learning
        projected_embeddings = self.embedding_projection(cls_embedding)  # [batch_size, hidden_dim//4]
        
        outputs = {
            'toxicity_scores': toxicity_scores,
            'context_logits': context_logits,
            'projected_embeddings': projected_embeddings
        }
        
        if return_embeddings:
            outputs['embeddings'] = cls_embedding
            
        return outputs
    
    def predict(self, texts: list) -> np.ndarray:
        """
        Make predictions on a list of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Array of toxicity scores
        """
        self.eval()
        
        # Tokenize inputs
        inputs = self.tokenize_inputs(texts)
        device = next(self.parameters()).device
        
        # Move to device
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            toxicity_scores = outputs['toxicity_scores'].cpu().numpy()
            
        return toxicity_scores

class BaselineModel(nn.Module):
    """Baseline model for comparison (simple classifier)."""
    
    def __init__(self, model_name: str = "roberta-base", max_length: int = 128):
        super(BaselineModel, self).__init__()
        
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
        # Simple classification head
        self.classifier = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        toxicity_scores = self.classifier(cls_embedding).squeeze(-1)
        return toxicity_scores
    
    def predict(self, texts: list) -> np.ndarray:
        self.eval()
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, 
            max_length=self.max_length, return_tensors="pt"
        )
        
        device = next(self.parameters()).device
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        with torch.no_grad():
            toxicity_scores = self.forward(input_ids, attention_mask).cpu().numpy()
            
        return toxicity_scores

if __name__ == "__main__":
    # Test model initialization
    model = CAFEModel()
    baseline = BaselineModel()
    
    # Test with dummy data
    texts = ["This is a test sentence.", "Another example text."]
    
    print("CAFE Model:")
    predictions = model.predict(texts)
    print(f"Predictions: {predictions}")
    
    print("\nBaseline Model:")
    baseline_predictions = baseline.predict(texts)
    print(f"Predictions: {baseline_predictions}")