"""
Custom PyTorch Dataset for Jigsaw Toxicity Data
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class JigsawDataset(Dataset):
    """
    PyTorch Dataset for Jigsaw Unintended Bias dataset.
    
    Handles tokenization and returns multi-task labels for:
    - Primary target: toxicity
    - Auxiliary targets: severe_toxicity, obscene, identity_attack, insult, threat, sexual_explicit
    """
    
    def __init__(self, dataframe, tokenizer, max_length=512, text_column='comment_text_cleaned'):
        """
        Initialize dataset.
        
        Args:
            dataframe (pd.DataFrame): DataFrame containing text and labels
            tokenizer: HuggingFace tokenizer
            max_length (int): Maximum sequence length for tokenization
            text_column (str): Name of the column containing text
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        
        # Define target columns
        self.primary_target = 'target'
        self.auxiliary_targets = [
            'severe_toxicity',
            'obscene', 
            'identity_attack',
            'insult',
            'threat',
            'sexual_explicit'
        ]
        
        # Verify all required columns exist
        required_cols = [self.text_column, self.primary_target] + self.auxiliary_targets
        missing_cols = [col for col in required_cols if col not in self.dataframe.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def __len__(self):
        """Return dataset size."""
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Args:
            idx (int): Sample index
            
        Returns:
            dict: Contains input_ids, attention_mask, and all labels
        """
        row = self.dataframe.iloc[idx]
        
        # Get text
        text = str(row[self.text_column])
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Get labels (all as floats for BCE loss)
        labels = {
            'toxicity': float(row[self.primary_target]),
            'severe_toxicity': float(row['severe_toxicity']),
            'obscene': float(row['obscene']),
            'identity_attack': float(row['identity_attack']),
            'insult': float(row['insult']),
            'threat': float(row['threat']),
            'sexual_explicit': float(row['sexual_explicit'])
        }
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels
        }
    
    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for DataLoader.
        
        Args:
            batch (list): List of samples from __getitem__
            
        Returns:
            dict: Batched tensors
        """
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        # Stack all labels
        labels = {}
        label_keys = batch[0]['labels'].keys()
        for key in label_keys:
            labels[key] = torch.tensor([item['labels'][key] for item in batch], dtype=torch.float32)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def calculate_sample_weights(df, identity_columns=None):
    """
    Calculate fairness-aware sample weights for training.
    
    Weighting scheme:
    - Base weight: 1.0 for all samples
    - Identity mention: 1.5 (any identity column > 0)
    - BPSN (Background Positive, Subgroup Negative): 2.0
      * Non-toxic (target < 0.5) + mentions identity
    - BNSP (Background Negative, Subgroup Positive): 2.0
      * Toxic (target >= 0.5) + does NOT mention identity
    
    Args:
        df (pd.DataFrame): Training dataframe
        identity_columns (list): List of identity column names
        
    Returns:
        np.ndarray: Array of sample weights
    """
    if identity_columns is None:
        # Default Jigsaw identity columns
        identity_columns = [
            'male', 'female', 'transgender', 'other_gender',
            'heterosexual', 'homosexual_gay_or_lesbian', 'bisexual',
            'other_sexual_orientation', 'christian', 'jewish', 'muslim',
            'hindu', 'buddhist', 'atheist', 'other_religion',
            'black', 'white', 'asian', 'latino', 'other_race_or_ethnicity',
            'physical_disability', 'intellectual_or_learning_disability',
            'psychiatric_or_mental_illness', 'other_disability'
        ]
    
    # Filter to columns that exist in dataframe
    available_identity_cols = [col for col in identity_columns if col in df.columns]
    
    if not available_identity_cols:
        print("Warning: No identity columns found. Using uniform weights.")
        return np.ones(len(df))
    
    # Initialize base weights
    weights = np.ones(len(df))
    
    # Check if any identity is mentioned (any identity column > 0)
    identity_mentioned = (df[available_identity_cols] > 0).any(axis=1)
    
    # Get toxicity labels
    is_toxic = df['target'] >= 0.5
    
    # Apply weighting scheme
    # 1. Identity mention: weight = 1.5
    weights[identity_mentioned] = 1.5
    
    # 2. BPSN (non-toxic + identity): weight = 2.0
    bpsn_mask = (~is_toxic) & identity_mentioned
    weights[bpsn_mask] = 2.0
    
    # 3. BNSP (toxic + no identity): weight = 2.0
    bnsp_mask = is_toxic & (~identity_mentioned)
    weights[bnsp_mask] = 2.0
    
    print(f"\nSample weight distribution:")
    print(f"  Base weight (1.0): {(weights == 1.0).sum()}")
    print(f"  Identity mention (1.5): {(weights == 1.5).sum()}")
    print(f"  BPSN samples (2.0): {bpsn_mask.sum()}")
    print(f"  BNSP samples (2.0): {bnsp_mask.sum()}")
    
    return weights