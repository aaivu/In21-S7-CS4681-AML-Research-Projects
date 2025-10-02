from torch.utils.data import Dataset
import torch

class RevisedToxicityDataset(Dataset):
    """Dataset for prompt-continuation pairs with derived attributes."""
    
    def __init__(self, dataframe, tokenizer, max_length: int = 128):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        prompt = row['prompt_text']
        continuation = row['continuation_text']
        
        # Tokenize with prompt-continuation pair encoding
        encoding = self.tokenizer(
            prompt,
            continuation,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'toxicity_score': torch.tensor(row['toxicity'], dtype=torch.float),
            'sensitive_group': torch.tensor(row['sensitive_attribute'], dtype=torch.long),
            'context_label': torch.tensor(row['context_attribute'], dtype=torch.long)
        }