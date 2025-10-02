import pandas as pd
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from typing import List, Dict
import random
from tqdm import tqdm
import nltk
import logging
import os
import requests
import json
from datasets import load_dataset

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logger = logging.getLogger(__name__)

class DataAugmenter:
    """Data augmentation for toxicity evaluation dataset."""
    
    def __init__(self, model_name: str = "t5-small"):
        self.paraphrase_model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.paraphrase_tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.paraphrase_model.to(self.device)
        
        # Sarcastic patterns for adversarial crafting
        self.sarcasm_patterns = [
            "Oh yeah, {text} - that's totally believable.",
            "Sure, {text}. Right.",
            "Wow, {text}. How original.",
            "{text}... said no one ever.",
            "Because {text} makes perfect sense."
        ]
        
        # Slang replacements
        self.slang_replacements = {
            "awesome": ["lit", "fire", "dope", "sick"],
            "bad": ["wack", "trash", "weak", "bogus"],
            "good": ["solid", "tight", "fresh", "clean"],
            "crazy": ["wild", "insane", "mental", "nuts"],
            "cool": ["chill", "tight", "smooth", "rad"]
        }
    
    def paraphrase_text(self, text: str, num_paraphrases: int = 1) -> List[str]:
        """Generate paraphrases using T5 model."""
        paraphrases = []
        
        for _ in range(num_paraphrases):
            input_text = f"paraphrase: {text}"
            inputs = self.paraphrase_tokenizer.encode(
                input_text, 
                return_tensors="pt", 
                max_length=128, 
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.paraphrase_model.generate(
                    inputs,
                    max_length=128,
                    num_beams=4,
                    temperature=0.8,
                    do_sample=True,
                    early_stopping=True
                )
            
            paraphrase = self.paraphrase_tokenizer.decode(outputs[0], skip_special_tokens=True)
            if paraphrase and paraphrase != text:
                paraphrases.append(paraphrase)
        
        return paraphrases
    
    def create_adversarial_samples(self, text: str, num_samples: int = 1) -> List[Dict[str, str]]:
        """Create adversarial samples with sarcasm and slang."""
        adversarial_samples = []
        
        for _ in range(num_samples):
            # Sarcastic variants
            if random.random() < 0.5:
                pattern = random.choice(self.sarcasm_patterns)
                sarcastic_text = pattern.format(text=text.lower())
                adversarial_samples.append({
                    'text': sarcastic_text,
                    'context_label': 1,  # 1 for non-literal/sarcastic
                    'augmentation_type': 'sarcasm'
                })
            
            # Slang variants
            words = text.split()
            modified_words = []
            for word in words:
                word_lower = word.lower()
                if word_lower in self.slang_replacements:
                    replacement = random.choice(self.slang_replacements[word_lower])
                    modified_words.append(replacement)
                else:
                    modified_words.append(word)
            
            slang_text = ' '.join(modified_words)
            if slang_text != text:
                adversarial_samples.append({
                    'text': slang_text,
                    'context_label': 0,  # 0 for literal
                    'augmentation_type': 'slang'
                })
        
        return adversarial_samples
    
    def augment_dataset(self, df: pd.DataFrame, num_augmentations: int = 2) -> pd.DataFrame:
        """Augment the entire dataset with paraphrases and adversarial samples."""
        logger.info(f"Starting dataset augmentation with {len(df)} original samples...")
        
        augmented_data = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting data"):
            text = row.get('prompt', '') + ' ' + row.get('continuation', '')
            text = text.strip()
            
            # Original sample
            original_sample = row.to_dict()
            original_sample['context_label'] = 0  # Default to literal
            original_sample['augmentation_type'] = 'original'
            augmented_data.append(original_sample)
            
            try:
                # Paraphrases
                paraphrases = self.paraphrase_text(text, num_paraphrases=1)
                for paraphrase in paraphrases:
                    aug_sample = row.to_dict()
                    # Split paraphrase back into prompt and continuation (simple heuristic)
                    words = paraphrase.split()
                    mid_point = len(words) // 2
                    aug_sample['prompt'] = ' '.join(words[:mid_point])
                    aug_sample['continuation'] = ' '.join(words[mid_point:])
                    aug_sample['context_label'] = 0
                    aug_sample['augmentation_type'] = 'paraphrase'
                    augmented_data.append(aug_sample)
                
                # Adversarial samples
                adversarial_samples = self.create_adversarial_samples(text, num_samples=1)
                for adv_sample in adversarial_samples:
                    aug_sample = row.to_dict()
                    # Split adversarial text back into prompt and continuation
                    words = adv_sample['text'].split()
                    mid_point = len(words) // 2
                    aug_sample['prompt'] = ' '.join(words[:mid_point])
                    aug_sample['continuation'] = ' '.join(words[mid_point:])
                    aug_sample['context_label'] = adv_sample['context_label']
                    aug_sample['augmentation_type'] = adv_sample['augmentation_type']
                    augmented_data.append(aug_sample)
                    
            except Exception as e:
                logger.warning(f"Error augmenting sample {idx}: {e}")
                continue
        
        augmented_df = pd.DataFrame(augmented_data)
        logger.info(f"Augmentation complete. Generated {len(augmented_df)} samples from {len(df)} original samples.")
        
        return augmented_df

def load_rtp_dataset(data_path: str = "data/raw") -> pd.DataFrame:
    """
    Load actual RealToxicityPrompts dataset.
    
    Args:
        data_path: Directory to save/load the dataset
        
    Returns:
        DataFrame with RTP data
    """
    os.makedirs(data_path, exist_ok=True)
    rtp_file = os.path.join(data_path, "realtoxicityprompts.jsonl")
    
    # Check if dataset already exists
    if os.path.exists(rtp_file):
        logger.info(f"Loading existing RTP dataset from {rtp_file}")
        return _load_rtp_from_file(rtp_file)
    
    # Method 1: Try loading from Hugging Face datasets
    try:
        logger.info("Attempting to load RTP dataset from Hugging Face...")
        dataset = load_dataset("allenai/real-toxicity-prompts", split="train")
        df = dataset.to_pandas()
        
        # Standardize column names
        df = _standardize_rtp_columns(df)
        
        # Save for future use
        df.to_csv(os.path.join(data_path, "rtp_dataset.csv"), index=False)
        logger.info(f"RTP dataset loaded and saved with {len(df)} samples")
        return df
        
    except Exception as e:
        logger.warning(f"Failed to load from Hugging Face: {e}")
    
    # Method 2: Try downloading directly from Allen AI
    try:
        logger.info("Attempting to download RTP dataset directly...")
        return _download_rtp_dataset(data_path)
        
    except Exception as e:
        logger.warning(f"Failed to download RTP dataset: {e}")
    
    # Method 3: Load from local file if available
    local_files = [
        "realtoxicityprompts-data.jsonl",
        "realtoxicityprompts.jsonl", 
        "rtp_prompts.jsonl"
    ]
    
    for filename in local_files:
        filepath = os.path.join(data_path, filename)
        if os.path.exists(filepath):
            logger.info(f"Loading RTP from local file: {filepath}")
            return _load_rtp_from_file(filepath)
    
    # Fallback: Create a message for manual download
    logger.error("Could not automatically load RTP dataset")
    _print_rtp_manual_instructions(data_path)
    
    # Return empty DataFrame as fallback
    return pd.DataFrame()

def _load_rtp_from_file(filepath: str) -> pd.DataFrame:
    """Load RTP data from JSONL file."""
    data = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed line: {e}")
                continue
    
    df = pd.DataFrame(data)
    return _standardize_rtp_columns(df)

def _standardize_rtp_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize RTP dataset column names and structure."""
    
    # Handle different possible structures
    if 'prompt' in df.columns and 'continuation' in df.columns:
        # Already in correct format
        pass
    elif 'prompt' in df.columns and isinstance(df['prompt'].iloc[0], dict):
        # Nested structure - extract text from prompt dict
        df['prompt_text'] = df['prompt'].apply(lambda x: x.get('text', '') if isinstance(x, dict) else str(x))
        df['prompt'] = df['prompt_text']
        
    elif 'challenging' in df.columns:
        # Different format with challenging flag
        df['prompt'] = df.get('text', df.get('prompt', ''))
        df['continuation'] = ''  # May need to be filled
        
    # Ensure required columns exist
    required_cols = ['prompt', 'continuation', 'toxicity']
    
    for col in required_cols:
        if col not in df.columns:
            if col == 'toxicity':
                # Extract toxicity from nested structure if available
                if 'continuation' in df.columns and isinstance(df['continuation'].iloc[0], dict):
                    df['toxicity'] = df['continuation'].apply(
                        lambda x: x.get('toxicity', 0.0) if isinstance(x, dict) else 0.0
                    )
                else:
                    df['toxicity'] = 0.0
            elif col == 'continuation':
                df['continuation'] = ''
            else:
                df[col] = ''
    
    # Extract additional toxicity scores if available
    toxicity_attributes = [
        'severe_toxicity', 'identity_attack', 'insult', 
        'threat', 'profanity', 'sexually_explicit', 'flirtation'
    ]
    
    for attr in toxicity_attributes:
        if attr not in df.columns:
            df[attr] = 0.0
    
    # Add identity mention flag
    identity_keywords = [
        'black', 'white', 'asian', 'latino', 'hispanic', 'jewish', 'muslim', 
        'christian', 'gay', 'lesbian', 'transgender', 'lgbtq'
    ]
    
    df['identity_mention'] = df.apply(
        lambda row: int(any(keyword in str(row['prompt']).lower() + ' ' + str(row['continuation']).lower() 
                           for keyword in identity_keywords)), axis=1
    )
    
    # Add context label (0 for literal, 1 for non-literal)
    sarcasm_indicators = ['yeah right', 'sure', 'totally', '!', 'obviously']
    df['context_label'] = df.apply(
        lambda row: int(any(indicator in str(row['prompt']).lower() + ' ' + str(row['continuation']).lower() 
                           for indicator in sarcasm_indicators)), axis=1
    )
    
    logger.info(f"Standardized RTP dataset: {len(df)} samples, {df.columns.tolist()}")
    return df

def _download_rtp_dataset(data_path: str) -> pd.DataFrame:
    """Download RTP dataset from official source."""
    
    # Official RTP data URL (this may need to be updated)
    urls = [
        "https://ai2-public-datasets.s3.amazonaws.com/realtoxicityprompts/realtoxicityprompts-data.jsonl.gz",
        "https://github.com/allenai/real-toxicity-prompts/releases/download/v1.0/realtoxicityprompts-data.jsonl.gz"
    ]
    
    for url in urls:
        try:
            logger.info(f"Downloading RTP dataset from {url}")
            
            # Download the file
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Save compressed file
            gz_path = os.path.join(data_path, "realtoxicityprompts-data.jsonl.gz")
            with open(gz_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract and load
            import gzip
            jsonl_path = os.path.join(data_path, "realtoxicityprompts-data.jsonl")
            
            with gzip.open(gz_path, 'rb') as f_in:
                with open(jsonl_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            
            # Load the extracted file
            df = _load_rtp_from_file(jsonl_path)
            
            # Clean up compressed file
            os.remove(gz_path)
            
            logger.info(f"Successfully downloaded and loaded RTP dataset with {len(df)} samples")
            return df
            
        except Exception as e:
            logger.warning(f"Failed to download from {url}: {e}")
            continue
    
    raise Exception("Could not download RTP dataset from any source")

def _print_rtp_manual_instructions(data_path: str):
    """Print instructions for manual RTP dataset download."""
    
    instructions = f"""
    
    ⚠️  MANUAL RTP DATASET DOWNLOAD REQUIRED ⚠️
    
    The RealToxicityPrompts dataset could not be automatically downloaded.
    Please follow these steps:
    
    1. Go to: https://github.com/allenai/real-toxicity-prompts
    2. Download the dataset file (realtoxicityprompts-data.jsonl.gz)
    3. Extract it and place the .jsonl file in: {data_path}/
    4. Rename it to: realtoxicityprompts.jsonl
    5. Re-run your code
    
    Alternative sources:
    - Hugging Face: https://huggingface.co/datasets/allenai/real-toxicity-prompts  
    - Direct download: https://ai2-public-datasets.s3.amazonaws.com/realtoxicityprompts/
    
    """
    
    print(instructions)
    logger.error(instructions)

if __name__ == "__main__":
    import os
    from utils import setup_logging
    
    setup_logging()
    
    # Create directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/augmented", exist_ok=True)
    
    # Load dataset
    df = load_rtp_dataset()
    
    # Augment dataset
    augmenter = DataAugmenter()
    augmented_df = augmenter.augment_dataset(df.head(50))  # Use subset for demo
    
    # Save augmented dataset
    augmented_df.to_csv("data/augmented/augmented_rtp.csv", index=False)
    print(f"Augmented dataset saved with {len(augmented_df)} samples")