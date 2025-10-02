import pandas as pd
import json
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class RTPProcessor:
    """Process RealToxicityPrompts dataset with attribute derivation."""
    
    def __init__(self):
        # Identity terms for sensitive attribute detection
        self.identity_terms = [
            'black', 'white', 'asian', 'latino', 'latina', 'hispanic', 
            'jewish', 'muslim', 'christian', 'catholic', 'buddhist', 'hindu',
            'gay', 'lesbian', 'transgender', 'lgbtq', 'queer',
            'woman', 'women', 'man', 'men', 'girl', 'boy',
            'african', 'american', 'european', 'mexican', 'arab'
        ]
        
        # Sarcasm/non-literal indicators for context attribute
        self.sarcasm_indicators = [
            'yeah right', 'sure thing', 'oh really', 'totally',
            'obviously', 'clearly', 'of course',  # can be sarcastic
            'lol', 'lmao', 'haha', 'smh',  # internet slang
            '...', '!!!', '!?',  # punctuation patterns
        ]
    
    def load_and_process_rtp(self, filepath: str) -> pd.DataFrame:
        """
        Load RTP dataset and process according to new methodology.
        
        Args:
            filepath: Path to RTP JSONL file
            
        Returns:
            Processed DataFrame with derived attributes
        """
        logger.info(f"Loading RTP dataset from {filepath}")
        
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    item = json.loads(line.strip())
                    processed = self._process_rtp_item(item)
                    if processed:
                        data.append(processed)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed line {line_num}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
                    continue
        
        df = pd.DataFrame(data)
        logger.info(f"Processed {len(df)} samples from RTP dataset")
        
        return df
    
    def _process_rtp_item(self, item: Dict) -> Dict:
        """Process a single RTP item."""
        
        # Extract prompt and continuation structures
        prompt = item.get('prompt', {})
        continuation = item.get('continuation', {})
        
        # Get texts
        if isinstance(prompt, dict):
            prompt_text = prompt.get('text', '')
        else:
            prompt_text = str(prompt)
            
        if isinstance(continuation, dict):
            continuation_text = continuation.get('text', '')
            # Get toxicity score for continuation
            continuation_toxicity = continuation.get('toxicity', 0.0)
        else:
            continuation_text = str(continuation)
            continuation_toxicity = 0.0
        
        # Skip if either is empty
        if not prompt_text or not continuation_text:
            return None
        
        # Concatenated text for attribute derivation
        full_text = f"{prompt_text} {continuation_text}"
        
        # Derive sensitive attribute (sens)
        # Mark sens=1 if identity terms found in prompt OR continuation
        sens = self._derive_sensitive_attribute(full_text)
        
        # Derive context attribute (ctx)
        # Mark ctx=1 if continuation shows sarcasm/non-literal patterns
        ctx = self._derive_context_attribute(continuation_text)
        
        return {
            'prompt_text': prompt_text.strip(),
            'continuation_text': continuation_text.strip(),
            'toxicity': continuation_toxicity,  # Target: continuation toxicity
            'sensitive_attribute': sens,  # For fairness loss
            'context_attribute': ctx,  # For context loss
            'full_text': full_text.strip()  # For reference
        }
    
    def _derive_sensitive_attribute(self, text: str) -> int:
        """
        Derive sensitive attribute from text.
        Returns 1 if identity terms found, 0 otherwise.
        """
        text_lower = text.lower()
        
        # Check for identity terms
        for term in self.identity_terms:
            if term in text_lower:
                return 1
        
        return 0
    
    def _derive_context_attribute(self, continuation_text: str) -> int:
        """
        Derive context attribute from continuation.
        Returns 1 if sarcasm/non-literal indicators found, 0 otherwise.
        """
        text_lower = continuation_text.lower()
        
        # Check for sarcasm indicators
        for indicator in self.sarcasm_indicators:
            if indicator in text_lower:
                return 1
        
        # Check for excessive punctuation
        if text_lower.count('!') >= 3 or text_lower.count('?') >= 3:
            return 1
        
        # Check for quotation marks (often used in sarcasm)
        if '"' in continuation_text and continuation_text.count('"') >= 2:
            return 1
        
        return 0
    
    def create_train_val_split(self, df: pd.DataFrame, 
                               test_size: float = 0.2,
                               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and validation sets."""
        from sklearn.model_selection import train_test_split
        
        # Stratify by toxicity binary to maintain distribution
        toxicity_binary = (df['toxicity'] >= 0.5).astype(int)
        
        train_df, val_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=toxicity_binary
        )
        
        logger.info(f"Train set: {len(train_df)} samples")
        logger.info(f"Val set: {len(val_df)} samples")
        logger.info(f"Train sensitive ratio: {train_df['sensitive_attribute'].mean():.3f}")
        logger.info(f"Train context ratio: {train_df['context_attribute'].mean():.3f}")
        
        return train_df, val_df