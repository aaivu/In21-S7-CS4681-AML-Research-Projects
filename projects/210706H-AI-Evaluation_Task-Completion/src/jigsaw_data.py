import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class JigsawDataLoader:
    """Loader for Jigsaw toxicity datasets."""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        
    def load_jigsaw_dataset(self, dataset_name: str = "civil_comments") -> pd.DataFrame:
        """
        Load Jigsaw dataset. For demo purposes, creates a representative dataset.
        In production, replace with actual Jigsaw data loading.
        """
        filepath = os.path.join(self.data_dir, f"{dataset_name}.csv")
        
        if os.path.exists(filepath):
            logger.info(f"Loading existing {dataset_name} dataset...")
            return pd.read_csv(filepath)
        else:
            logger.info(f"Creating simulated {dataset_name} dataset...")
            return self._create_simulated_jigsaw_dataset()
    
    def _create_simulated_jigsaw_dataset(self) -> pd.DataFrame:
        """
        Create a simulated Jigsaw dataset that represents the key characteristics:
        - Identity mentions (for fairness evaluation)
        - Various toxicity levels
        - Context-sensitive examples (sarcasm, slang)
        """
        np.random.seed(42)
        
        # Base comment templates
        templates = [
            # Non-toxic comments
            "I really enjoyed this article about {}.",
            "The new movie was {} and entertaining.",
            "That restaurant serves {} food.",
            "The weather today is absolutely {}.",
            "This book about {} was fascinating.",
            
            # Potentially toxic (but context-dependent)
            "That's {} - totally unbelievable!",  # Could be sarcastic
            "Yeah right, {} makes perfect sense.",  # Sarcastic
            "This {} situation is crazy.",  # Slang usage
            "The {} policy is completely insane.",  # Strong language, not necessarily toxic
            
            # Identity-related (test fairness)
            "The {} community deserves better representation.",
            "As a {} person, I think this is important.",
            "The {} culture has rich traditions.",
            
            # Actually toxic examples
            "Those {} people are always causing trouble.",
            "I hate it when {} act like that.",
            "All {} are the same - completely worthless.",
        ]
        
        # Fill-in words
        positive_words = ["amazing", "wonderful", "great", "excellent", "fantastic"]
        negative_words = ["terrible", "awful", "bad", "horrible", "disappointing"]
        identity_groups = ["Black", "White", "Christian", "Muslim", "Jewish", "LGBTQ", "Latino", "Asian"]
        neutral_topics = ["technology", "education", "healthcare", "sports", "music"]
        
        data = []
        
        for i in range(2000):  # Create 2000 samples
            template = np.random.choice(templates)
            
            # Determine toxicity level based on template
            if "hate" in template or "worthless" in template or "causing trouble" in template:
                toxicity = np.random.uniform(0.7, 0.95)  # High toxicity
                word_choice = np.random.choice(identity_groups + negative_words)
                identity_mention = 1 if word_choice in identity_groups else 0
            elif "Yeah right" in template or "totally unbelievable" in template:
                toxicity = np.random.uniform(0.3, 0.6)  # Medium (context-dependent)
                word_choice = np.random.choice(neutral_topics + negative_words)
                identity_mention = 0
                context_label = 1  # Sarcastic/non-literal
            elif "crazy" in template or "insane" in template:
                toxicity = np.random.uniform(0.2, 0.5)  # Low-medium (slang)
                word_choice = np.random.choice(neutral_topics + negative_words)
                identity_mention = 0
            elif "community" in template or "culture" in template or "As a" in template:
                toxicity = np.random.uniform(0.05, 0.25)  # Low toxicity
                word_choice = np.random.choice(identity_groups)
                identity_mention = 1
            else:
                toxicity = np.random.uniform(0.01, 0.2)  # Very low toxicity
                word_choice = np.random.choice(positive_words + neutral_topics)
                identity_mention = 0
            
            comment_text = template.format(word_choice)
            
            data.append({
                'comment_text': comment_text,
                'toxicity': toxicity,
                'severe_toxicity': toxicity * np.random.uniform(0.3, 0.7),
                'identity_attack': toxicity * np.random.uniform(0.2, 0.8) if identity_mention else np.random.uniform(0.01, 0.1),
                'insult': toxicity * np.random.uniform(0.4, 0.9),
                'threat': toxicity * np.random.uniform(0.1, 0.4),
                'profanity': toxicity * np.random.uniform(0.2, 0.6),
                'identity_mention': identity_mention,
                'context_label': 1 if any(marker in template for marker in ["Yeah right", "totally", "!"]) else 0
            })
        
        df = pd.DataFrame(data)
        
        # Save for future use
        os.makedirs(self.data_dir, exist_ok=True)
        df.to_csv(os.path.join(self.data_dir, "civil_comments.csv"), index=False)
        
        logger.info(f"Created Jigsaw dataset with {len(df)} samples")
        return df