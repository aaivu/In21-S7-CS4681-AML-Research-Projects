import pandas as pd
import numpy as np
import os
import logging
from datasets import load_dataset

logger = logging.getLogger(__name__)

class JigsawDataLoader:
    """Loader for Jigsaw toxicity datasets."""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir

    def load_jigsaw_dataset(dataset_name: str = "civil_comments", data_path: str = "data/raw") -> pd.DataFrame:
        """
        Load actual Jigsaw Unintended Bias in Toxicity Classification dataset.
        
        Args:
            dataset_name: Name of the Jigsaw dataset to load
            data_path: Directory to save/load the dataset
            
        Returns:
            DataFrame with Jigsaw data
        """
        os.makedirs(data_path, exist_ok=True)
        
        # Method 1: Try loading from Hugging Face datasets
        try:
            logger.info(f"Loading Jigsaw {dataset_name} dataset from Hugging Face...")
            
            if dataset_name == "civil_comments":
                # Load the civil comments dataset
                dataset = load_dataset("civil_comments", split="train")
                df = dataset.to_pandas()
                
            elif dataset_name == "jigsaw_toxicity_pred":
                # Alternative Jigsaw dataset
                dataset = load_dataset("jigsaw_toxicity_pred", split="train")  
                df = dataset.to_pandas()
                
            else:
                # Try generic approach
                dataset = load_dataset(dataset_name, split="train")
                df = dataset.to_pandas()
            
            # Standardize the dataset
            df = _standardize_jigsaw_columns(df)
            
            # Save for future use
            csv_path = os.path.join(data_path, f"{dataset_name}.csv")
            df.to_csv(csv_path, index=False)
            
            logger.info(f"Jigsaw dataset loaded with {len(df)} samples")
            return df
            
        except Exception as e:
            logger.warning(f"Failed to load from Hugging Face: {e}")
        
        # Method 2: Try downloading from Kaggle (requires kaggle API)
        try:
            return _download_jigsaw_from_kaggle(dataset_name, data_path)
        except Exception as e:
            logger.warning(f"Failed to download from Kaggle: {e}")
        
        # Method 3: Check for local files
        local_files = [
            f"{dataset_name}.csv",
            "train.csv",
            "civil_comments.csv",
            "jigsaw-unintended-bias-train.csv"
        ]
        
        for filename in local_files:
            filepath = os.path.join(data_path, filename)
            if os.path.exists(filepath):
                logger.info(f"Loading Jigsaw dataset from local file: {filepath}")
                df = pd.read_csv(filepath)
                return _standardize_jigsaw_columns(df)
        
        # Fallback: Print manual instructions
        logger.error("Could not automatically load Jigsaw dataset")
        _print_jigsaw_manual_instructions(data_path, dataset_name)
        
        return pd.DataFrame()
        
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
    
def _standardize_jigsaw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize Jigsaw dataset columns."""

    # Map common column variations
    column_mappings = {
        'comment_text': 'comment_text',
        'text': 'comment_text', 
        'comment': 'comment_text',
        'target': 'toxicity',
        'toxic': 'toxicity',
        'toxicity_score': 'toxicity'
    }

    # Apply mappings
    for old_col, new_col in column_mappings.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]

    # Ensure required columns exist
    if 'comment_text' not in df.columns:
        if 'text' in df.columns:
            df['comment_text'] = df['text']
        else:
            logger.error("No text column found in Jigsaw dataset")
            df['comment_text'] = ''

    if 'toxicity' not in df.columns:
        if 'target' in df.columns:
            df['toxicity'] = df['target']
        else:
            logger.warning("No toxicity score found, setting to 0")
            df['toxicity'] = 0.0

    # Add other toxicity attributes if they exist
    toxicity_attrs = [
        'severe_toxicity', 'obscene', 'threat', 'insult', 
        'identity_attack', 'sexual_explicit'
    ]

    for attr in toxicity_attrs:
        if attr not in df.columns:
            df[attr] = 0.0

    # Identity mentions (look for identity columns or keywords)
    identity_columns = [col for col in df.columns if any(identity in col.lower() 
                    for identity in ['male', 'female', 'christian', 'muslim', 'jewish', 'lgbtq', 
                                    'black', 'white', 'psychiatric', 'disability'])]

    if identity_columns:
        # If identity columns exist, use them
        df['identity_mention'] = df[identity_columns].max(axis=1).fillna(0)
        df['identity_mention'] = (df['identity_mention'] > 0.5).astype(int)
    else:
        # Otherwise, search for identity keywords in text
        identity_keywords = [
            'black', 'white', 'asian', 'latino', 'hispanic', 'jewish', 'muslim', 
            'christian', 'gay', 'lesbian', 'transgender', 'lgbtq', 'men', 'women'
        ]
        
        df['identity_mention'] = df['comment_text'].apply(
            lambda text: int(any(keyword in str(text).lower() for keyword in identity_keywords))
        )

    # Context labels (sarcasm, non-literal)
    sarcasm_patterns = [
        'yeah right', 'sure thing', 'totally', 'obviously', 'clearly',
        '!!!', '??', 'lol', 'haha', 'smh'
    ]

    df['context_label'] = df['comment_text'].apply(
        lambda text: int(any(pattern in str(text).lower() for pattern in sarcasm_patterns))
    )

    # Sample dataset if too large (for manageable processing)
    if len(df) > 50000:
        logger.info(f"Sampling dataset from {len(df)} to 50000 samples")
        df = df.sample(n=50000, random_state=42).reset_index(drop=True)

    logger.info(f"Standardized Jigsaw dataset: {len(df)} samples")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Identity mentions: {df['identity_mention'].sum()}")
    logger.info(f"Context labels: {df['context_label'].sum()}")

    return df

def _download_jigsaw_from_kaggle(dataset_name: str, data_path: str) -> pd.DataFrame:
    """Download Jigsaw dataset from Kaggle."""

    try:
        import kaggle
        
        # Kaggle dataset identifiers
        kaggle_datasets = {
            "civil_comments": "jigsaw-team/jigsaw-unintended-bias-in-toxicity-classification",
            "toxic_comments": "c/jigsaw-toxic-comment-classification-challenge"
        }
        
        dataset_id = kaggle_datasets.get(dataset_name, kaggle_datasets["civil_comments"])
        
        logger.info(f"Downloading {dataset_name} from Kaggle: {dataset_id}")
        
        # Download dataset
        kaggle.api.dataset_download_files(dataset_id, path=data_path, unzip=True)
        
        # Find the downloaded CSV file
        csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
        
        if not csv_files:
            raise Exception("No CSV files found after Kaggle download")
        
        # Load the main training file
        train_files = [f for f in csv_files if 'train' in f.lower()]
        csv_file = train_files[0] if train_files else csv_files[0]
        
        df = pd.read_csv(os.path.join(data_path, csv_file))
        df = _standardize_jigsaw_columns(df)
        
        logger.info(f"Successfully downloaded Jigsaw dataset with {len(df)} samples")
        return df
        
    except Exception as e:
        logger.error(f"Kaggle download failed: {e}")
        raise

def _print_jigsaw_manual_instructions(data_path: str, dataset_name: str):
    """Print manual download instructions for Jigsaw dataset."""

    instructions = f"""

    ⚠️  MANUAL JIGSAW DATASET DOWNLOAD REQUIRED ⚠️

    The Jigsaw dataset could not be automatically downloaded.
    Please follow these steps:

    1. Go to Kaggle:
    - Civil Comments: https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification
    - Toxic Comments: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

    2. Download the training dataset (train.csv)
    3. Place it in: {data_path}/
    4. Rename to: {dataset_name}.csv
    5. Re-run your code

    Alternative: Set up Kaggle API
    - Install: pip install kaggle
    - Setup API key: https://github.com/Kaggle/kaggle-api
    - The code will automatically download

    """

    print(instructions)
    logger.error(instructions)