"""
Data Loading and Preprocessing for Jigsaw Unintended Bias Dataset
"""

import re
import os
import unicodedata
from pathlib import Path
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

# Contractions dictionary for expansion
CONTRACTIONS = {
    "ain't": "am not", "aren't": "are not", "can't": "cannot", 
    "can't've": "cannot have", "could've": "could have", "couldn't": "could not",
    "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not",
    "don't": "do not", "hadn't": "had not", "hadn't've": "had not have",
    "hasn't": "has not", "haven't": "have not", "he'd": "he would",
    "he'd've": "he would have", "he'll": "he will", "he's": "he is",
    "how'd": "how did", "how'll": "how will", "how's": "how is",
    "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
    "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
    "it'd've": "it would have", "it'll": "it will", "it's": "it is",
    "let's": "let us", "ma'am": "madam", "might've": "might have",
    "mightn't": "might not", "must've": "must have", "mustn't": "must not",
    "needn't": "need not", "shan't": "shall not", "she'd": "she would",
    "she'd've": "she would have", "she'll": "she will", "she's": "she is",
    "should've": "should have", "shouldn't": "should not", "that'd": "that would",
    "that's": "that is", "there'd": "there would", "there's": "there is",
    "they'd": "they would", "they'll": "they will", "they're": "they are",
    "they've": "they have", "wasn't": "was not", "we'd": "we would",
    "we'll": "we will", "we're": "we are", "we've": "we have",
    "weren't": "were not", "what'll": "what will", "what're": "what are",
    "what's": "what is", "what've": "what have", "when's": "when is",
    "where'd": "where did", "where's": "where is", "who'll": "who will",
    "who's": "who is", "won't": "will not", "wouldn't": "would not",
    "you'd": "you would", "you'll": "you will", "you're": "you are",
    "you've": "you have"
}


def load_jigsaw_data():
    """
    Load the Jigsaw Unintended Bias in Toxicity Classification dataset
    from Kaggle competition.
    
    Prerequisites:
    1. Install kaggle: pip install kaggle
    2. Get API credentials from Kaggle (Account → API → Create New Token)
    3. Place kaggle.json in ~/.kaggle/ (or upload in Colab)
    4. Accept competition rules at:
       https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/rules
    
    Returns:
        dataset: HuggingFace DatasetDict with train and validation splits
    """
    import subprocess
    import zipfile
    from pathlib import Path
    from sklearn.model_selection import train_test_split
    from datasets import Dataset, DatasetDict
    
    print("="*80)
    print("Loading Jigsaw Dataset from Kaggle")
    print("="*80)
    
    # Define paths
    data_dir = Path('/content/jigsaw') if 'COLAB_GPU' in os.environ or 'google.colab' in str(os.getcwd()) else Path('./jigsaw_data')
    kaggle_dir = Path.home() / '.kaggle'
    train_csv = data_dir / 'train.csv'
    
    # Check if data already exists
    if train_csv.exists():
        print(f"✓ Data already exists at {data_dir}")
    else:
        # Setup Kaggle credentials
        print("\nSetting up Kaggle credentials...")
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        kaggle_json = kaggle_dir / 'kaggle.json'
        
        if not kaggle_json.exists():
            print("\n" + "!"*80)
            print("KAGGLE CREDENTIALS NOT FOUND")
            print("!"*80)
            print("\nPlease set up Kaggle credentials:")
            print("1. Go to: https://www.kaggle.com/account")
            print("2. Click 'Create New API Token' to download kaggle.json")
            
            # Check if in Colab
            try:
                from google.colab import files
                print("\n3. Running in Colab - Upload your kaggle.json file:")
                uploaded = files.upload()
                
                if 'kaggle.json' in uploaded:
                    with open(kaggle_json, 'wb') as f:
                        f.write(uploaded['kaggle.json'])
                    os.chmod(kaggle_json, 0o600)
                    print("✓ Kaggle credentials uploaded successfully")
                else:
                    raise FileNotFoundError("kaggle.json not uploaded")
            except ImportError:
                print("\n3. Place kaggle.json in:", kaggle_dir)
                print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
                raise FileNotFoundError(f"Please place kaggle.json in {kaggle_dir}")
        else:
            print(f"✓ Kaggle credentials found at {kaggle_json}")
            os.chmod(kaggle_json, 0o600)
        
        # Install kaggle package if needed
        try:
            import kaggle
        except ImportError:
            print("\nInstalling kaggle package...")
            subprocess.check_call(['pip', 'install', '-q', 'kaggle'])
            import kaggle
        
        # Download dataset
        print(f"\nDownloading dataset to {data_dir}...")
        print("(This may take 5-10 minutes for ~1.5GB download)")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            subprocess.run([
                'kaggle', 'competitions', 'download', 
                '-c', 'jigsaw-unintended-bias-in-toxicity-classification',
                '-p', str(data_dir)
            ], check=True, capture_output=True, text=True)
            print("✓ Download complete")
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Download failed: {e.stderr}")
            print("\nMake sure you have:")
            print("1. Accepted the competition rules at:")
            print("   https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/rules")
            print("2. Valid Kaggle API credentials")
            raise
        
        # Unzip files
        print("\nExtracting files...")
        zip_files = list(data_dir.glob('*.zip'))
        for zip_file in zip_files:
            print(f"  Extracting {zip_file.name}...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            zip_file.unlink()  # Remove zip after extraction
        print("✓ Extraction complete")
    
    # Load data
    print("\nLoading data into pandas...")
    if not train_csv.exists():
        raise FileNotFoundError(f"train.csv not found in {data_dir}. Download may have failed.")
    
    train_full = pd.read_csv(train_csv)
    print(f"✓ Loaded {len(train_full):,} training samples")
    
    # Create validation split (5% of data)
    print("\nCreating train/validation split (95%/5%)...")
    train_df, val_df = train_test_split(
        train_full,
        test_size=0.05,
        random_state=42,
        stratify=(train_full['target'] >= 0.5).astype(int)  # Stratify by toxic/non-toxic
    )
    
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Validation: {len(val_df):,} samples")
    
    # Convert to HuggingFace Dataset format for compatibility
    print("\nConverting to HuggingFace Dataset format...")
    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df.reset_index(drop=True), preserve_index=False),
        'validation': Dataset.from_pandas(val_df.reset_index(drop=True), preserve_index=False)
    })
    
    print("\n✓ Dataset loaded successfully!")
    print("="*80)
    
    return dataset


def preprocess_text(text):
    """
    Comprehensive text preprocessing pipeline for toxicity classification.
    
    Steps:
    1. Remove HTML tags
    2. Remove accented characters (normalize to ASCII)
    3. Convert to lowercase
    4. Remove IP addresses, URLs, and numbers
    5. Expand contractions
    6. Remove special characters (preserve . ? !)
    7. Add spaces around preserved punctuation and remove duplicates
    
    Args:
        text (str): Raw input text
        
    Returns:
        str: Preprocessed text
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # 1. Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # 2. Remove accented characters (normalize to ASCII)
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    # 3. Convert to lowercase
    text = text.lower()
    
    # 4. Remove IP addresses, URLs, and numbers
    # IP addresses
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '', text)
    # URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Numbers
    text = re.sub(r'\b\d+\b', '', text)
    
    # 5. Expand contractions
    for contraction, expansion in CONTRACTIONS.items():
        text = re.sub(r'\b' + contraction + r'\b', expansion, text)
    
    # 6. Remove special characters but preserve . ? !
    text = re.sub(r'[^a-z\s.?!]', ' ', text)
    
    # 7. Add single space around preserved punctuation and remove duplicates
    # First, replace multiple occurrences of punctuation with single one
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r'!{2,}', '!', text)
    
    # Add spaces around punctuation
    text = re.sub(r'([.?!])', r' \1 ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def main():
    """
    Main execution: Load data, preprocess, and save results.
    """
    # Load dataset
    dataset = load_jigsaw_data()
    
    # Convert to pandas for easier manipulation
    print("\nPreprocessing train split...")
    train_df = pd.DataFrame(dataset['train'])
    
    # Apply preprocessing with progress bar
    tqdm.pandas(desc="Processing train texts")
    train_df['comment_text_cleaned'] = train_df['comment_text'].progress_apply(preprocess_text)
    
    print("\nPreprocessing validation split...")
    val_df = pd.DataFrame(dataset['validation'])
    
    tqdm.pandas(desc="Processing validation texts")
    val_df['comment_text_cleaned'] = val_df['comment_text'].progress_apply(preprocess_text)
    
    # Save preprocessed data
    print("\nSaving preprocessed data...")
    train_df.to_parquet('jigsaw_train_preprocessed.parquet', index=False)
    val_df.to_parquet('jigsaw_val_preprocessed.parquet', index=False)
    
    print("\nPreprocessing complete!")
    print(f"Train data saved to: jigsaw_train_preprocessed.parquet ({len(train_df):,} samples)")
    print(f"Validation data saved to: jigsaw_val_preprocessed.parquet ({len(val_df):,} samples)")
    
    # Show sample
    print("\n" + "="*80)
    print("Sample preprocessed examples:")
    print("="*80)
    for i in range(min(3, len(train_df))):
        print(f"\nOriginal: {train_df.iloc[i]['comment_text'][:150]}")
        print(f"Cleaned:  {train_df.iloc[i]['comment_text_cleaned'][:150]}")
        print("-"*80)
    
    # Show data statistics
    print("\n" + "="*80)
    print("Dataset Statistics:")
    print("="*80)
    print(f"\nToxicity distribution (train):")
    print(f"  Toxic (target >= 0.5): {(train_df['target'] >= 0.5).sum():,} ({(train_df['target'] >= 0.5).mean()*100:.2f}%)")
    print(f"  Non-toxic (target < 0.5): {(train_df['target'] < 0.5).sum():,} ({(train_df['target'] < 0.5).mean()*100:.2f}%)")
    print(f"\nTarget statistics:")
    print(train_df['target'].describe())


if __name__ == "__main__":
    main()