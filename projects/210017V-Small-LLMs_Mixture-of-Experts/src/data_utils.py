################################################################################
# Data Utilities for WikiText-2 Parquet Dataset
################################################################################
import torch
import pandas as pd
import pyarrow.parquet as pq
from collections import Counter

################################################################################
# Tokenization, Vocabulary, and Numericalization
################################################################################

def tokenize_line(line):
    """Simple whitespace tokenizer."""
    return line.strip().split()

def build_vocab(lines_iter, min_freq=1, special_tokens=['<pad>', '<unk>', '<eos>']):
    """
    Builds vocabulary with minimum frequency threshold.
    Returns: vocab, stoi, itos, unk_idx
    """
    counter = Counter()
    for line in lines_iter:
        toks = tokenize_line(line)
        counter.update(toks)
    it = (tok for tok, freq in counter.items() if freq >= min_freq)
    vocab = list(special_tokens) + sorted(it)
    stoi = {tok: i for i, tok in enumerate(vocab)}
    itos = {i: tok for tok, i in stoi.items()}
    unk_idx = stoi['<unk>']
    return vocab, stoi, itos, unk_idx

def numericalize(lines_iter, stoi, unk_idx):
    """Converts tokenized text lines into numeric IDs."""
    ids = []
    for line in lines_iter:
        toks = tokenize_line(line)
        ids.extend(
            [stoi.get(tok, unk_idx) for tok in toks] + 
            [stoi.get('<eos>', unk_idx)]
        )
    return torch.tensor(ids, dtype=torch.long)

################################################################################
# Batch Processing (for Language Modeling)
################################################################################

def batchify(data, batch_size, device):
    """
    Converts flat token IDs into [batch_size, num_cols] tensor for LM training.
    """
    n = data.size(0) // batch_size
    data = data[:n * batch_size]
    data = data.view(batch_size, n).contiguous().to(device)
    return data

def get_batch(source, i, seq_len):
    """
    Returns (data, target) batch pair for next-token prediction.
    """
    seq_len_eff = min(seq_len, source.size(1) - 1 - i)
    data = source[:, i:i + seq_len_eff]
    target = source[:, i + 1:i + 1 + seq_len_eff]
    return data, target

################################################################################
# Loading WikiText-2 Parquet Dataset
################################################################################

def load_wikitext2_parquet(dataset_path, batch_size=32, seq_len=35, min_freq=1, device='cpu'):
    """
    Loads WikiText-2 parquet dataset and returns processed train/val/test tensors.
    dataset_path should contain train-*.parquet, validation-*.parquet, test-*.parquet
    """
    # Read parquet files
    train_df = pq.read_table(f"{dataset_path}/train-00000-of-00001.parquet").to_pandas()
    val_df   = pq.read_table(f"{dataset_path}/validation-00000-of-00001.parquet").to_pandas()
    test_df  = pq.read_table(f"{dataset_path}/test-00000-of-00001.parquet").to_pandas()

    # Extract text lists
    train_lines = train_df["text"].tolist()
    val_lines   = val_df["text"].tolist()
    test_lines  = test_df["text"].tolist()

    # Build vocabulary
    vocab, stoi, itos, unk_idx = build_vocab(train_lines, min_freq=min_freq)

    # Convert to numeric tensors
    train_ids = numericalize(train_lines, stoi, unk_idx)
    val_ids   = numericalize(val_lines, stoi, unk_idx)
    test_ids  = numericalize(test_lines, stoi, unk_idx)

    # Batchify data
    train_data = batchify(train_ids, batch_size, device)
    val_data   = batchify(val_ids, batch_size, device)
    test_data  = batchify(test_ids, batch_size, device)

    return train_data, val_data, test_data, vocab, stoi, itos, get_batch
