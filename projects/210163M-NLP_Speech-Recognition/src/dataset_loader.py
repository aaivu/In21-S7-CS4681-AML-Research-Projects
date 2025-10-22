# Utilities to load LibriSpeech splits using Hugging Face `datasets`.
# Also supports reading CSV manifest files with columns: "path","text"

import os
from datasets import load_dataset, Audio, Dataset, DatasetDict
import pandas as pd

def load_librispeech_hf(split="train.clean.100"):
    """
    Loads LibriSpeech split via Hugging Face datasets.
    Example splits: "train.clean.100", "validation.clean", "test.clean"
    """
    ds = load_dataset("librispeech_asr", "clean", split=split)
    # ensure audio streaming is set to native sampling rate
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds

def load_manifest_csv(manifest_path):
    """
    Loads a CSV manifest with columns ['path','text'] into HF Dataset.
    Use this if you created your own manifest files.
    """
    df = pd.read_csv(manifest_path)
    # ensure columns
    assert "path" in df.columns and "text" in df.columns, "Manifest must contain 'path' and 'text' columns"
    dataset = Dataset.from_pandas(df[["path","text"]])
    # convert to a format similar to LibriSpeech HF dataset
    dataset = dataset.rename_column("path", "audio_filepath")
    return dataset

if __name__ == "__main__":
    # quick CLI demo
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation.clean")
    args = parser.parse_args()
    ds = load_librispeech_hf(args.split)
    print(ds[0])
