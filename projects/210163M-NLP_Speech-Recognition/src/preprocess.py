# Preprocessing: normalize text, convert to lowercase, map to labels, resample audio if needed.
# Creates processed HF dataset with 'input_values' and 'labels' when run once.

import argparse
import os
import json
from datasets import Audio
from dataset_loader import load_librispeech_hf
from model import Model
import re
from tqdm import tqdm

def normalize_text(text):
    text = text.lower()
    # keep a-z, space, apostrophe
    text = re.sub(r"[^a-z' ]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def prepare_dataset(processor_name, split, output_path, batch_size):
    asr = Model(processor_dir=processor_name)
    processor = asr.get_processor()
    ds = load_librispeech_hf(split=split)

    # map with batched=False to compute per-example
    ds_p = ds.map(asr.prepare, batched=True, batch_size=batch_size, remove_columns=ds.column_names)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ds_p.save_to_disk(output_path)
    print(f"Saved processed dataset to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processor_name", default="facebook/wav2vec2-base-960h")
    parser.add_argument("--split", default="train.clean.100")
    parser.add_argument("--output_path", default="../data/processed/train_clean_100")
    parser.add_argument("--batch", default=8)
    args = parser.parse_args()
    prepare_dataset(args.processor_name, args.split, args.output_path, args.batch)
