# Decode model logits with pyctcdecode (KenLM) and print WER/CER on test split.
# Requires kenlm and pyctcdecode installed.

import argparse
import os
from model import Model
from dataset_loader import load_librispeech_hf
import torch
from pyctcdecode import build_ctcdecoder
import numpy as np
from tqdm import tqdm
from utils import save_json, get_wer_cer

def logits_to_texts(logits, decoder):
    """
    Convert raw logits (numpy) to text using pyctcdecode decoder (with LM).
    logits: ndarray (T, V)
    returns: string
    """
    return decoder.decode(logits)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--kenlm_path", required=True)
    parser.add_argument("--split", default="test.clean")
    parser.add_argument("--alpha", default=0.5)
    parser.add_argument("--beta", default=0.1)
    args = parser.parse_args()

    asr = Model(model_dir=args.model_dir, processor_dir=args.model_dir)
    processor = asr.get_processor()
    model = asr.get_model().eval()

    # Build decoder: labels must be list of str matching tokenizer
    # Note: processor.tokenizer.get_vocab() returns mapping token->id; we need labels in id order
    vocab = [x[0] for x in sorted(processor.tokenizer.get_vocab().items(), key=lambda item: item[1])]
    # Some tokenizers include special tokens; ensure blanks mapped correctly
    # pyctcdecode expects labels excluding CTC blank - pyctcdecode handles blank as last index by default
    decoder = build_ctcdecoder(
        labels=vocab,
        kenlm_model_path=args.kenlm_path,
        alpha=alpha,
        beta=beta
    )

    ds = load_librispeech_hf(args.split)
    ds = ds.map(asr.predict_logits, batched=True, batch_size=8, remove_columns=["audio"])
    logits = [np.array(l) for l in ds["logits"]]
    beam_lm = [decoder.decode(logit) for logit in logits]
    w,c = get_wer_cer(ds["text"], beam_lm)
    results = {"wer": w, "cer": c}
    print("KenLM decode results:", results)
    save_json(os.path.join(args.model_dir, "kenlm_decode_results.json"), results)

if __name__ == "__main__":
    main()
