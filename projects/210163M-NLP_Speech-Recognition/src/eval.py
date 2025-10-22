# evaluate.py
# Load a trained checkpoint and run evaluation on test-clean split, printing WER/CER.

import argparse
import os
from transformers import AutoProcessor, AutoModelForCTC
from dataset_loader import load_librispeech_hf
import numpy as np
import torch
from utils import save_json, get_wer_cer

def predict_batch(batch, processor, model, device):
    inputs = processor(batch["audio"]["array"], sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits
    pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
    preds = processor.batch_decode(pred_ids)
    return preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--split", default="test.clean")
    args = parser.parse_args()

    asr = Model(model_dir=args.model_dir, processor_dir=args.model_dir)
    processor = asr.get_processor()
    model = asr.get_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = lload_librispeech_hf(args.split)
    ds = ds.map(asr.predict_logits, batched=True, batch_size=8, remove_columns=["audio"])
    predictions = [np.array(l) for l in ds["logits"]]
    w,c = get_wer_cer(ds["text"], predictions)
    results = {"wer": w, "cer": c}
    save_json(os.path.join(args.model_dir, "eval_results.json"), results)

if __name__ == "__main__":
    main()
