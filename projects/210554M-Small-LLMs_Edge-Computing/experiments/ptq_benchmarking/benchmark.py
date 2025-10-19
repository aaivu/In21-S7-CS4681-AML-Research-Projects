# ptq_qwen_bnb.py
"""
Post-Training Quantization (8-bit) for Qwen-0.5B using bitsandbytes.
Loads quantized model, runs evaluation (perplexity/BoolQ/SQuAD/latency),
and writes a CSV line with the quantized metrics to results CSV.
"""

import os
import time
import math
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from benchmark_utils import (
    choose_device, compute_perplexity, evaluate_boolq,
    evaluate_squad_prompt, get_model_size_gb, _prepare_inputs_for_prompt
)
# NOTE: _prepare_inputs_for_prompt used if you want Gemma-style chat inputs;
# for Qwen we use normal tokenize.

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = "../../results/ptq_benchmarking_results"
CSV_FILENAME = "ptq_results_cpu_4bit.csv"   # same pattern as your pipeline
CSV_PATH = os.path.join(OUTPUT_DIR, CSV_FILENAME)

# PTQ parameters (bitsandbytes style)
DEVICE = choose_device("cpu")
N_RUNS = 3
N_TOKENS = 50
PPL_SAMPLES = 2
BOOLQ_SAMPLES = 2
SQUAD_SAMPLES = 2

def measure_latency_bnb(model, tokenizer, prompt="The disaster response team should", n_tokens=50, n_runs=3):
    model_device = next(model.parameters()).device
    # prepare tokens
    inputs = tokenizer(prompt, return_tensors="pt").to(model_device)
    latencies = []
    for _ in range(max(1, n_runs)):
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=inputs["input_ids"].shape[1] + n_tokens)
        end = time.time()
        n_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
        if n_generated <= 0:
            latencies.append(0.0)
        else:
            latencies.append((end - start) / n_generated)
    return float(sum(latencies) / len(latencies) * 1000.0)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading Qwen in 8-bit on device {DEVICE} (this uses bitsandbytes)...")
    # load tokenizer normally
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # 4-bit config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",       # NormalFloat4 usually best
        bnb_4bit_use_double_quant=True,  # double quantization reduces memory
        bnb_4bit_compute_dtype=torch.float16,  # safer than bfloat16 on CPU
    )

    # load_in_8bit=True uses bitsandbytes 8-bit weights (PTQ style)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        # load_in_8bit=False,
        quantization_config=bnb_config,
        device_map="cpu"   # auto maps layers to devices (GPU/CPU); change as needed
    )

    model.eval()

    # size: note bitsandbytes weights still appear as normal model params on save/load,
    # but this gives an estimate of param memory in host mem (FP32 view). We still report it.
    size_gb = get_model_size_gb(model)

    # latency
    latency_ms = measure_latency_bnb(model, tokenizer, n_tokens=N_TOKENS, n_runs=N_RUNS)

    # Perplexity (use causal perplexity function from benchmark_utils)
    wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ppl = compute_perplexity(model, tokenizer, DEVICE, wikitext, max_samples=PPL_SAMPLES)

    # BoolQ
    boolq_acc = evaluate_boolq(model, tokenizer, DEVICE, max_samples=BOOLQ_SAMPLES, gemma=False)

    # SQuAD EM/F1
    em, f1 = evaluate_squad_prompt(model, tokenizer, DEVICE, max_samples=SQUAD_SAMPLES, gemma=False)

    result = {
        "Model": MODEL_NAME + " (8bit-bnb)",
        "Size (GB)": round(size_gb, 2),
        "Latency (ms/token)": round(latency_ms, 2),
        "Perplexity (WikiText-2)": round(ppl, 2),
        "BoolQ Acc (%)": round(boolq_acc, 2),
        "SQuAD EM (%)": round(em, 2),
        "SQuAD F1 (%)": round(f1, 2)
    }

    # Append to CSV (create if not exists)
    if os.path.exists(CSV_PATH):
        df_existing = pd.read_csv(CSV_PATH)
        df_combined = pd.concat([df_existing, pd.DataFrame([result])], ignore_index=True)
        df_combined.to_csv(CSV_PATH, index=False)
        print(f"Appended quantized result to {CSV_PATH}")
    else:
        pd.DataFrame([result]).to_csv(CSV_PATH, index=False)
        print(f"Saved quantized result to {CSV_PATH}")

    print("Result:", result)

if __name__ == "__main__":
    main()