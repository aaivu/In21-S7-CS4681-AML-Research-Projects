# ptq_benchmark_disaster.py
"""
Post-Training Quantization (8-bit) benchmarking on disaster dataset.
Safe for both CPU and CUDA — avoids mixed-device issues.
"""

import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from benchmark_utils_disaster import (
    choose_device, load_disaster_dataset,
    compute_perplexity, measure_latency, get_model_size_gb
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = "../../results/disaster_corpus_benchmarking_results"
CSV_PATH = os.path.join(OUTPUT_DIR, "ptq_results.csv")
DATA_PATH = "../../data/disaster_corpus/flood.json"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------- Force CPU for consistency ----------
    device = choose_device("cpu")

    # ---------- Dataset ----------
    train_texts, test_texts = load_disaster_dataset(DATA_PATH)

    # ---------- Quantized Model ----------
    print(f"Loading {MODEL_NAME} in 8-bit (bitsandbytes) on CPU...")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": "cpu"}     # <--- force CPU to avoid cross-device issues
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model.eval()

    # ---------- Benchmark ----------
    ppl = compute_perplexity(model, tokenizer, device, test_texts)
    latency = measure_latency(model, tokenizer, device)
    size_gb = get_model_size_gb(model)

    result = {
        "Model": "Qwen2.5-0.5B (PTQ-8bit)",
        "Size (GB)": round(size_gb, 2),
        "Perplexity": round(ppl, 2),
        "Latency (ms/token)": round(latency, 2),
    }

    # ---------- Save Results ----------
    if os.path.exists(CSV_PATH):
        pd.concat([pd.read_csv(CSV_PATH), pd.DataFrame([result])], ignore_index=True).to_csv(CSV_PATH, index=False)
    else:
        pd.DataFrame([result]).to_csv(CSV_PATH, index=False)

    print("\n✅ PTQ Benchmark Complete:")
    print(result)

if __name__ == "__main__":
    main()