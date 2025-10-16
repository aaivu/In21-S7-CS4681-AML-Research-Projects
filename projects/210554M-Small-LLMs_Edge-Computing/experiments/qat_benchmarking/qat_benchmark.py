# qat_benchmark.py
import os
import pandas as pd
import torch
from transformers import BitsAndBytesConfig
import benchmark_utils as bu

# -----------------------------
# SETTINGS
# -----------------------------
QAT_FP_DIR = "qat_checkpoints/qwen_qat_fp"
RESULTS_CSV = "../../results/qat_benchmarking_results/qat_results.csv"
DEVICE = bu.choose_device("auto")
USE_BNB = True
BNB_DEVICE_MAP = "auto"

CFG = {
    "hyperparams": {
        "latency": {"prompt": "The disaster response team should", "n_tokens": 50, "n_runs": 3},
        "perplexity": {"max_samples": 10},
        "boolq": {"max_samples": 10},
        "squad": {"max_samples": 10}
    }
}

# -----------------------------
def main():
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    results = []

    # 1) Baseline FP model (hub)
    # try:
    #     model, tok, _ = bu.load_model_and_tokenizer("Qwen/Qwen2.5-0.5B", DEVICE, use_bnb_8bit=False)
    #     res = bu.evaluate_loaded_model(model, tok, DEVICE, CFG, model_label="Qwen (hub FP)")
    #     results.append(res)
    #     del model; torch.cuda.empty_cache()
    # except Exception as e:
    #     print("Baseline FP eval failed:", e)

    # 2) PTQ (hub in 8-bit)
    # if USE_BNB:
    #     try:
    #         model, tok, _ = bu.load_model_and_tokenizer("Qwen/Qwen2.5-0.5B", DEVICE, use_bnb_8bit=True, bnb_device_map=BNB_DEVICE_MAP)
    #         res = bu.evaluate_loaded_model(model, tok, DEVICE, CFG, model_label="Qwen (hub 8bit)")
    #         results.append(res)
    #         del model; torch.cuda.empty_cache()
    #     except Exception as e:
    #         print("Hub 8bit eval failed:", e)

    # 3) QAT-trained checkpoint, loaded in 8-bit runtime
    if os.path.isdir(QAT_FP_DIR):
        try:
            model, tok, _ = bu.load_model_and_tokenizer(QAT_FP_DIR, DEVICE, use_bnb_8bit=True, bnb_device_map=BNB_DEVICE_MAP)
            res = bu.evaluate_loaded_model(model, tok, DEVICE, CFG, model_label="Qwen-QAT (8bit-runtime)")
            results.append(res)
            del model; torch.cuda.empty_cache()
        except Exception as e:
            print("QAT 8bit eval failed:", e)
    else:
        print("QAT FP checkpoint not found; run qat_train.py first.")

    # Save results
    df = pd.DataFrame(results)
    if os.path.exists(RESULTS_CSV):
        df_existing = pd.read_csv(RESULTS_CSV)
        df_combined = pd.concat([df_existing, df], ignore_index=True)
        df_combined.to_csv(RESULTS_CSV, index=False)
    else:
        df.to_csv(RESULTS_CSV, index=False)
    print("Results saved to", RESULTS_CSV)

if __name__ == "__main__":
    main()
