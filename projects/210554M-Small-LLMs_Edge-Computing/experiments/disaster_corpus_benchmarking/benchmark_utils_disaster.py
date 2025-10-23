# benchmark_utils_disaster.py
import os
import time
import math
import json
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --------------------------
# Device
# --------------------------
def choose_device(pref="auto"):
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# Fake Quantization Wrapper
# --------------------------
class FakeQuantLinear(nn.Module):
    def __init__(self, linear, num_bits=8, fake_quant_activations=True):
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.num_bits = num_bits
        self.fake_quant_activations = fake_quant_activations

    def _scale_zp(self, tensor):
        qmin, qmax = 0, 2**self.num_bits - 1
        min_val, max_val = float(tensor.min()), float(tensor.max())
        scale = (max_val - min_val) / (qmax - qmin) if max_val != min_val else 1.0
        zp = qmin - min_val / scale if scale != 0 else 0
        return scale, int(zp)

    def forward(self, x):
        s_w, zp_w = self._scale_zp(self.weight)
        fq_w = torch.fake_quantize_per_tensor_affine(self.weight, s_w, zp_w, 0, 2**self.num_bits - 1)
        if self.fake_quant_activations:
            s_x, zp_x = self._scale_zp(x)
            fq_x = torch.fake_quantize_per_tensor_affine(x, s_x, zp_x, 0, 2**self.num_bits - 1)
        else:
            fq_x = x
        return nn.functional.linear(fq_x, fq_w, self.bias)

def replace_linear_with_fakequant(module, num_bits=8, fake_quant_activations=True):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, FakeQuantLinear(child, num_bits, fake_quant_activations))
        else:
            replace_linear_with_fakequant(child, num_bits, fake_quant_activations)

def revert_fakequant_to_linear(module):
    for name, child in list(module.named_children()):
        if isinstance(child, FakeQuantLinear):
            linear = nn.Linear(child.in_features, child.out_features, bias=child.bias is not None)
            linear.weight = child.weight
            linear.bias = child.bias
            setattr(module, name, linear)
        else:
            revert_fakequant_to_linear(child)

# --------------------------
# Dataset Loader
# --------------------------
def load_disaster_dataset(json_path, test_split=0.2):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [d["chunk"] for d in data if "chunk" in d and d["chunk"].strip()]
    random.shuffle(texts)
    n_test = int(len(texts) * test_split)
    return texts[n_test:], texts[:n_test]  # train, test

# --------------------------
# Perplexity
# --------------------------
def compute_perplexity(model, tokenizer, device, texts, max_samples=200):
    """
    Compute perplexity safely across device configurations (CPU/CUDA/mixed).
    Automatically aligns inputs to model parameter device.
    """
    total_loss, total_tokens = 0.0, 0
    samples = texts[:max_samples]

    # detect model base device (works for quantized + normal)
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = device

    for text in tqdm(samples, desc="Perplexity", leave=False):
        if not isinstance(text, str) or not text.strip():
            continue

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        # move all inputs to the same device as model
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

        total_loss += loss.item() * inputs["input_ids"].numel()
        total_tokens += inputs["input_ids"].numel()

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    return float(math.exp(avg_loss))

# --------------------------
# Latency
# --------------------------
def measure_latency(model, tokenizer, device, prompt="Flood risk assessment is", n_tokens=50, n_runs=3):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    latencies = []
    for _ in range(n_runs):
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=inputs["input_ids"].shape[1] + n_tokens)
        end = time.time()
        latencies.append((end - start) / n_tokens)
    return sum(latencies) / len(latencies) * 1000.0  # ms/token

# --------------------------
# Size
# --------------------------
def get_model_size_gb(model):
    params = sum(p.nelement() * p.element_size() for p in model.parameters())
    return params / (1024 ** 3)

# --------------------------
# Plot Generator
# --------------------------
def generate_plots(csv_path, save_path="results/comparison_plots.png"):
    import pandas as pd
    df = pd.read_csv(csv_path)
    metrics = ["Perplexity", "Latency (ms/token)"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, metrics):
        df.plot.bar(x="Model", y=metric, ax=ax, legend=False)
        ax.set_title(metric)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Saved plots to {save_path}")