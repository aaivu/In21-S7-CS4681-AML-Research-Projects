# benchmark_utils.py
import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from collections import Counter
from tqdm import tqdm


# =========================================================
# QAT FakeQuant wrappers (used only during training)
# =========================================================
def _get_scale_zero_point(tensor, num_bits=8, symmetric=False):
    qmin, qmax = 0, 2 ** num_bits - 1
    min_val, max_val = float(tensor.min().detach().cpu()), float(tensor.max().detach().cpu())
    if symmetric:
        m = max(abs(min_val), abs(max_val))
        min_val, max_val = -m, m
    scale = (max_val - min_val) / float(qmax - qmin) if (max_val - min_val) != 0 else 1.0
    zero_point = round(qmin - min_val / scale) if scale != 0 else 0
    return float(scale), int(zero_point)

# --- FakeQuant helpers ---
class FakeQuantLinear(nn.Module):
    def __init__(self, linear_module, num_bits=8, fake_quant_activations=True):
        super().__init__()
        self.in_features = linear_module.in_features
        self.out_features = linear_module.out_features
        self.bias = linear_module.bias is not None
        self.weight = linear_module.weight
        self.bias_param = linear_module.bias
        self.num_bits = num_bits
        self.fake_quant_activations = fake_quant_activations

    def _get_scale_zero_point(self, tensor, num_bits=8, symmetric=False):
        qmin, qmax = 0, 2**num_bits - 1
        min_val = float(tensor.min().detach().cpu())
        max_val = float(tensor.max().detach().cpu())
        if symmetric:
            m = max(abs(min_val), abs(max_val))
            min_val, max_val = -m, m
        if max_val - min_val == 0:
            scale = 1.0
        else:
            scale = (max_val - min_val) / float(qmax - qmin)
        zero_point = round(qmin - min_val / scale) if scale != 0 else 0
        return float(scale), int(zero_point)

    def forward(self, input):
        # Fake quantize weights
        scale, zero_point = self._get_scale_zero_point(self.weight, self.num_bits)
        qmin, qmax = 0, 2**self.num_bits - 1
        fq_w = torch.fake_quantize_per_tensor_affine(
            self.weight, scale, zero_point, qmin, qmax
        )

        # Fake quantize activations if enabled
        if self.fake_quant_activations:
            a_scale, a_zero = self._get_scale_zero_point(input, self.num_bits)
            fq_input = torch.fake_quantize_per_tensor_affine(
                input, a_scale, a_zero, qmin, qmax
            )
            return nn.functional.linear(fq_input, fq_w, self.bias_param)
        else:
            return nn.functional.linear(input, fq_w, self.bias_param)


def replace_linear_with_fakequant(module, num_bits=8, fake_quant_activations=True):
    """
    Replace all nn.Linear layers with FakeQuantLinear wrappers.
    """
    replaced = 0
    for name, child in list(module.named_children()):
        if len(list(child.children())) > 0:
            replaced += replace_linear_with_fakequant(child, num_bits, fake_quant_activations)
        if isinstance(child, nn.Linear):
            setattr(module, name, FakeQuantLinear(child, num_bits, fake_quant_activations))
            replaced += 1
    return replaced


def revert_fakequant_to_linear(module):
    """
    After training, revert FakeQuantLinear wrappers back to plain nn.Linear
    so the checkpoint is saved with normal weights only.
    """
    reverted = 0
    for name, child in list(module.named_children()):
        if len(list(child.children())) > 0:
            reverted += revert_fakequant_to_linear(child)
        if isinstance(child, FakeQuantLinear):
            # Recreate nn.Linear with trained weights
            linear = nn.Linear(child.in_features, child.out_features, bias=child.bias)
            linear.weight = child.weight
            linear.bias = child.bias_param
            setattr(module, name, linear)
            reverted += 1
    return reverted

# Optional: Gemma model handling
try:
    from transformers import Gemma3ForCausalLM
    _GEMMA_AVAILABLE = True
except Exception:
    Gemma3ForCausalLM = None
    _GEMMA_AVAILABLE = False

# =========================================================
# DEVICE
# =========================================================
def choose_device(pref: str = "auto"):
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# NORMALIZATION + F1
# =========================================================
def normalize_text(s: str) -> str:
    import re, string
    if s is None:
        return ""
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())
    def remove_punc(text): return "".join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(a_gold: str, a_pred: str) -> float:
    gold_tokens = normalize_text(a_gold).split()
    pred_tokens = normalize_text(a_pred).split()
    common = Counter(gold_tokens) & Counter(pred_tokens)
    num_same = sum(common.values())
    if len(gold_tokens) == 0 or len(pred_tokens) == 0:
        return float(gold_tokens == pred_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

# =========================================================
# MODEL SIZE
# =========================================================
def get_model_size_gb(model: torch.nn.Module) -> float:
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    return param_size / (1024 ** 3)

# =========================================================
# PROMPT INPUTS
# =========================================================
def _prepare_inputs_for_prompt(tokenizer, prompt: str, device: torch.device, gemma: bool):
    if gemma:
        messages = [
            [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
        ]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
        return inputs
    else:
        return tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

# =========================================================
# LATENCY
# =========================================================
def measure_latency(model, tokenizer, device: torch.device,
                    prompt="The disaster response team should",
                    n_tokens=50, n_runs=3, gemma=False):
    inputs = _prepare_inputs_for_prompt(tokenizer, prompt, device, gemma)
    latencies = []
    for _ in range(max(1, int(n_runs))):
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=(inputs["input_ids"].shape[1] + n_tokens))
        end = time.time()
        n_gen = outputs.shape[1] - inputs["input_ids"].shape[1]
        latencies.append((end - start) / max(1, n_gen))
    return float(sum(latencies) / len(latencies) * 1000.0)  # ms/token

# =========================================================
# PERPLEXITY
# =========================================================
def compute_perplexity(model, tokenizer, device: torch.device, dataset, max_samples=200):
    total_loss = 0.0
    total_tokens = 0
    for i, text in enumerate(dataset["text"][:max_samples]):
        if not isinstance(text, str) or not text.strip():
            continue
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        n_tokens = inputs["input_ids"].numel()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
    if total_tokens == 0:
        return float("inf")
    avg_loss = total_loss / total_tokens
    return float(math.exp(avg_loss))

# =========================================================
# BOOLQ
# =========================================================
def evaluate_boolq(model, tokenizer, device: torch.device, max_samples=200, gemma=False):
    ds = load_dataset("boolq", split=f"validation[:{max_samples}]")
    correct = 0
    for ex in tqdm(ds, desc="BoolQ", leave=False):
        q, p, label = ex["question"], ex["passage"], ex["answer"]
        prompt = f"Passage: {p}\nQuestion: {q}\nAnswer yes or no:"
        inputs = _prepare_inputs_for_prompt(tokenizer, prompt, device, gemma)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=inputs["input_ids"].shape[1] + 5)
        pred_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).lower()
        pred = ("yes" in pred_text[:5])
        if pred == label:
            correct += 1
    return float(correct / len(ds) * 100.0)

# =========================================================
# SQUAD
# =========================================================
def evaluate_squad_prompt(model, tokenizer, device: torch.device, max_samples=200, gemma=False):
    ds = load_dataset("squad_v2", split=f"validation[:{max_samples}]")
    em_total, f1_total, n = 0, 0, len(ds)
    for ex in tqdm(ds, desc="SQuAD", leave=False):
        context, q, answers = ex["context"], ex["question"], ex["answers"]["text"]
        prompt = f"Context: {context}\nQuestion: {q}\nAnswer:"
        inputs = _prepare_inputs_for_prompt(tokenizer, prompt, device, gemma)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=inputs["input_ids"].shape[1] + 40)
        pred = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        em = 0
        f1 = 0.0
        if answers:
            em = 1 if any(normalize_text(pred) == normalize_text(a) for a in answers) else 0
            f1 = max(compute_f1(a, pred) for a in answers)
        else:  # unanswerable
            em = 1 if normalize_text(pred) == "" else 0
            f1 = 1.0 if em == 1 else 0.0
        em_total += em
        f1_total += f1
    return float(em_total / n * 100.0), float(f1_total / n * 100.0)

# =========================================================
# MODEL LOADER
# =========================================================
def load_model_and_tokenizer(model_name_or_path: str,
                             device: torch.device,
                             use_bnb_8bit=False,
                             bnb_device_map="auto"):
    gemma_flag = model_name_or_path.startswith("google/gemma")
    if gemma_flag:
        if not _GEMMA_AVAILABLE:
            raise RuntimeError("Gemma not supported in this environment.")
        model = Gemma3ForCausalLM.from_pretrained(model_name_or_path).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    else:
        if use_bnb_8bit:
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                quantization_config=bnb_cfg,
                device_map=bnb_device_map
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
            ).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    return model, tokenizer, gemma_flag

# =========================================================
# FULL EVALUATION
# =========================================================
def evaluate_loaded_model(model, tokenizer, device, cfg, model_label="Model"):
    lat_hp = cfg["hyperparams"]["latency"]
    ppl_hp = cfg["hyperparams"]["perplexity"]
    boolq_hp = cfg["hyperparams"]["boolq"]
    squad_hp = cfg["hyperparams"]["squad"]

    gemma_flag = isinstance(model, Gemma3ForCausalLM)

    size_gb = get_model_size_gb(model)
    latency = measure_latency(model, tokenizer, device,
                              prompt=lat_hp["prompt"],
                              n_tokens=lat_hp["n_tokens"],
                              n_runs=lat_hp["n_runs"],
                              gemma=gemma_flag)
    wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ppl = compute_perplexity(model, tokenizer, device, dataset=wikitext, max_samples=ppl_hp["max_samples"])
    boolq_acc = evaluate_boolq(model, tokenizer, device, max_samples=boolq_hp["max_samples"], gemma=gemma_flag)
    em, f1 = evaluate_squad_prompt(model, tokenizer, device, max_samples=squad_hp["max_samples"], gemma=gemma_flag)

    return {
        "Model": model_label,
        "Size (GB)": round(size_gb, 2),
        "Latency (ms/token)": round(latency, 2),
        "Perplexity (WikiText-2)": round(ppl, 2),
        "BoolQ Acc (%)": round(boolq_acc, 2),
        "SQuAD EM (%)": round(em, 2),
        "SQuAD F1 (%)": round(f1, 2),
    }