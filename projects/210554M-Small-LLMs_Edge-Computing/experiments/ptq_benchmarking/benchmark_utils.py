import os
import time
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from collections import Counter
from tqdm import tqdm

# Try to import Gemma class if available; else handle gracefully
try:
    from transformers import Gemma3ForCausalLM  # optional special class
    _GEMMA_AVAILABLE = True
except Exception:
    Gemma3ForCausalLM = None
    _GEMMA_AVAILABLE = False

# -----------------------------
# Device selection
# -----------------------------
def choose_device(pref: str = "auto"):
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Text normalization & F1
# -----------------------------
def normalize_text(s: str) -> str:
    import re, string
    if s is None:
        return ""
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text): return ''.join(ch for ch in text if ch not in set(string.punctuation))
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

# -----------------------------
# Model utilities
# -----------------------------
def get_model_size_gb(model: torch.nn.Module) -> float:
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    return param_size / (1024 ** 3)

def _prepare_inputs_for_prompt(tokenizer, prompt: str, device: torch.device, gemma: bool):
    """
    For gemma=True use chat template; otherwise simple tokenizer(prompt).
    Returns tensors on model device.
    """
    if gemma:
        # messages format expected by Gemma tokenizer apply_chat_template
        messages = [
            [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
        ]
        # apply_chat_template may require token/kwargs depending on Transformers version
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        # move to device
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
        return inputs
    else:
        return tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

# -----------------------------
# Latency measurement (gemma-aware)
# -----------------------------
def measure_latency(model, tokenizer, device: torch.device,
                    prompt: str = "The disaster response team should",
                    n_tokens: int = 50, n_runs: int = 3, gemma: bool = False):
    model_device = next(model.parameters()).device
    # prepare once
    inputs = _prepare_inputs_for_prompt(tokenizer, prompt, model_device, gemma=gemma)

    latencies = []
    for _ in range(max(1, int(n_runs))):
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=(inputs["input_ids"].shape[1] + n_tokens))
        end = time.time()
        n_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
        if n_generated <= 0:
            latencies.append(0.0)
        else:
            latencies.append((end - start) / n_generated)
    # ms per token
    return float(sum(latencies) / len(latencies) * 1000.0)

# -----------------------------
# Perplexity for causal LM (manual)
# -----------------------------
def compute_perplexity(model, tokenizer, device: torch.device, dataset, max_samples: int = 200):
    total_loss = 0.0
    total_tokens = 0
    model_device = next(model.parameters()).device

    for i, text in enumerate(dataset["text"][:max_samples]):
        if not isinstance(text, str) or not text.strip():
            continue
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model_device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        n_tokens = inputs["input_ids"].numel()
        total_loss += float(loss.item()) * n_tokens
        total_tokens += n_tokens
    if total_tokens == 0:
        return float("inf")
    avg_loss = total_loss / total_tokens
    return float(math.exp(avg_loss))

# -----------------------------
# BoolQ evaluation (prompt-based yes/no)
# -----------------------------
def evaluate_boolq(model, tokenizer, device: torch.device, max_samples: int = 200, gemma: bool = False):
    ds = load_dataset("boolq", split=f"validation[:{max_samples}]")
    correct = 0
    model_device = next(model.parameters()).device
    for ex in tqdm(ds, desc="BoolQ", leave=False):
        question = ex["question"]
        passage = ex["passage"]
        label = ex["answer"]  # bool
        prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer yes or no:"
        inputs = _prepare_inputs_for_prompt(tokenizer, prompt, model_device, gemma=gemma)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=inputs["input_ids"].shape[1] + 5)
        pred_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).lower()
        pred = ("yes" in pred_text[:5])
        if pred == label:
            correct += 1
    return float(correct / len(ds) * 100.0) if len(ds) > 0 else 0.0

# -----------------------------
# SQuAD prompt evaluation (EM + F1)
# -----------------------------
def evaluate_squad_prompt(model, tokenizer, device: torch.device, max_samples: int = 200, gemma: bool = False):
    ds = load_dataset("squad_v2", split=f"validation[:{max_samples}]")
    em_total = 0.0
    f1_total = 0.0
    model_device = next(model.parameters()).device
    n = len(ds)
    if n == 0:
        return 0.0, 0.0
    for ex in tqdm(ds, desc="SQuAD", leave=False):
        context = ex.get("context", "")
        question = ex.get("question", "")
        answers = ex.get("answers", {}).get("text", [])
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        inputs = _prepare_inputs_for_prompt(tokenizer, prompt, model_device, gemma=gemma)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=inputs["input_ids"].shape[1] + 40)
        pred = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        em_here = 0
        f1_here = 0.0
        if answers:
            em_here = 1 if any(normalize_text(pred) == normalize_text(a) for a in answers) else 0
            f1_here = max(compute_f1(a, pred) for a in answers)
        else:
            em_here = 1 if normalize_text(pred) == "" else 0
            f1_here = 1.0 if em_here == 1 else 0.0
        em_total += em_here
        f1_total += f1_here
    em_score = float(em_total / n * 100.0)
    f1_score = float(f1_total / n * 100.0)
    return em_score, f1_score

# -----------------------------
# Model loader and full evaluation wrapper (Gemma-aware)
# -----------------------------
def run_model_evaluation(model_name: str, cfg: dict):
    """
    Load model & tokenizer, run evaluations and return a dict with metrics.
    cfg is the entire config dict (used for hyperparams and device).
    """
    eval_cfg = cfg.get("evaluation", {})
    hp = cfg.get("hyperparams", {})

    device = choose_device(eval_cfg.get("device", "auto"))
    device_str = "cuda" if device.type == "cuda" else "cpu"

    lat_hp = hp.get("latency", {})
    ppl_hp = hp.get("perplexity", {})
    boolq_hp = hp.get("boolq", {})
    squad_hp = hp.get("squad", {})

    print(f"Loading model {model_name} on device {device_str} ...")

    gemma_flag = model_name.startswith("google/gemma")

    # load models
    try:
        if gemma_flag:
            if not _GEMMA_AVAILABLE:
                raise RuntimeError("Gemma model class not available in this Transformers build.")
            hf_token = os.environ.get("HF_TOKEN", None)
            model = Gemma3ForCausalLM.from_pretrained(
                model_name,
                token=hf_token,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
            ).to(device).eval()
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, use_fast=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
            ).to(device).eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load model/tokenizer for {model_name}: {e}")

    # measure & evaluate
    size_gb = get_model_size_gb(model)
    latency_ms = measure_latency(
        model, tokenizer, device,
        prompt=lat_hp.get("prompt", "The disaster response team should"),
        n_tokens=int(lat_hp.get("n_tokens", 50)),
        n_runs=int(lat_hp.get("n_runs", 3)),
        gemma=gemma_flag
    )

    # Perplexity
    wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ppl = compute_perplexity(
        model, tokenizer, device, dataset=wikitext,
        max_samples=int(ppl_hp.get("max_samples", 200))
    )

    # BoolQ
    boolq_acc = evaluate_boolq(
        model, tokenizer, device,
        max_samples=int(boolq_hp.get("max_samples", 200)),
        gemma=gemma_flag
    )

    # SQuAD
    em, f1 = evaluate_squad_prompt(
        model, tokenizer, device,
        max_samples=int(squad_hp.get("max_samples", 200)),
        gemma=gemma_flag
    )

    return {
        "Model": model_name,
        "Size (GB)": round(size_gb, 2),
        "Latency (ms_per_token)": round(latency_ms, 2),
        "Perplexity (WikiText-2)": round(ppl, 2),
        "BoolQ Acc (%)": round(boolq_acc, 2),
        "SQuAD EM (%)": round(em, 2),
        "SQuAD F1 (%)": round(f1, 2)
    }