"""
Inference script for QAT fine-tuned Qwen2.5-0.5B on disaster-domain dataset.
Loads the saved checkpoint and performs text generation.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

MODEL_PATH = "model_checkpoints/resqedge"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_text(prompt, max_new_tokens=100):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    end = time.time()

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    latency = (end - start) / max_new_tokens * 1000
    print(f"⏱ Avg latency: {latency:.2f} ms/token\n")
    print("=== Generated Text ===")
    print(text)


if __name__ == "__main__":
    while True:
        prompt = input("Enter a prompt (or 'exit' to quit): ")
        if prompt.lower() == "exit":
            break
        generate_text(prompt)