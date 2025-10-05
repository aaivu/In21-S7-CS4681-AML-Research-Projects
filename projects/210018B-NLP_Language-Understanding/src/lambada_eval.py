"""
LAMBADA Evaluation with GPT API (Optimized Prompt Engineering Version)
---------------------------------------------------------------------
Features:
- Strong instruction formatting
- Semantic few-shot selection (SentenceTransformers)
- Cloze-mode fill-in-the-blank style
- Multi-sampling (self-consistency)
- Clean example filtering
- Result logging for analysis
"""

from openai import OpenAI
from datasets import load_dataset
import random, re, csv, os, torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
load_dotenv()
API_KEY = os.getenv("API_KEY")
client = OpenAI(api_key=API_KEY)

device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device=device)

# Load dataset
dataset = load_dataset("lambada", split="test")
dev_data = load_dataset("lambada", split="validation")

# ---------------------------------------------------------------------
# Clean dev set: remove multiword or punctuation-heavy endings
# ---------------------------------------------------------------------
clean_dev = []
for ex in dev_data:
    text = ex["text"].strip()
    if " " not in text:
        continue
    ctx, gold = text.rsplit(" ", 1)
    if len(gold.split()) > 1 or not re.match(r"^[A-Za-z'-]+$", gold):
        continue
    clean_dev.append({"text": text})

print(f"Cleaned dev examples: {len(clean_dev)} / {len(dev_data)}")

# ---------------------------------------------------------------------
# Precompute embeddings for semantic similarity
# ---------------------------------------------------------------------
dev_texts = [ex["text"] for ex in clean_dev]
dev_embs = embedder.encode(dev_texts, convert_to_tensor=True, show_progress_bar=True)
print("Embeddings ready.")

# ---------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------
def build_prompt(context, k=5, mode="cloze", use_semantic=True):
    """
    Build a strong, instruction-driven few-shot prompt.
    """

    if mode not in ["cloze", "default"]:
        mode = "cloze"  # enforce clean structure

    # === Instructional prefix ===
    prefix = (
    "Below are examples where you must predict the final missing word.\n"
    "Each passage ends with a blank (_____), and the correct word follows after '→'.\n"
    "Your task: predict the final word for the last passage.\n"
    "Rules:\n"
    "- Output exactly ONE meaningful English word (noun, verb, or name).\n"
    "- DO NOT output continuation words like: I,he,the, a, an, and, of, to, in, on, or and any other of that type.\n"
    "- No punctuation or explanations.\n\n"
    )

    few_shot_examples = ""

    # === Few-shot example selection ===
    if k > 0:
        if use_semantic:
            context_emb = embedder.encode(context, convert_to_tensor=True)
            similarities = util.pytorch_cos_sim(context_emb, dev_embs)[0]
            top_indices = similarities.topk(k).indices.tolist()
            examples = [clean_dev[i] for i in top_indices]
        else:
            examples = random.sample(clean_dev, k)

        for ex in examples:
            text = ex["text"].strip()
            ctx, gold = text.rsplit(" ", 1)
            few_shot_examples += f"{ctx} _____ → {gold}\n\n"


    # === Format modes ===
    if mode == "cloze":
        prompt = (
        prefix
        + few_shot_examples
        + f"{context} _____ →"
    )

    else:
        prompt = prefix + few_shot_examples + "\nNow solve for the next passage.\n" + f"Passage: {context}\nAnswer:"

    return prompt


# ---------------------------------------------------------------------
# Model querying 
# ---------------------------------------------------------------------

def query_model(prompt, model="gpt-3.5-turbo-instruct", max_tokens=3): 
    """ Query the GPT model """ 
    response = client.completions.create( 
        model=model, 
        prompt=prompt, 
        max_tokens=max_tokens, 
        temperature=0, 
        top_p=1.0, 
    ) 
    text = response.choices[0].text.strip() 
    text = re.sub(r"[^A-Za-z'-]+", "", text).lower() 
    return text


# ---------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------
def evaluate(
    n_samples=100,
    k=5,
    model="gpt-3.5-turbo-instruct",
    mode="cloze",
    use_semantic=True,
    log_path="results.csv",
):
    correct, total = 0, 0
    results = []

    print(f"Running {n_samples} samples | {k}-shot | Semantic={use_semantic} | Mode={mode}")

    # Prepare CSV logger
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["#","Gold","Prediction","Correct","PromptLength"])

        for idx, ex in enumerate(dataset.select(range(n_samples))):
            text = ex["text"].strip()
            if " " not in text:
                continue
            ctx, gold = text.rsplit(" ", 1)
            prompt = build_prompt(ctx, k=k, mode=mode, use_semantic=use_semantic)
            pred = query_model(prompt, model=model)

            is_correct = pred.lower() == gold.lower()
            correct += int(is_correct)
            total += 1

            writer.writerow([total, gold, pred, int(is_correct), len(prompt)])
            print(f"[{total}] Pred: {pred:<15} | Gold: {gold:<15} | {'True' if is_correct else 'False'}")

    acc = correct / total * 100 if total > 0 else 0
    print(f"\n Accuracy on {n_samples} samples ({k}-shot): {acc:.2f}%")
    return acc



