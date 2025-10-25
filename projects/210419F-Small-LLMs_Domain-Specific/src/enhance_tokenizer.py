#!/usr/bin/env python3
"""
enhance_tokenizer.py

Enhance DistilBERT tokenizer with PubMed-specific tokens using KL divergence.

Steps:
1. Train a WordPiece tokenizer on PubMed corpus.
2. Count token frequencies for PubMed and Wikipedia.
3. Compute KL divergence contributions (PubMed || Wikipedia).
4. Select top-N tokens not in DistilBERT.
5. Extend tokenizer (and optionally resize model).
"""

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from tqdm import tqdm

from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer

from transformers import AutoTokenizer, AutoModelForMaskedLM

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

# -------------------- Helpers --------------------

def stream_jsonl(path: str) -> Iterable[str]:
    """Yield 'text' field line by line from JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "text" in obj:
                txt = obj["text"].strip()
                if txt:
                    yield txt

def train_wordpiece(pubmed_jsonl: str, vocab_size: int = 60000, min_freq: int = 2) -> Tokenizer:
    """Train WordPiece tokenizer on PubMed corpus."""
    tok = Tokenizer(WordPiece(unk_token="[UNK]"))

    tok.normalizer = normalizers.Sequence([
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase()
    ])
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=SPECIALS,
        continuing_subword_prefix="##"
    )

    tok.train_from_iterator(stream_jsonl(pubmed_jsonl), trainer=trainer)
    return tok

def count_tokens(tokenizer: Tokenizer, texts: Iterable[str]) -> Counter:
    """Count token frequencies in a corpus."""
    counts = Counter()
    for txt in texts:
        enc = tokenizer.encode(txt)
        for tok in enc.tokens:
            if tok not in SPECIALS:
                counts[tok] += 1
    return counts

def kl_contributions(pub_counts: Counter, wiki_counts: Counter, alpha: float = 1.0):
    """Compute per-token KL(P_pub || P_wiki)."""
    vocab = set(pub_counts.keys()) | set(wiki_counts.keys())
    V = len(vocab)

    total_pub = sum(pub_counts.values())
    total_wiki = sum(wiki_counts.values())

    denom_pub = total_pub + alpha * V
    denom_wiki = total_wiki + alpha * V

    contribs = {}
    total_kl = 0.0
    for tok in vocab:
        p = (pub_counts.get(tok, 0) + alpha) / denom_pub
        q = (wiki_counts.get(tok, 0) + alpha) / denom_wiki
        val = p * math.log(p / q)
        contribs[tok] = val
        total_kl += val

    return contribs, total_kl

def looks_informative(tok: str) -> bool:
    """Heuristics to filter junk tokens."""
    core = tok[2:] if tok.startswith("##") else tok
    if len(core) < 3:
        return False
    if not any(ch.isalpha() for ch in core):
        return False
    return True

# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki", required=True, help="Normalized wiki.jsonl")
    ap.add_argument("--pubmed", required=True, help="Normalized pubmed.jsonl")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--base_model", default="distilbert-base-uncased")
    ap.add_argument("--candidate_vocab", type=int, default=60000)
    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument("--kl_alpha", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=2000)
    ap.add_argument("--min_pubmed_count", type=int, default=5)
    ap.add_argument("--save_model", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Train candidate tokenizer
    print("Training PubMed WordPiece tokenizer...")
    cand_tok = train_wordpiece(args.pubmed, args.candidate_vocab, args.min_freq)
    cand_tok.save(str(out_dir / "candidate_tokenizer.json"))

    # 2. Count tokens
    print("Counting tokens...")
    pub_counts = count_tokens(cand_tok, stream_jsonl(args.pubmed))
    wiki_counts = count_tokens(cand_tok, stream_jsonl(args.wiki))

    # 3. KL contributions
    print("Computing KL contributions...")
    contribs, total_kl = kl_contributions(pub_counts, wiki_counts, args.kl_alpha)
    print(f"Total KL divergence = {total_kl:.4f}")

    # Save scored list
    with open(out_dir / "kl_scored.tsv", "w", encoding="utf-8") as f:
        f.write("token\tpubmed_count\twiki_count\tkl_contribution\n")
        for tok, score in sorted(contribs.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{tok}\t{pub_counts.get(tok,0)}\t{wiki_counts.get(tok,0)}\t{score:.8f}\n")

    # 4. Select top tokens not in DistilBERT
    base_tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    base_vocab = set(base_tok.get_vocab().keys())

    candidates = []
    for tok, score in sorted(contribs.items(), key=lambda x: x[1], reverse=True):
        if tok in base_vocab:
            continue
        if pub_counts.get(tok, 0) < args.min_pubmed_count:
            continue
        if not looks_informative(tok):
            continue
        candidates.append(tok)

    new_tokens = candidates[:args.top_k]

    with open(out_dir / "selected_tokens.txt", "w", encoding="utf-8") as f:
        for t in new_tokens:
            f.write(t + "\n")

    print(f"Selected {len(new_tokens)} new tokens")

    # 5. Extend tokenizer (and optionally model)
    added = base_tok.add_tokens(new_tokens)
    print(f"Added {added} tokens to tokenizer")

    tok_out = out_dir / "tokenizer"
    base_tok.save_pretrained(tok_out)
    print(f"Saved tokenizer to {tok_out}")

    if args.save_model:
        model = AutoModelForMaskedLM.from_pretrained(args.base_model)
        model.resize_token_embeddings(len(base_tok))
        model.save_pretrained(out_dir / "model")
        print(f"Saved resized model with new vocab to {out_dir/'model'}")

if __name__ == "__main__":
    main()
