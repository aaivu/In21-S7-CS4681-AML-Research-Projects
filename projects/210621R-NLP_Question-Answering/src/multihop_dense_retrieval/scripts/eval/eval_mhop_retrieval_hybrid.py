import argparse
import json
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import RobertaTokenizer
import sys
sys.path.append('./')
from mdr.retrieval.models.mhop_retriever import RobertaRetriever
from mdr.retrieval.utils.utils import load_saved

# Add after existing imports
def hybrid_score(dense_scores, sparse_scores, alpha=0.7):
    """Combine dense and sparse scores with alpha weighting"""
    dense_norm = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-10)
    sparse_norm = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min() + 1e-10)
    return alpha * dense_norm + (1 - alpha) * sparse_norm

def build_bm25_index(corpus_texts):
    """Build BM25 index from corpus"""
    tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
    return BM25Okapi(tokenized_corpus)

def retrieve_hybrid(query, dense_retriever, bm25_index, corpus, k=100, alpha=0.7):
    """Hybrid retrieval combining dense and BM25"""
    # Dense retrieval
    with torch.no_grad():
        query_embed = dense_retriever.encode_query(query)
        dense_scores = dense_retriever.get_scores(query_embed)
    
    # BM25 retrieval
    tokenized_query = query.lower().split()
    sparse_scores = bm25_index.get_scores(tokenized_query)
    
    # Combine scores
    final_scores = hybrid_score(dense_scores, sparse_scores, alpha)
    
    # Get top-k
    top_indices = np.argsort(final_scores)[-k:][::-1]
    return top_indices, final_scores[top_indices]

# Modify main evaluation function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('question_file', type=str)
    parser.add_argument('corpus_file', type=str)
    parser.add_argument('id2doc_file', type=str)
    parser.add_argument('model_path', type=str)
    parser.add_argument('--alpha', type=float, default=0.7, help='Weight for dense retrieval')
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--save-path', type=str, required=True)
    args = parser.parse_args()
    
    # Load model
    retriever = RobertaRetriever.from_pretrained(args.model_path)
    if args.gpu:
        retriever = retriever.cuda()
    
    # Load corpus
    with open(args.id2doc_file) as f:
        id2doc = json.load(f)
    corpus_texts = [id2doc[str(i)]['text'] for i in range(len(id2doc))]
    
    # Build BM25 index
    print("Building BM25 index...")
    bm25_index = build_bm25_index(corpus_texts)
    
    # Load questions
    with open(args.question_file) as f:
        questions = json.load(f)
    
    results = []
    for qa in questions:
        indices, scores = retrieve_hybrid(
            qa['question'], 
            retriever, 
            bm25_index, 
            corpus_texts, 
            k=args.topk,
            alpha=args.alpha
        )
        results.append({
            'question': qa['question'],
            'retrieved': [id2doc[str(i)] for i in indices],
            'scores': scores.tolist()
        })
    
    with open(args.save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.save_path}")

if __name__ == '__main__':
    main()