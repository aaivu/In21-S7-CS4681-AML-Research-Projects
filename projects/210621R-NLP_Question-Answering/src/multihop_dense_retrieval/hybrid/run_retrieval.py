"""
Script to run the hybrid retrieval system.
"""
import argparse
import json
import torch
from tqdm import tqdm
from pathlib import Path

from ..dense.model import DenseRetriever
from ..sparse.retriever import SparseRetriever
from ..hybrid.retriever import HybridRetriever

def load_hotpotqa(filepath):
    """Load HotpotQA dataset."""
    with open(filepath) as f:
        data = json.load(f)
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Weight for dense scores")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to input dataset")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to save predictions")
    parser.add_argument("--model_name", type=str, default="roberta-base",
                        help="Base model for dense retriever")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for dense retriever")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for retrieval")
    parser.add_argument("--top_k", type=int, default=100,
                        help="Number of documents to retrieve")
    args = parser.parse_args()

    # Load models
    print("Loading models...")
    dense_retriever = DenseRetriever(
        model_name=args.model_name,
        device=args.device
    )
    sparse_retriever = SparseRetriever()
    sparse_retriever.load_index("data/bm25_index.pkl")

    # Initialize hybrid retriever
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        alpha=args.alpha
    )

    # Load data
    print(f"Loading data from {args.input_file}")
    data = load_hotpotqa(args.input_file)

    # Run retrieval
    print("Running retrieval...")
    predictions = []
    for batch_idx in tqdm(range(0, len(data), args.batch_size)):
        batch = data[batch_idx:batch_idx + args.batch_size]
        queries = [item["question"] for item in batch]
        
        # Get predictions
        batch_results = hybrid_retriever.batch_retrieve(
            queries,
            top_k=args.top_k
        )
        
        # Format results
        for item, (doc_ids, scores) in zip(batch, batch_results):
            predictions.append({
                "question_id": item["_id"],
                "retrieved_docs": doc_ids,
                "scores": scores
            })

    # Save predictions
    print(f"Saving predictions to {args.output_file}")
    output_dir = Path(args.output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.output_file, "w") as f:
        json.dump(predictions, f)

if __name__ == "__main__":
    main()