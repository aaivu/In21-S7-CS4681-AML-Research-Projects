"""
Build BM25 index for Wikipedia corpus.
"""
import json
import pickle
from pathlib import Path
from tqdm import tqdm
from ..sparse.retriever import SparseRetriever

def load_wiki_data(filepath: str):
    """Load Wikipedia corpus."""
    print(f"Loading Wikipedia data from {filepath}")
    with open(filepath) as f:
        data = json.load(f)
    return data

def main():
    # Initialize paths
    wiki_path = Path("data/hotpotqa/wiki.json")
    index_path = Path("data/bm25_index.pkl")
    
    # Create output directory
    index_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load Wikipedia data
    wiki_data = load_wiki_data(wiki_path)
    
    # Extract documents and IDs
    print("Processing documents...")
    documents = []
    doc_ids = []
    for doc_id, content in tqdm(wiki_data.items()):
        # Concatenate title and text
        doc_text = f"{content['title']} {content['text']}"
        documents.append(doc_text)
        doc_ids.append(doc_id)
    
    # Build index
    print("Building BM25 index...")
    retriever = SparseRetriever()
    retriever.build_index(
        documents=documents,
        doc_ids=doc_ids,
        index_path=str(index_path)
    )
    
    print(f"Index saved to {index_path}")

if __name__ == "__main__":
    main()