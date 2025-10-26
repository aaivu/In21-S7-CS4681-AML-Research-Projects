"""
BM25 retrieval implementation using rank_bm25 library.
"""
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Dict
import pickle
import os

class SparseRetriever:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.doc_ids = None
        self.index_path = None

    def build_index(self, documents: List[str], doc_ids: List[str], index_path: str):
        """Build BM25 index from documents."""
        # Tokenize documents
        tokenized_docs = [doc.split() for doc in documents]
        
        # Create BM25 index
        self.bm25 = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)
        self.doc_ids = doc_ids
        self.index_path = index_path
        
        # Save index
        self._save_index()

    def _save_index(self):
        """Save BM25 index to disk."""
        if self.index_path:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            with open(self.index_path, 'wb') as f:
                pickle.dump({
                    'bm25': self.bm25,
                    'doc_ids': self.doc_ids
                }, f)

    def load_index(self, index_path: str):
        """Load BM25 index from disk."""
        with open(index_path, 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data['bm25']
            self.doc_ids = data['doc_ids']
            self.index_path = index_path

    def retrieve(self, query: str, top_k: int = 100) -> Dict[str, float]:
        """
        Retrieve top-k documents for query.
        Returns dict mapping doc_ids to BM25 scores.
        """
        if not self.bm25:
            raise ValueError("Index not built or loaded")
            
        # Tokenize query
        tokenized_query = query.split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k doc ids and scores
        top_k_indices = np.argsort(scores)[-top_k:][::-1]
        results = {
            self.doc_ids[idx]: scores[idx]
            for idx in top_k_indices
        }
        
        return results

    def batch_retrieve(self, queries: List[str], top_k: int = 100) -> List[Dict[str, float]]:
        """Batch retrieve for multiple queries."""
        return [self.retrieve(q, top_k) for q in queries]