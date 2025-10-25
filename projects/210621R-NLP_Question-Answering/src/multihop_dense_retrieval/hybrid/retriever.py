"""
Hybrid retrieval implementation combining dense and sparse scores.
"""
import numpy as np
from typing import Dict, List, Tuple
from ..dense.model import DenseRetriever
from ..sparse.retriever import SparseRetriever

class HybridRetriever:
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        alpha: float = 0.7,
        epsilon: float = 1e-10
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.alpha = alpha
        self.epsilon = epsilon

    def normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Min-max normalize scores to [0,1] range."""
        if not scores:
            return scores
            
        values = np.array(list(scores.values()))
        min_val = values.min()
        max_val = values.max()
        
        # Normalize
        normalized = {
            doc_id: (score - min_val) / (max_val - min_val + self.epsilon)
            for doc_id, score in scores.items()
        }
        
        return normalized

    def combine_scores(
        self,
        dense_scores: Dict[str, float],
        sparse_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Combine normalized dense and sparse scores using weighted sum.
        """
        # Normalize scores
        dense_norm = self.normalize_scores(dense_scores)
        sparse_norm = self.normalize_scores(sparse_scores)
        
        # Get union of doc ids
        all_docs = set(dense_norm.keys()) | set(sparse_norm.keys())
        
        # Combine scores
        combined = {}
        for doc_id in all_docs:
            dense_score = dense_norm.get(doc_id, 0.0)
            sparse_score = sparse_norm.get(doc_id, 0.0)
            combined[doc_id] = (
                self.alpha * dense_score + 
                (1 - self.alpha) * sparse_score
            )
            
        return combined

    def retrieve(
        self,
        query: str,
        top_k: int = 100
    ) -> Tuple[List[str], List[float]]:
        """
        Retrieve documents using hybrid scoring.
        Returns tuple of (doc_ids, scores).
        """
        # Get dense and sparse scores
        dense_scores = self.dense_retriever.retrieve(query, top_k)
        sparse_scores = self.sparse_retriever.retrieve(query, top_k)
        
        # Combine scores
        combined = self.combine_scores(dense_scores, sparse_scores)
        
        # Sort by score
        sorted_results = sorted(
            combined.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Get top-k
        top_k_docs = sorted_results[:top_k]
        doc_ids, scores = zip(*top_k_docs)
        
        return list(doc_ids), list(scores)

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 100
    ) -> List[Tuple[List[str], List[float]]]:
        """Batch retrieve for multiple queries."""
        return [self.retrieve(q, top_k) for q in queries]