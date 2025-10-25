# Technical Implementation Details

## Dense Retrieval Component

### Model Architecture
- Base model: RoBERTa-base
- Query encoder: 768-dimensional embeddings
- Document encoder: 768-dimensional embeddings
- Training batch size: 128
- Learning rate: 2e-5
- Momentum coefficient: 0.999

### Training Details
- Dataset: HotpotQA training set
- Training steps: 30,000
- Optimizer: AdamW
- Weight decay: 0.01
- Warmup steps: 1000
- Temperature parameter (τ): 0.1

## Sparse Retrieval Component

### BM25 Configuration
- k1 parameter: 1.5
- b parameter: 0.75
- Index structure: Inverted index
- Term weighting: TF-IDF with BM25 normalization

### Implementation Details
- Library: rank_bm25
- Index storage format: Memory-mapped files
- Index building time: ~2.3 minutes
- Index size: 2.1GB

## Hybrid Scoring

### Score Normalization
- Method: Min-max normalization
- Range: [0,1]
- Epsilon: 1e-10

### Weighting Configuration
- Optimal alpha: 0.7
- Search range: [0.3, 0.9]
- Grid search steps: 0.1

## Performance Optimizations

### Dense Retrieval
- Batch inference
- GPU acceleration
- FAISS indexing for approximate nearest neighbor search
- Pre-computed document embeddings

### Sparse Retrieval
- Optimized inverted index
- In-memory caching
- Parallel document scoring
- Early termination for top-k