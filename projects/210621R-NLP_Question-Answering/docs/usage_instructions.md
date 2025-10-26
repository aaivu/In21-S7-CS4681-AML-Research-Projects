# Usage Instructions: Multi-hop QA Hybrid Retrieval System

## Setup Requirements

### Hardware Requirements

- GPU: NVIDIA V100 or equivalent
- RAM: 256GB minimum
- Storage: 10GB free space

### Software Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers library
- rank_bm25
- FAISS

## Installation

1. Clone the repository:

```bash
git clone https://github.com/aaivu/In21-S7-CS4681-AML-Research-Projects.git
cd projects/210621R-NLP_Question-Answering
```

1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Download pre-trained models:

```bash
python scripts/download_models.py
```

## Data Preparation

1. Download HotpotQA dataset:

```bash
python scripts/download_hotpotqa.py
```

1. Build indices:

```bash
python scripts/build_indices.py
```

## Running the System

### Training

1. Train dense retriever:

```bash
python train.py --config configs/dense_retriever.yaml
```

1. Train hybrid scorer:

```bash
python train.py --config configs/hybrid_scorer.yaml
```

### Evaluation

1. Run evaluation:

```bash
python evaluate.py --model hybrid --split dev
```

1. Generate analysis:

```bash
python analyze.py --results path/to/results.json
```

## Configuration

Edit `configs/config.yaml` to modify:

- Model architecture parameters
- Training hyperparameters
- Evaluation settings
- Hardware utilization

## Troubleshooting

Common issues and solutions:

1. Out of memory: Reduce batch size in config
2. Slow retrieval: Enable FAISS indexing
3. Poor accuracy: Verify data preprocessing

## Contributing

1. Create feature branch
1. Make changes
1. Run tests:

```bash
python -m pytest tests/
```

1. Submit pull request

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{arumugam2025hybrid,
  title={Hybrid Dense-Sparse Retrieval for Multi-hop Question Answering},
  author={Arumugam, Subavarshana and Thayasivam, Uthayasanker},
  booktitle={Proceedings of AIRCC},
  year={2025}
}
```

## Support

For issues and questions:

1. Check documentation
1. Search existing issues
1. Create new issue with details:
   - System configuration
   - Error messages
   - Steps to reproduce

---

**Note:** Keep checking for updates as the system evolves.