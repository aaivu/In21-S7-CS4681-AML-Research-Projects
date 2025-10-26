# Hybrid Dense-Sparse Retrieval for Multi-hop Question Answering

## Student Information

- **Index Number:** 210621R
- **Research Area:** NLP:Question Answering
- **GitHub Username:** @subu0106
- **Email:** subavarshanaa.21@cse.mrt.ac.lk
- **Supervisor:** Dr. Uthayasanker Thayasivam

## Project Overview

This research proposes a novel hybrid retrieval framework for multi-hop question answering that synergistically combines dense neural representations with sparse lexical matching. The system achieves a 3.4% improvement in top-1 retrieval accuracy over baseline Multi-hop Dense Retrieval (MDR) systems, with minimal computational overhead.

### Key Features

- Hybrid retrieval combining dense and sparse paradigms
- Normalized weighted scoring mechanism
- Efficient implementation with minimal latency
- Comprehensive evaluation on HotpotQA

## Project Structure

```
210621R-NLP_Question-Answering/
├── README.md                    # Project overview
├── docs/
│   ├── research_proposal.md     # Detailed research proposal
│   ├── literature_review.md     # Comprehensive literature review
│   ├── methodology.md           # Technical methodology
│   ├── usage_instructions.md    # System usage guide
│   └── progress_reports/        # Weekly updates
├── src/
│   ├── models/                  # Model implementations
│   │   ├── dense_retriever.py  # Dense retrieval component
│   │   ├── sparse_retriever.py # BM25 implementation
│   │   └── hybrid_scorer.py    # Score combination
│   ├── data/                   # Data processing
│   ├── utils/                  # Helper functions
│   └── eval/                   # Evaluation scripts
├── experiments/
│   ├── configs/                # Configuration files
│   └── scripts/                # Training/eval scripts
├── data/
│   ├── hotpotqa/              # HotpotQA dataset
│   ├── indices/               # BM25 and FAISS indices
│   └── embeddings/            # Pre-computed embeddings
├── results/                    # Experimental results
└── requirements.txt            # Project dependencies
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/aaivu/In21-S7-CS4681-AML-Research-Projects.git
cd projects/210621R-NLP_Question-Answering
```

1. Create and activate virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

1. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

1. Download required data:

```bash
python scripts/download_data.py
```

1. Train the model:

```bash
python train.py --config configs/hybrid_retriever.yaml
```

1. Run evaluation:

```bash
python evaluate.py --model hybrid --split dev
```

## Key Results

- Top-1 Accuracy: 61.7% (+3.4% over baseline)
- Top-10 Accuracy: 87.1% (+2.9%)
- MRR: 0.694 (+0.029)
- Average latency: 53-57ms per query

## Documentation

- [Research Proposal](docs/research_proposal.md)
- [Literature Review](docs/literature_review.md)
- [Methodology](docs/methodology.md)
- [Usage Instructions](docs/usage_instructions.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Research supported by the Department of Computer Science and Engineering, University of Moratuwa
- Computing resources provided by University of Moratuwa
- HotpotQA dataset creators and maintainers

---

**Note:** For detailed setup and usage instructions, please refer to [Usage Instructions](docs/usage_instructions.md).