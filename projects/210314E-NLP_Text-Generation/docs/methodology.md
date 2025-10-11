# Methodology: NLP:Text Generation

**Student:** 210314E
**Research Area:** NLP:Text Generation
**Date:** 2025-09-01

## 1. Overview

This methodology outlines the approach for improving the PEGASUS-X model for long document abstractive summarization. Text summarization condenses lengthy documents into concise summaries while preserving key information. Transformer-based models like PEGASUS-X have shown strong performance due to pretraining and encoder-decoder architectures, but face challenges with quadratic complexity in attention for very long documents. PEGASUS-X extends PEGASUS with efficient attention mechanisms and pretraining on long inputs, handling up to 16K tokens.

The problem addressed is optimizing PEGASUS-X's architecture for longer documents through lightweight modifications, such as changing activation functions and adding layers, to enhance efficiency and effectiveness without significantly increasing computational cost. The research explores targeted modifications to improve summarization accuracy for longer texts, evaluating their impact on quality metrics and generalizability across tasks.

## 2. Research Design

The research follows an experimental approach to enhance PEGASUS-X for long document summarization. The main objective is to explore lightweight architectural and training modifications during fine-tuning, focusing on efficiency and stability. Key goals include:

- Experimenting with architectural modifications such as changing activation functions (e.g., replacing GeLU with Swish or SwiGLU) and adding intermediate layers (e.g., LayerNorm and linear projections).
- Evaluating modifications using standard metrics like ROUGE, BLEU, and BERTScore.
- Comparing modified models against the baseline PEGASUS-X on long and short document datasets.
- Investigating generalizability across summarization tasks.

Experiments will be conducted on cloud GPUs using PyTorch and Hugging Face libraries, loading the baseline model from Hugging Face. Modifications are applied during fine-tuning on long document benchmarks, with optimizations like gradient checkpointing, adjusted learning rates, and gradient accumulation for stability.

## 3. Data Collection

### 3.1 Data Sources
- arXiv: Scientific articles
- PubMed: Biomedical research
- GovReport: Government reports
- BigPatent: Patent documents
- CNN/DailyMail: News articles
- XSum: News articles

### 3.2 Data Description
| Dataset       | Domain              | Avg. Input Length |
|---------------|---------------------|-------------------|
| arXiv         | Scientific articles | ∼6,900 tokens    |
| PubMed        | Biomedical research | ∼4,700 tokens    |
| GovReport     | Government reports  | ∼8,000 tokens    |
| BigPatent     | Patent documents    | up to 10,000 tokens |
| CNN/DailyMail | News articles       | ∼700 tokens      |
| XSum          | News articles       | ∼430 tokens      |

Long document datasets (arXiv, PubMed, GovReport, BigPatent) are the primary focus for fine-tuning and evaluation. Short document datasets (CNN/DailyMail, XSum) are used for comparison to ensure modifications do not degrade performance on shorter inputs.

### 3.3 Data Preprocessing
Data preprocessing involves fine-tuning the model on these datasets with input lengths up to 16K tokens. No additional preprocessing steps are detailed beyond standard tokenization using the Hugging Face tokenizer matching the PEGASUS-X checkpoint.

## 4. Model Architecture

The baseline model is PEGASUS-X, an extension of PEGASUS designed for long input sequences up to 16,384 tokens. It uses an encoder-decoder architecture with the following key modifications:

- **Efficient Attention Mechanism**: The encoder employs block-local attention where tokens are divided into fixed blocks and attend only within their block. Staggered blocks shift boundaries across layers to allow information flow. Global tokens (special learnable embeddings) attend to and are attended by all tokens for global context.
- **Architecture Adjustments**: Minimal new parameters added, including global token embeddings and additional LayerNorm layers. Input context extended from 512 to 16K tokens during fine-tuning.
- **Pretraining and Fine-Tuning**: Pretrained on short sequences (512 tokens) with masked sentence prediction, followed by additional pretraining on longer inputs (4096 tokens) for 300K steps. Fine-tuned on downstream tasks with up to 16K tokens.

Proposed modifications include:
- Replacing GeLU activation with Swish or SwiGLU in feed-forward layers for better training stability.
- Adding intermediate LayerNorm and linear projection layers between attention and feed-forward blocks in selected encoder layers.

## 5. Experimental Setup

### 5.1 Evaluation Metrics
- ROUGE-1: Unigram overlap between system and reference summaries.
- ROUGE-2: Bigram overlap, capturing short phrase similarity.
- ROUGE-L: Longest common subsequence, reflecting fluency and structural similarity.
- BLEU: Precision-oriented n-gram overlap, penalizing extraneous content.
- BERTScore: Semantic similarity using contextual embeddings, capturing paraphrasing.

ROUGE F1 scores and geometric mean of ROUGE scores will be primarily reported, with BLEU and BERTScore as complementary measures.

### 5.2 Baseline Models
The baseline is PEGASUS-X, compared against modified variants. PEGASUS-X achieves state-of-the-art results on long document benchmarks like arXiv, PubMed, and GovReport, outperforming models like LongT5 while maintaining performance on short inputs.

### 5.3 Hardware/Software Requirements
Experiments run on cloud GPUs via Kaggle Notebooks and Google Colab. Main libraries: PyTorch and Hugging Face transformers. Model and tokenizer loaded from Hugging Face repository to ensure compatibility.

## 6. Implementation Plan

| Phase | Tasks | Duration | Deliverables |
|-------|-------|----------|--------------|
| Preparation | Literature review, environment setup, codebase understanding | 14 Aug - 28 Aug | Environment configured, codebase explored |
| Implementation and Testing | Baseline testing, model modifications and training, evaluation | 26 Aug - 29 Sep | Modified models trained, performance metrics |
| Documentation | Progress report, short paper, final paper, submission | 22 Aug - 5 Oct | Reports and papers completed |

## 7. Risk Analysis

- **Computational Constraints**: Limited to GPUs on Google Colab and Kaggle, potentially restricting maximum input lengths. Mitigation: Use gradient checkpointing and efficient batching.
- **Time Constraints**: Seven-week period limits experiments and hyperparameter tuning. Mitigation: Prioritize key modifications and datasets.
- **No Large-Scale Pretraining**: Improvements restricted to fine-tuning. Mitigation: Focus on lightweight changes that build on existing pretraining.
- **Dataset Coverage**: Subset of datasets used due to resources. Mitigation: Select representative long document datasets like GovReport and arXiv.

## 8. Expected Outcomes

- Improved summarization quality with higher ROUGE, BLEU, and BERTScore metrics compared to baseline PEGASUS-X.
- Better training stability through modifications like gradient checkpointing and adjusted learning rates.
- Insights into efficiency tradeoffs between input length, memory usage, and quality.
- Empirical comparison of activation functions (GeLU, Swish, SwiGLU) for long document tasks.
- Reproducible implementation using Hugging Face libraries, contributing to scalable long document summarization research.

---

**Note:** Update this document as your methodology evolves during implementation.