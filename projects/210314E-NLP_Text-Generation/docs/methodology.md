# Methodology: Impact of Activation Functions on PEGASUS-X for Abstractive Text Summarization

**Student:** 210314E <br>
**Research Area:** NLP: Text Generation (Abstractive Summarization) <br>
**Date:** 2025-09-01 <br>


## 1. Overview

This methodology outlines the approach for improving the PEGASUS-X model for long document abstractive summarization. Text summarization condenses lengthy documents into concise summaries while preserving key information. Transformer-based models like PEGASUS-X have shown strong performance due to pretraining and encoder-decoder architectures, but face challenges with quadratic complexity in attention for very long documents. PEGASUS-X extends PEGASUS with efficient attention mechanisms and pretraining on long inputs, handling up to 16K tokens.

The problem addressed is optimizing PEGASUS-X's architecture for long documents through lightweight modifications, such as changing activation functions and adding layers, to enhance efficiency and effectiveness without significantly increasing computational cost. The research explores targeted modifications to improve summarization accuracy for longer texts, evaluating their impact on quality metrics and generalizability across tasks.

## 2. Research Design

The research follows an experimental approach to enhance PEGASUS-X for long document summarization. The main objective is to explore lightweight architectural and training modifications during fine-tuning, focusing on efficiency and stability. Key goals include:

- Experimenting with architectural modifications such as changing activation functions and adding intermediate layers.
- Evaluating modifications using standard metrics like ROUGE.
- Comparing modified models against the baseline PEGASUS-X on long and short document datasets.
- Investigating generalizability across summarization tasks.

Experiments will be conducted on cloud GPUs using PyTorch and Hugging Face libraries, loading the baseline model from Hugging Face. Modifications are applied during fine-tuning on long document benchmarks, with optimizations like gradient checkpointing, adjusted learning rates, and gradient accumulation for stability.

## 3. Data Collection

### 3.1 Data Sources
- Kaggle
- GitHub
- Hugging Face

### 3.2 Data Description
To comprehensively evaluate the influence of activation functions on the PEGASUS-X model, experiments were conducted across six benchmark summarisation datasets that vary in length, style, and domain. This diversity ensured that the findings generalise across both short and long-document summarisation settings.

- **GovReport** - The GovReport dataset contains long-form government reports and policy documents summarised into concise executive summaries. With documents averaging around 9,000 tokens and summaries approximately 500 tokens, this dataset evaluates model scalability and the stability of gradient dynamics in long-context settings. It consists of 17,000 training samples, 1,000 validation samples, and 1,000 test samples.
- **CNN/DailyMail** - The CNN/DailyMail dataset consists of online news articles paired with multi-sentence highlights written by journalists. It contains approximately 287,000 training samples, 13,000 validation samples, and 11,000 test samples. Articles average about 760 tokens, while summaries are around 60 tokens long. This dataset primarily measures the model’s ability to produce coherent, factual, and moderately abstractive multi-sentence summaries. Due to computational resource constraints, only a part of the dataset was used to fine tune the model.
- **XSum** - The XSum dataset contains BBC news articles with single-sentence abstractive summaries designed to capture the core message of each article. It comprises roughly 204,000 training samples, 11,000 validation samples, and 11,000 test samples. Documents average 430 tokens, requiring the model to generate concise, information-dense summaries rather than extractive paraphrases.
- **SummScreen** - SummScreen is a dialogue-centric dataset built from television and movie transcripts paired with human-written recaps. It includes approximately 26,000 examples, with transcripts often exceeding 6,000 tokens per episode. This dataset tests the ability of PEGASUS-X to handle extended contexts and conversational input structures, offering insight into how activation functions behave in long-sequence summarisation.
- **QMSum** - The QMSum dataset is a human-annotated benchmark dataset consisting of long transcripts of meetings. The average input length is around 9,100 words. The dataset comprises 1,808 query-summary pairs derived from 232 meetings across academic, product, and committee domains. The reference summaries are relatively short, averaging approximately 70 words. QMSum evaluates a model’s ability to handle long-context dialogues and perform focused, information-seeking summarisation.
- **BIGPATENT** - The BIGPATENT dataset is a large-scale collection of U.S. patent documents for abstractive summarisation. It contains approximately 1.3 million records, where the input document is the detailed patent description and the summary is the human-written patent abstract. This dataset is designed to challenge summarisation models by requiring them to handle specialised, long-form technical text with a complex discourse structure. The summaries generally exhibit lower lexical overlap with the source compared to news datasets, promoting highly abstractive generation. Due to computational resource constraints, only a part of the dataset was used to fine tune the model.

Long document datasets (GovReport, SummScreen, QMSum, BIGPATENT) are the primary focus for fine-tuning and evaluation. Short document datasets (CNN/DailyMail, XSum) are used for comparison to ensure modifications do not degrade performance on shorter inputs.

### 3.3 Data Preprocessing
To ensure consistent and reproducible data handling across all datasets, a standardised preprocessing pipeline was implemented. The steps were designed to align with PEGASUS-X’s tokenisation and long-context processing capabilities.

- Raw text from each dataset was first cleaned to remove HTML tags, escape characters, and extra whitespace. For GovReport, section headers and bullet points were merged into continuous text to maintain coherence during tokenisation.
- Sentences were lowercased to maintain consistency across datasets.
- All datasets were tokenised using the standard PEGASUS tokenizer, configured with a vocabulary size of 96,000 subword units. Tokenisation was consistently applied across the datasets using the Hugging Face transformers library, ensuring compatibility with PEGASUS-X’s pretraining scheme and shared embeddings across experiments.
- To accommodate PEGASUS-X’s extended context capability, dataset-specific maximum input lengths were applied. Inputs shorter than the maximum length were dynamically padded, while longer sequences were truncated from the end, as preliminary trials indicated that critical information typically occurs earlier in documents.
- Official train/validation/test splits provided with each dataset were used to ensure comparability with prior work.
- Each model variant was evaluated on the held-out test split for each dataset. The final scores were averaged over multiple runs to mitigate variance caused by stochastic factors in training.

This evaluation framework facilitated the examination of activation function behaviour across diverse text characteristics, ranging from short, factual summaries to long, multi-paragraph reports. Combined with a unified preprocessing pipeline that enforces consistent tokenisation, truncation, and batching strategies, this design provided a controlled environment in which the influence of activation functions could be assessed independently of the variability of the dataset. This standardisation ensured that the observed performance differences reflect the true impact of activation functions on summarisation quality and training stability.

| **Dataset**       | **Max Input Tokens** | **Max Output Tokens** |
|---------------|------------------|-------------------|
| XSum          | 1024             | 128               |
| CNN/DailyMail | 1024             | 128               |
| QMSum         | 16384            | 256               |
| SummScreen    | 16384            | 256               |
| GovReport     | 12288            | 1024              |
| Big Patent    | 16384            | 256               |

## 4. Model Architecture

The baseline model is PEGASUS-X, an extension of PEGASUS designed for long input sequences up to 16,384 tokens. It uses an encoder-decoder architecture with the following key modifications:

- **Efficient Attention Mechanism**: The encoder employs block-local attention where tokens are divided into fixed blocks and attend only within their block. Staggered blocks shift boundaries across layers to allow information flow. Global tokens  attend to and are attended by all tokens for global context.
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


### 5.2 Baseline Models
The baseline is PEGASUS-X, compared against modified variants. PEGASUS-X achieves state-of-the-art results on long document benchmarks like arXiv, PubMed, and GovReport, outperforming models like LongT5 while maintaining performance on short inputs.

### 5.3 Hardware/Software Requirements
Experiments run on cloud GPUs via Kaggle Notebooks. Main libraries will be PyTorch and Hugging Face transformers. Model and tokenizer will be loaded from Hugging Face repository to ensure compatibility.

## 6. Implementation Plan

| **Phase** | **Tasks** | **Duration** | **Deliverables** |
|-------|-------|----------|--------------|
| Preparation | Literature review, environment setup, codebase understanding | 14 Aug - 28 Aug | Environment configured, codebase explored |
| Implementation and Testing | Baseline testing, model modifications and training, evaluation | 26 Aug - 29 Sep | Modified models trained, performance metrics |
| Documentation | Progress report, short paper, final paper, submission | 22 Aug - 5 Oct | Reports and papers completed |

## 7. Risk Analysis

| **Risk** | **Description** | **Mitigation Strategy** |
|------|-------------|-------------------|
| Computational Constraints | Limited to GPUs on Kaggle, potentially restricting maximum input lengths | Use gradient checkpointing and efficient batching |
| Time Constraints | Seven-week period limits experiments and hyperparameter tuning | Prioritize key modifications and datasets |
| No Large-Scale Pretraining | Improvements restricted to fine-tuning | Focus on lightweight changes that build on existing pretraining |
| Dataset Coverage | Subset of datasets used due to resources | Select representative long document datasets like GovReport and arXiv |

## 8. Expected Outcomes

- Improved summarization quality with higher ROUGE, BLEU, and BERTScore metrics compared to baseline PEGASUS-X.
- Insights into efficiency tradeoffs between input length, memory usage, and quality.
- Empirical comparison of activation functions for long document tasks.
- Reproducible implementation using Hugging Face libraries, contributing to scalable long document summarization research.