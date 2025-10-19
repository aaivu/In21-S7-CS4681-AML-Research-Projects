# Literature Review: Biocomputing:DNA Computing

**Student:** 210483T
**Research Area:** Biocomputing:DNA Computing
**Date:** 2025-09-01

## Abstract

This literature review explores the rapid development of DNA foundation models, transformer-based genomics, and parameter-efficient learning techniques in biocomputing. It examines key advances such as DNABERT and Nucleotide Transformers, the integration of DNA embeddings with large language models, and the emergence of LoRA fine-tuning for efficient model adaptation. The review also highlights trends in hybrid bio-AI systems, identifies research gaps in scalable and interpretable genomic reasoning, and establishes a methodological foundation for developing efficient disease prediction systems based on genomic data.

## 1. Introduction

Biocomputing, particularly DNA computing, leverages computational models to interpret genomic information and derive meaningful biological insights. The availability of large-scale genomic datasets and the rise of foundation models have transformed how computational genomics approaches problems like variant effect prediction and disease classification. Unlike traditional bioinformatics methods, transformer-based architectures can learn complex dependencies in DNA sequences, enabling more accurate downstream predictions. This review focuses on foundational works, recent advances, and methodological directions relevant to adapting DNA foundation models for disease prediction.

## 2. Search Methodology

### Search Terms Used
- Nucleotide Transformer
- DNA foundation model
- LoRA Finetuning

### Databases Searched
- [x] IEEE Xplore
- [ ] ACM Digital Library
- [x] Google Scholar
- [x] ArXiv
- [x] Other: Biorxiv

### Time Period
2018-2025

## 3. Key Areas of Research

### 3.1 Transformer-based DNA Foundation Models
Recent years have seen the rise of transformer-based models for genomic sequence modeling, following the success of language models in NLP.

**Key Papers:**
    - Ji et al. (2021) — DNABERT introduced the first BERT-style DNA language model, demonstrating k-mer tokenization and masked language modeling significantly improves performance on promoter and enhancer prediction tasks.

    - Dalla-Torre et al. (2025) — Nucleotide Transformer introduced large DNA foundation models (NT-500M to 2.5B) pretrained on human genome, achieving state-of-the-art results on multiple benchmarks.

    - Nguyen et al. (2024) — HyenaDNA proposed long-range genomic modeling using efficient attention alternatives to scale to hundreds of thousands of tokens.

### 3.2 Parameter-efficient Adaptation & Linear Probing
Fine-tuning large foundation models is often computationally prohibitive in genomic settings. Parameter-efficient methods like LoRA and linear probing address this challenge.

**Key Papers:**

    -Hu et al. (2021) — LoRA introduced low-rank adaptation, showing that adding trainable low-rank matrices enables efficient fine-tuning with minimal performance loss.

    -Hayou et al. (2024) — LoRA+ improved LoRA performance with better convergence and stability.

    -Schmirler et al. (2024) — showed linear probing on pretrained protein models can outperform fine-tuning for small datasets.

## 4. Research Gaps and Opportunities

[Identify gaps in current research that your project could address]

### Gap 1: Lack of lightweight, high-performing adaptation methods for DNA foundation models.
**Why it matters:** Full fine-tuning is resource-intensive and difficult to scale in many labs.
**How your project addresses it:** Evaluates LoRA vs. linear probing for NT-500M to identify efficient strategies.

### Gap 2: Limited comparative studies of LoRA and embedding-based methods in genomic disease prediction.
**Why it matters:** Most work focuses on NLP or protein modeling, not DNA-specific applications.
**How your project addresses it:** Performs systematic comparison using KEGG variant–disease dataset.

## 5. Theoretical Framework

This study builds on:
    - Parameter-efficient adaptation — LoRA as a fine-tuning alternative.
    - Linear probing — leveraging frozen embeddings for classification without modifying the base model.

## 6. Methodology Insights

Common methodologies in this area include:

- Transformer pretraining with masked language modeling on DNA sequences.

- LoRA fine-tuning to minimize trainable parameters.

- Embedding extraction for downstream MLP or logistic regression classifiers.

- Use of KEGG, Ensembl, or other curated genomic datasets.

Evaluation with accuracy, precision, recall, and F1-score.

For this project, embedding-based linear probing is especially promising due to its superior performance and low computational cost
## 7. Conclusion

The literature reveals a strong shift toward foundation models for genomic sequence understanding. Transformer-based approaches like Nucleotide Transformer have set new performance standards, while parameter-efficient strategies like LoRA and linear probing make adaptation practical. However, few studies have systematically compared these methods in the context of genomic disease prediction. This project addresses that gap, aiming to deliver a simple yet powerful adaptation strategy that balances performance and efficiency.

## References

[Use academic citation format - APA, IEEE, etc.]

 1. R. Schmirler, M. Heinzinger, and B. Rost, “Fine-tuning protein
    language models boosts predictions across diverse tasks,” Nat. Commun., vol.
    15,  no. 1, p. 7407, Aug. 2024, doi: 10.1038/s41467-024-51844-2.
2.  H. Dalla-Torre et al., “Nucleotide Transformer: building and
    evaluating robust foundation models for human genomics,” Nat. Methods, vol. 22,
    no. 2, pp. 287–297, Feb. 2025, doi: 10.1038/s41592-024-02523-z.
    
3. A. Fallahpour et al., “BioReason: Incentivizing Multimodal Biological
    Reasoning within a DNA-LLM Model,” May 29, 2025, arXiv:
    arXiv:2505.23579. doi: 10.48550/arXiv.2505.23579.  
4. G. Brixi et al., “Genome modeling and design across all domains
    of life with Evo 2,” Feb. 21, 2025, Genomics. doi:
    10.1101/2025.02.18.638918.
    
5. E. J. Hu et al., “LoRA: Low-Rank Adaptation of Large Language Models,” Oct. 16, 2021, arXiv: arXiv:2106.09685. doi:10.48550/arXiv.2106.09685.
    
6.  Y. Ji, Z. Zhou, H. Liu, and R. V. Davuluri, “DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNAlanguage in genome,” Bioinformatics, vol. 37, no. 15, pp. 2112–2120, Aug. 2021, doi: 10.1093/bioinformatics/btab083.
    
7. Z. Zhou, Y. Ji, W. Li, P. Dutta, R. Davuluri, and H. Liu, “DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome,” Mar. 18, 2024, arXiv: arXiv:2306.15006. doi:10.48550/arXiv.2306.15006.

8.  A. T. Merchant, S. H. King, E. Nguyen, and B. L. Hie, “Semantic
    mining of functional de novo genes from a genomic language model,” Dec. 18, 2024, Synthetic Biology. doi: 10.1101/2024.12.17.628962
    
 9. E. Nguyen et al., “Sequence modeling and design from molecular
    to genome scale with Evo”.
    
  10. E. Nguyen, M. Poli, and M. Faizi, “HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution”.
    
   11.  Z. Avsec et al., “Effective gene expression prediction from
    sequence by integrating long-range interactions,” Nat. Methods, vol. 18, no. 10,
    pp. 1196–1203, Oct. 2021, doi: 10.1038/s41592-021-01252-x.
    
    12.  M. Zvyagin et al., “GenSLMs: Genome-scale language models
    reveal SARS-CoV-2 evolutionary dynamics,” 2022
    
    13.  S. Hayou, N. Ghosh, and B. Yu, “LoRA+: Efficient Low Rank Adaptation
    of Large Models,” July 04, 2024, arXiv: arXiv:2402.12354. doi:10.48550/arXiv.2402.12354
    
    14. M. Kanehisa and S. Goto, “KEGG: Kyoto Encyclopedia of Genes
    and Genomes”.
...

---

**Notes:**
- Aim for 15-20 high-quality references minimum
- Focus on recent work (last 5 years) unless citing seminal papers
- Include a mix of conference papers, journal articles, and technical reports
- Keep updating this document as you discover new relevant work