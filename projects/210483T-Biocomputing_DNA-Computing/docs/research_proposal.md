# Research Proposal: Biocomputing:DNA Computing

  

**Student:** 210483T

**Research Area:** Biocomputing:DNA Computing

**Date:** 2025-09-01

  

## Abstract

  

Foundation models trained on DNA sequences have shown promise in genomics but remain underexplored for complex reasoning tasks like disease prediction. This research evaluates two adaptation strategies for the 500M-parameter Nucleotide Transformer (NT-500M): parameter-efficient fine-tuning using Low-Rank Adaptation (LoRA) and linear probing with frozen embeddings and a lightweight MLP classifier. Using a KEGG-derived biological reasoning dataset containing reference and variant DNA sequences, we compare these approaches across multiple evaluation metrics. Experimental results reveal that the embedding-based approach outperforms LoRA, achieving 91.78% accuracy and 78.68% F1-score. This work demonstrates that leveraging pretrained DNA embeddings with a simple downstream model provides an efficient and effective alternative to heavy fine-tuning, with potential applications in variant-disease prediction and computational genomics.

  

## 1. Introduction

  

Large pretrained DNA foundation models such as Nucleotide Transformers have transformed computational genomics by enabling knowledge transfer through contextual embeddings. However, applying them effectively for reasoning-based disease prediction tasks remains challenging. Traditional fine-tuning is computationally expensive and complex. Parameter-efficient methods like LoRA and lightweight downstream classifiers offer promising alternatives. This research aims to evaluate and optimize adaptation strategies for NT-500M to improve genomic disease prediction accuracy and efficiency.

  

## 2. Problem Statement

  

Although DNA foundation models capture rich sequence information, their downstream performance in disease prediction remains suboptimal. Full fine-tuning requires significant resources, and hybrid LLM approaches are complex. There is a need for an efficient strategy that leverages pretrained DNA embeddings to achieve high accuracy in disease classification with minimal computational cost.

  

## 3. Literature Review Summary

  

Previous works such as DNABERT and Nucleotide Transformer established transformer-based models for genomic sequences, achieving state-of-the-art performance on various prediction tasks. Later research like BioReason combined DNA embeddings with LLMs for reasoning, improving performance but at high computational cost. LoRA has emerged as a parameter-efficient fine-tuning method, while linear probing demonstrates strong performance in NLP and other domains. However, a systematic comparison of these strategies on reasoning-based genomic disease prediction tasks remains limited. This research addresses that gap.

  

## 4. Research Objectives

  

### Primary Objective

To compare and evaluate LoRA fine-tuning and linear probing strategies for adapting NT-500M in genomic disease prediction using a KEGG-derived dataset.

  

### Secondary Objectives

- Evaluate performance of LoRA fine-tuning on variant and reference sequences.

- Investigate the effectiveness of frozen NT embeddings with an MLP classifier.

- Identify the most efficient and accurate adaptation strategy for genomic disease prediction.

  

## 5. Methodology

  

Dataset: KEGG-derived dataset containing reference and variant sequences for 37 disease classes.

Preprocessing: Tokenization and embedding extraction using NT-500M.

Approach 1: LoRA fine-tuning on variant and reference sequences with cross-entropy loss.

Approach 2: Linear probing using frozen embeddings fed into an MLP classifier.

Evaluation: Accuracy, precision, recall, and F1-score across configurations.

  

## 6. Expected Outcomes

  

1. A clear performance comparison between LoRA fine-tuning and linear probing on DNA foundation models.
2. Empirical evidence that embedding-based classification achieves superior results with lower resource requirements.
3. A scalable and efficient framework for applying Nucleotide Transformers to disease prediction tasks.

  
  

## 7. Timeline

  

| Week | Task |

|------|------|

| 1-2 | Literature Review |

| 3-4 | Methodology Development |

| 5-8 | Implementation |

| 9-12 | Experimentation |

| 13-15| Analysis and Writing |

| 16 | Final Submission |

  

## 8. Resources Required

Dataset:  KEGG-derived DNA variant-disease dataset
Models: NT-500M (pretrained Nucleotide Transformer)
Tools: Python, PyTorch, LoRA PEFT library, MLP implementation
Hardware: GPU (for fine-tuning), CPU (for embedding extraction and classification)

## References

  

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

---

  

**Submission Instructions:**

1. Complete all sections above

2. Commit your changes to the repository

3. Create an issue with the label "milestone" and "research-proposal"

4. Tag your supervisors in the issue for review