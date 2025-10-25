# Literature Review: Impact of Activation Functions on PEGASUS-X for Abstractive Text Summarization

**Student:** 210314E <br>
**Research Area:** NLP: Text Generation (Abstractive Summarization) <br>
**Date:** 2025-09-01 <br>

## Abstract

This literature review examines the role of activation functions in transformer-based models for abstractive text summarization, with a focus on the PEGASUS-X model. It covers the evolution of transformer architectures for long-document summarization, the theoretical underpinnings of activation functions, and their empirical impact on model performance. Key findings highlight the dominance of GELU in modern transformers, the potential benefits of alternatives like SiLU and ReLU, and the lack of comprehensive studies on activation function modifications in summarization-specific models. The review identifies opportunities for optimizing PEGASUS-X through activation function experimentation to improve efficiency and performance on long inputs.

## 1. Introduction

Abstractive text summarization aims to condense lengthy documents into concise, coherent summaries while preserving key information. Transformer-based models, particularly encoder-decoder architectures like PEGASUS and its extension PEGASUS-X, have achieved state-of-the-art results in this domain. However, the choice of activation functions within these models remains underexplored, despite their critical role in governing convergence, representational power, and overall performance. This review synthesizes literature on transformer architectures for summarization, activation functions in deep learning, and their interplay, providing a foundation for investigating activation function modifications in PEGASUS-X to enhance long-document summarization.

## 2. Search Methodology

### Search Terms Used
- Abstractive summarization
- Transformer models
- PEGASUS-X
- Activation functions
- Long document summarization

- Synonyms: Nonlinear activations, feed-forward networks, encoder-decoder architectures

### Databases Searched
- [x] IEEE Xplore
- [x] ACM Digital Library
- [x] Google Scholar
- [x] ArXiv
- [x] ACL Anthology
- [x] EMNLP Proceedings

### Time Period
2018-2024, with emphasis on recent advancements in transformer architectures and activation functions, while including seminal works from earlier periods.

## 3. Key Areas of Research

### 3.1 Transformer-Based Models for Abstractive Summarization
Transformer architectures have revolutionized abstractive summarization through self-attention mechanisms and large-scale pretraining. The original Transformer [Vaswani et al., 2017] introduced encoder-decoder structures with multi-head attention, forming the basis for models like BERT [Devlin et al., 2019] and GPT. In summarization, BART [Lewis et al., 2019] uses denoising objectives for pretraining, while T5 [Raffel et al., 2020] treats all tasks as text-to-text generation. PEGASUS [Zhang et al., 2020] employs gap-sentences generation for self-supervised pretraining, achieving strong performance on short inputs.

PEGASUS-X [Phang et al., 2023] extends PEGASUS for long documents (up to 16K tokens) by incorporating block-local attention, staggered blocks, and global tokens. This enables efficient processing of extended contexts, outperforming models like LongT5 [Guo et al., 2022] on benchmarks such as GovReport and arXiv.

**Key Papers:**
- Lewis et al. (2019) - Introduced BART, a denoising sequence-to-sequence model for generation tasks, including summarization.
- Zhang et al. (2020) - Proposed PEGASUS with gap-sentences generation, demonstrating superior performance on summarization benchmarks.
- Phang et al. (2023) - Developed PEGASUS-X with efficient attention for long inputs, achieving state-of-the-art results on long-document datasets.

### 3.2 Efficient Attention Mechanisms for Long Inputs
Standard transformers suffer from quadratic complexity in attention, limiting scalability. Solutions include sparse attention patterns: Longformer [Beltagy et al., 2020] uses local windows and global tokens; BigBird [Zaheer et al., 2020] combines local, random, and global attention; LongT5 [Guo et al., 2022] employs transient global attention. PEGASUS-X integrates block-local attention with staggered boundaries and global tokens to capture both local and global context efficiently.

Hierarchical approaches model document structure: HAT [Rohde et al., 2021] uses sentence-level and document-level layers; HMNet [Zhu et al., 2020] processes turns and meetings hierarchically. Multi-stage summarization, like SUMMN [Zhang et al., 2022], divides inputs into segments for intermediate summaries.

**Key Papers:**
- Beltagy et al. (2020) - Longformer enables linear-complexity attention for long sequences using local windows and global tokens.
- Zaheer et al. (2020) - BigBird introduces sparse random attention, scaling transformers to longer inputs.
- Guo et al. (2022) - LongT5 uses transient global attention for efficient long-sequence processing.

### 3.3 Activation Functions in Deep Neural Networks
Activation functions introduce nonlinearity, enabling complex feature learning. Early models used sigmoid and tanh, but saturating gradients led to vanishing issues [Hochreiter et al., 1997]. ReLU [Nair and Hinton, 2010] addressed this with piecewise linearity, promoting sparse representations. Variants like Leaky ReLU [Maas et al., 2013] and PReLU [He et al., 2015] handle negative inputs.

Smooth functions emerged: ELU [Clevert et al., 2016] and SELU [Klambauer et al., 2017] provide exponential behavior. GELU [Hendrycks and Gimpel, 2016], a probabilistic variant of ReLU, weights inputs by Gaussian probability, improving stability in deep networks. SiLU (Swish) [Ramachandran et al., 2018] combines sigmoid and linear, enhancing gradient flow. SwiGLU [Shazeer, 2020] gates linear units for better capacity.

In transformers, GELU is default (e.g., BERT, PEGASUS), but alternatives like Swish show promise in vision and NLP tasks [Ramachandran et al., 2018; Elfwing et al., 2018].

**Key Papers:**
- Hendrycks and Gimpel (2016) - Introduced GELU, adopted in major transformers for smoother gradients.
- Ramachandran et al. (2018) - Discovered Swish via architecture search, outperforming ReLU in some benchmarks.
- Dubey et al. (2022) - Comprehensive survey of activation functions, benchmarking performance across tasks.

### 3.4 Activation Functions in Transformer Models
Transformers rarely revisit activations post-initial design. The original Transformer used ReLU [Vaswani et al., 2017], but BERT shifted to GELU. Fang et al. [2023] introduced learnable rational activations, improving GLUE and SQuAD scores over fixed GELU. In NLP, Eger et al. [2018] compared 21 activations, finding penalized tanh effective.

Summarization models inherit activations without modification. No studies specifically evaluate ReLU vs. GELU in summarization, though GELU's smoothness aids long-context tasks.

**Key Papers:**
- Fang et al. (2023) - Learnable activations outperform fixed GELU in BERT-like models.
- Eger et al. (2018) - Benchmark of activations across NLP tasks, highlighting stability of alternatives.

## 4. Research Gaps and Opportunities

### Gap 1: Lack of Activation Function Studies in Summarization Models
While activation functions are explored in general NLP and vision, their impact on summarization-specific models like PEGASUS-X is unexamined. Existing work focuses on attention or pretraining, ignoring nonlinearity choices.

**Why it matters:** Activation functions affect convergence and representational power, potentially optimizing performance on long documents without architectural overhaul.

**How this project addresses it:** Experimentally replace GELU with ReLU and SiLU in PEGASUS-X, evaluating on diverse datasets to quantify trade-offs.

### Gap 2: Limited Exploration of Alternatives for Long-Document Tasks
Long-document summarization emphasizes efficiency, but activation smoothness for gradient stability in extended contexts is underexplored.

**Why it matters:** Smoother activations may improve stability and quality for complex inputs like GovReport.

**How this project addresses it:** Fine-tune PEGASUS-X variants on long datasets, measuring ROUGE scores to identify optimal activations.

## 5. Theoretical Framework

Activation functions modulate neuron outputs, influencing gradient flow and representational capacity. GELU's probabilistic gating [Hendrycks and Gimpel, 2016] provides smoother transitions than ReLU's hard thresholding, reducing inactive neurons and enhancing feature learning. In transformers, this aids attention mechanisms in capturing nuanced semantics for summarization. Theoretical analyses [Ramachandran et al., 2018] suggest non-monotonic functions like SiLU balance sparsity and smoothness, potentially benefiting deep encoder-decoder structures.

## 6. Methodology Insights

Common methodologies include fine-tuning pretrained models on summarization datasets, evaluating with ROUGE metrics. Experimental setups vary hyperparameters like learning rate and batch size, using libraries like Hugging Face Transformers. Promising approaches for this project include systematic activation replacement during fine-tuning, controlled comparisons across datasets, and ablation studies on convergence and performance.

## 7. Conclusion

The literature underscores transformers' dominance in summarization, PEGASUS-X's efficiency for long inputs, and GELU's prevalence in activations. However, gaps in activation-specific studies for summarization present opportunities for optimization. This review informs the project's focus on activation modifications to enhance PEGASUS-X, potentially yielding better ROUGE scores and stability. Future work could explore learnable activations or hybrid functions.

## References

[1] M. Lewis, Y. Liu, N. Goyal, et al., "BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension," arXiv preprint arXiv:1910.13461, 2019.

[2] Y. Liu, "Fine-tune BERT for extractive summarization," arXiv preprint arXiv:1903.10318, 2019.

[3] I. Beltagy, M. E. Peters, and A. Cohan, "Longformer: The long-document transformer," arXiv preprint arXiv:2004.05150, 2020.

[4] J. Zhang, Y. Zhao, M. Saleh, and P. Liu, "PEGASUS: Pre-training with extracted gap-sentences for abstractive summarization," in International conference on machine learning, PMLR, 2020, pp. 11 328–11 339.

[5] J. Phang, Y. Zhao, and P. Liu, "Investigating efficiently extending transformers for long input summarization," in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, Singapore: Association for Computational Linguistics, Dec. 2023, pp. 3946–3961.

[6] A. Cohan, F. Dernoncourt, D. S. Kim, et al., "A discourse-aware attention model for abstractive summarization of long documents," in Proceedings of NAACL-HLT, 2018, pp. 615–626.

[7] L. Huang, D. Wu, P. Wang, et al., "Efficient attentions for long document summarization," in Findings of ACL, 2021, pp. 1412–1426.

[8] M. Zaheer, G. Guruganesh, K. A. Dubey, et al., "Big bird: Transformers for longer sequences," Advances in neural information processing systems, vol. 33, pp. 17 283–17 297, 2020.

[9] M. Guo, J. Ainslie, D. Uthus, et al., "LongT5: Efficient text-to-text transformer for long sequences," in Findings of the Association for Computational Linguistics: NAACL 2022, M. Carpuat, M.-C. de Marneffe, and I. V. Meza Ruiz, Eds., Seattle, United States: Association for Computational Linguistics, Jul. 2022, pp. 724–736.

[10] T. Rohde, X. Wu, and Y. Liu, "Hierarchical learning for generation with long source sequences," arXiv preprint arXiv:2104.07545, 2021.

[11] C. Zhu, R. Xu, M. Zeng, and X. Huang, "A hierarchical network for abstractive meeting summarization with cross-domain pretraining," in Findings of the Association for Computational Linguistics: EMNLP 2020, T. Cohn, Y. He, and Y. Liu, Eds., Online: Association for Computational Linguistics, Nov. 2020, pp. 194–203.

[12] Y. Zhang, A. Ni, Z. Mao, et al., "SummN: A multi-stage summarization framework for long input dialogues and documents," in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), S. Muresan, P. Nakov, and A. Villavicencio, Eds., Dublin, Ireland: Association for Computational Linguistics, May 2022, pp. 1592–1604.

[13] A. Karotia and S. Susan, "BioLayAKSS at BioLaySumm: Domain adaptation by two-stage fine-tuning of large language models used for biomedical lay summary generation," in Proceedings of the 23rd Workshop on Biomedical Natural Language Processing, D. Demner-Fushman, S. Ananiadou, M. Miwa, K. Roberts, and J. Tsujii, Eds., Bangkok, Thailand: Association for Computational Linguistics, Aug. 2024, pp. 762–768.

[14] W. Xiong, A. Gupta, S. Toshniwal, Y. Mehdad, and S. Yih, "Adapting pretrained text-to-text models for long text sequences," in Findings of the Association for Computational Linguistics: EMNLP 2023, H. Bouamor, J. Pino, and K. Bali, Eds., Singapore: Association for Computational Linguistics, Dec. 2023, pp. 5566–5578.

[15] D. Hendrycks and K. Gimpel, "Gaussian Error Linear Units (GELUs)," arXiv preprint arXiv:1606.08415, 2016.

[16] P. Ramachandran, B. Zoph, and Q. V. Le, "Searching for activation functions," in International Conference on Learning Representations (ICLR) Workshop, 2018.

[17] N. Shazeer, "GLU variants improve transformer," arXiv preprint arXiv:2002.05202, 2020.

[18] C.-Y. Lin, "ROUGE: A package for automatic evaluation of summaries," in Text Summarization Branches Out: Proceedings of the ACL-04 Workshop, Barcelona, Spain: Association for Computational Linguistics, 2004, pp. 74–81.

[19] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, "BLEU: A method for automatic evaluation of machine translation," in Proceedings of ACL, 2002, pp. 311–318.

[20] T. Zhang, V. Kishore, F. Wu, K. Q. Weinberger, and Y. Artzi, "BERTScore: Evaluating text generation with BERT," in International Conference on Learning Representations (ICLR), 2020.

[21] E. Sharma, C. Li, and L. Wang, "BIGPATENT: A large-scale dataset for abstractive and coherent summarization," in Proceedings of ACL, 2019, pp. 2204–2213.

[22] A. See, P. J. Liu, and C. D. Manning, "Get to the point: Summarization with pointer-generator networks," in Proceedings of ACL, 2017, pp. 1073–1083.

[23] S. Narayan, S. B. Cohen, and M. Lapata, "Don't give me the details, just the summary! Topic-aware convolutional neural networks for extreme summarization," in Proceedings of EMNLP, 2018, pp. 1797–1807.

[24] H. Fang, J.-U. Lee, N. S. Moosavi, and I. Gurevych, "Transformers with learnable activation functions," in Findings of the Association for Computational Linguistics: EACL 2023, A. Vlachos and I. Augenstein, Eds., Dubrovnik, Croatia: Association for Computational Linguistics, May 2023, pp. 2382–2398.

[25] S. Eger, P. Youssef, and I. Gurevych, "Is it time to swish? comparing deep learning activation functions across NLP tasks," in Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, E. Riloff, D. Chiang, J. Hockenmaier, and J. Tsujii, Eds., Brussels, Belgium: Association for Computational Linguistics, Oct. 2018, pp. 4415–4424.

[26] S. R. Dubey, S. K. Singh, and B. B. Chaudhuri, "Activation functions in deep learning: A comprehensive survey and benchmark," Neurocomputing, vol. 503, pp. 92–108, 2022.

[27] A. Vaswani, N. Shazeer, N. Parmar, et al., "Attention is all you need," arXiv preprint arXiv:1706.03762, 2017.

[28] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), J. Burstein, C. Doran, and T. Solorio, Eds., Minneapolis, Minnesota: Association for Computational Linguistics, Jun. 2019, pp. 4171–4186.

[29] C. Raffel, N. Shazeer, A. Roberts, et al., "Exploring the limits of transfer learning with a unified text-to-text transformer," Journal of Machine Learning Research, vol. 21, no. 140, pp. 1–67, 2020.

[30] S. Hochreiter, Y. Bengio, P. Frasconi, and J. Schmidhuber, "Gradient flow in recurrent nets: the difficulty of learning long-term dependencies," in A Field Guide to Dynamical Recurrent Neural Networks, IEEE Press, 1997.

[31] V. Nair and G. E. Hinton, "Rectified linear units improve restricted boltzmann machines," in Proceedings of the 27th International Conference on International Conference on Machine Learning, ser. ICML'10, Haifa, Israel: Omnipress, 2010, pp. 807–814.

[32] A. L. Maas, "Rectifier nonlinearities improve neural network acoustic models," 2013.

[33] K. He, X. Zhang, S. Ren, and J. Sun, "Delving deep into rectifiers: Surpassing human-level performance on imagenet classification," ser. ICCV '15, USA: IEEE Computer Society, 2015, pp. 1026–1034.

[34] D.-A. Clevert, T. Unterthiner, and S. Hochreiter, "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)," arXiv preprint arXiv:1511.07289, 2016.

[35] G. Klambauer, T. Unterthiner, A. Mayr, and S. Hochreiter, "Self-normalizing neural networks," in Proceedings of the 31st International Conference on Neural Information Processing Systems, ser. NIPS'17, Long Beach, California, USA: Curran Associates Inc., 2017, pp. 972–981.

[36] S. Elfwing, E. Uchibe, and K. Doya, "Sigmoid-weighted linear units for neural network function approximation in reinforcement learning," Neural Networks, vol. 107, pp. 3–11, 2018.