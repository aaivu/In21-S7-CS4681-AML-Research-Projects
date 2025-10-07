# Literature Review: Healthcare AI:Medical Imaging

**Student:** 210329E
**Research Area:** Healthcare AI:Medical Imaging
**Date:** 2025-09-01

## Abstract

This literature review investigates advancements in deep learning for medical imaging, focusing on multi-label thoracic disease classification using chest X-rays. It synthesizes research across four interconnected domains: (A) benchmarking architectures for thoracic disease detection, (B) the theoretical framework and clinical relevance of Uncertainty Quantification (UQ), (C) probabilistic and non-probabilistic UQ methodologies such as Monte Carlo Dropout (MCD) and Deep Ensembles (DE), and (D) specialized multi-label loss functions addressing class imbalance and label ambiguity. The review finds that while MCD offers computational efficiency, Deep Ensembles achieve superior calibration, reliability, and uncertainty decomposition, making them more suitable for clinical deployment. Furthermore, the predominance of aleatoric uncertainty in datasets like NIH ChestX-ray14 underscores the importance of data quality and label fidelity over architectural complexity. Collectively, these insights establish the empirical and theoretical basis for developing uncertainty-aware, trustworthy diagnostic systems in high-stakes healthcare applications.

## 1. Introduction

The integration of artificial intelligence (AI) into medical imaging has transformed diagnostic research and clinical decision support, particularly through the adoption of deep learning (DL) models. These systems, exemplified by CheXNet, have achieved radiologist-level performance in detecting and classifying thoracic diseases from chest X-rays, which represent one of the most widely used and clinically significant imaging modalities. Despite such achievements, the deterministic nature of conventional deep learning models remains a critical limitation, as they fail to convey how confident the model is in its predictions.

In high-stakes clinical environments, predictive uncertainty is as important as accuracy. An incorrect diagnosis made with high confidence can lead to severe clinical consequences. This challenge has led to growing research interest in Uncertainty Quantification (UQ), which enables models to represent confidence and distinguish between aleatoric uncertainty (irreducible data noise) and epistemic uncertainty (limited model knowledge). By providing interpretable confidence estimates, UQ helps clinicians identify ambiguous or out-of-distribution cases that require human review.

This literature review focuses on the intersection of deep learning and uncertainty quantification for multi-label thoracic disease classification. It examines four core research areas: (A) benchmarking architectures and reproducibility in chest X-ray classification, (B) the theoretical framework and clinical significance of UQ, (C) comparative methodologies in probabilistic and non-probabilistic UQ, and (D) specialized multi-label loss functions addressing class imbalance and label ambiguity. Through this synthesis, the review establishes a foundation for developing uncertainty-aware, reliable, and clinically interpretable AI diagnostic systems.

## 2. Search Methodology

### Search Terms Used

#### Primary Search Terms:

Core Terms: "uncertainty quantification", "deep learning", "medical imaging", "chest X-ray classification"

Architectural Terms: "deep ensembles", "Monte Carlo dropout", "Bayesian neural networks", "DenseNet", "CheXNet"

UQ-Specific: "aleatoric uncertainty", "epistemic uncertainty", "predictive calibration", "Expected Calibration Error"

Multi-label Learning: "multi-label classification", "class imbalance", "focal loss", "ZLPR loss"

Clinical Context: "computer-aided diagnosis", "radiologist-level performance", "clinical decision support"

#### Synonyms and Variations Used:

- "UQ in medical AI"

- "Predictive uncertainty in deep learning"

- "CheXNet reproducibility"

- "ChestX-ray14 dataset"

- "Calibration in deep neural networks"

### Databases Searched

- [x] IEEE Xplore
- [x] ACM Digital Library
- [x] Google Scholar
- [x] ArXiv
- [x] NeurIPS Proceedings
- [x] Other: PubMed Central, Medical Image Analysis Journals

### Time Period

Primary Focus: 2017–2025 (8 years)

#### Rationale:
This range was selected to encompass the period starting from the release of CheXNet (2017), which established a key benchmark for thoracic disease classification, through to the latest advancements in uncertainty quantification and ensemble-based deep learning.

## 3. Key Areas of Research

### 3.1 Benchmarking in Thoracic Disease Classification

This area focuses on establishing reliable baselines for automated thoracic disease classification using chest X-rays. The **NIH ChestX-ray14** dataset, comprising over 112,000 frontal-view X-rays labeled for 14 pathologies, remains the most widely used benchmark. However, label noise from NLP-extracted radiology reports introduces significant uncertainty, estimated at nearly 10%. Studies have also emphasized issues such as class imbalance and label co-occurrence, which complicate multi-label learning and hinder reproducibility across studies.

**Key Papers:**

* **Rajpurkar et al., 2017** – Introduced *CheXNet*, a DenseNet-121 model achieving radiologist-level performance in pneumonia detection, setting a foundation for large-scale thoracic disease classification.
* **Strick et al., 2025** – Proposed *DannyNet*, a reproducible DenseNet variant validated on ChestX-ray14, improving consistency and evaluation transparency over CheXNet.

---

### 3.2 Theoretical Framework and Importance of Uncertainty

Uncertainty Quantification (UQ) plays a vital role in ensuring clinical reliability by distinguishing between *aleatoric* uncertainty (inherent data noise) and *epistemic* uncertainty (model-related uncertainty reducible with more data). In medical imaging, this framework supports safer deployment by flagging ambiguous cases for expert review and enhancing model interpretability.

**Key Papers:**

* **Kendall and Gal, 2017** – Formally decomposed total predictive uncertainty into aleatoric and epistemic components, introducing probabilistic modeling for computer vision tasks.
* **Baur et al., 2025** – Reviewed recent UQ strategies in multi-label chest X-ray classification, emphasizing the need for better disentanglement of uncertainty sources in clinical applications.

---

### 3.3 Probabilistic Methods in Uncertainty Quantification

Probabilistic approaches estimate full predictive distributions to capture model confidence. Techniques range from fully Bayesian neural networks to approximate methods such as Monte Carlo Dropout and Deep Ensembles. These methods enhance trustworthiness but often trade off computational efficiency for statistical rigor.

**Key Papers:**

* **Gal and Ghahramani, 2016** – Proposed *Monte Carlo Dropout* (MCD) as a practical approximation to Bayesian inference through stochastic forward passes.
* **Lakshminarayanan et al., 2017** – Introduced *Deep Ensembles (DE)* for robust UQ, demonstrating improved calibration and out-of-distribution detection compared to MCD.
* **Papamarkou et al., 2022** – Discussed the limitations of Markov Chain Monte Carlo (MCMC) methods for Bayesian neural networks in high-dimensional parameter spaces.
* **Whata et al., 2024** – Combined MCD and DE through hybrid ensemble techniques to improve calibration in chest X-ray classification tasks.

---

### 3.4 Non-Probabilistic Methods in Uncertainty Quantification

Non-probabilistic techniques estimate uncertainty using deterministic models or post-hoc calibration methods without explicit probabilistic modeling. These approaches are computationally efficient and can be integrated into pre-trained models to improve reliability in decision-making.

**Key Papers:**

* **Guo et al., 2017** – Introduced *Temperature Scaling* for post-hoc calibration of softmax outputs, significantly reducing confidence overestimation in deep neural networks.
* **Angelopoulos and Bates, 2021** – Presented *Conformal Prediction*, offering statistical coverage guarantees for prediction sets in deep learning applications.
* **Zeng et al., 2025** – Demonstrated improved calibration using temperature scaling in complex biological data, reinforcing its adaptability to diverse domains.

---

### 3.5 Multi-Label Classification Loss Functions

Multi-label thoracic disease classification requires loss functions capable of addressing class imbalance and inter-label dependencies. Advanced loss designs help models learn from underrepresented classes and uncertain labels, improving robustness in noisy medical datasets such as ChestX-ray14.

**Key Papers:**

* **Lin et al., 2017** – Proposed *Focal Loss* to focus training on hard-to-classify examples and mitigate imbalance in large datasets.
* **Su et al., 2022** – Introduced *ZLPR Loss* to handle uncertain label counts and exploit correlations among co-occurring thoracic pathologies.

## 4. Research Gaps and Opportunities

### Gap 1: Reproducibility and Stability in Benchmark Models

**Why it matters:** Foundational benchmarks such as CheXNet have demonstrated high accuracy but exhibit metric inconsistencies when evaluated on public datasets. This instability limits the ability to reliably compare models and hinders the integration of uncertainty quantification in clinical settings. Unreliable baselines can reduce trust in AI-driven diagnostic tools and may lead to incorrect clinical decisions.
**How my project addresses it:** This project adopts reproducible baseline models, such as DannyNet, and explores architectural diversity in Deep Ensembles to establish stable, reliable platforms for uncertainty-aware thoracic disease classification.

### Gap 2: Robust Integration of Uncertainty Quantification Without Performance Degradation

**Why it matters:** Approximate probabilistic methods, such as Monte Carlo Dropout, can compromise classification accuracy or calibration, particularly in multi-label tasks. Full Bayesian neural networks, while theoretically sound, are computationally intensive and impractical for large datasets like NIH ChestX-ray14. Overconfident predictions in high-stakes medical applications pose significant risks to patient safety.
**How my project addresses it:** By implementing Deep Ensembles combined with specialized multi-label loss functions, the project achieves reliable uncertainty decomposition while maintaining or improving classification performance. This approach ensures that uncertainty measures enhance clinical trust rather than detract from model effectiveness.

### Gap 3: Disentanglement of Aleatoric and Epistemic Uncertainty in Noisy Medical Datasets

**Why it matters:** Chest X-ray datasets often contain label noise, ambiguities, and class imbalance, making it difficult to identify whether predictive uncertainty arises from inherent data limitations (aleatoric) or model deficiencies (epistemic). Without this understanding, interventions may target the wrong source, limiting improvements in diagnostic reliability.
**How my project addresses it:** The project explicitly formulates uncertainty decomposition within ensemble models, enabling clear separation of aleatoric and epistemic contributions. This allows clinicians and researchers to identify whether performance gains require improved data quality, enhanced model capacity, or both.

## 5. Theoretical Framework

The theoretical foundation for uncertainty-aware thoracic disease classification is based on **Deep Ensemble (DE) construction** and **uncertainty decomposition**, enabling interpretable and reliable predictions in multi-label chest X-ray tasks.

### 5.1 Deep Ensemble Construction

The ensemble consists of (M = 9) carefully selected models, chosen based on complementary architectures, high test AUROC, and F1 scores across 14 trials. Let (p_k^{(m)}) denote the predicted probability for class (k) from the (m)-th model. The ensemble prediction (\bar{p}_k) is computed as a uniform average over all members:

[
\bar{p}*k = \frac{1}{M} \sum*{m=1}^{M} p_k^{(m)}
]

This approach ensures **functional diversity**, improves robustness, and provides a practical approximation of Bayesian model averaging without excessive computational overhead.

### 5.2 Uncertainty Decomposition

The ensemble's prediction uncertainty is decomposed into **Total Uncertainty (TU)**, **Aleatoric Uncertainty (AU)**, and **Epistemic Uncertainty (EU)** as follows:

* **Total Uncertainty (TU):** Measures the overall uncertainty in the ensemble prediction:
  [
  TU = - \sum_{k} \bar{p}_k \log \bar{p}_k
  ]

* **Aleatoric Uncertainty (AU):** Captures irreducible uncertainty inherent in the data, such as noise, artifacts, or ambiguous labels:
  [
  AU = \frac{1}{M} \sum_{m=1}^{M} \left( - \sum_{k} p_k^{(m)} \log p_k^{(m)} \right)
  ]

* **Epistemic Uncertainty (EU):** Quantifies uncertainty due to the model's lack of knowledge and is reducible with additional data or improved architectures:
  [
  EU = TU - AU
  ]

This decomposition allows for **interpretable predictions**, highlighting cases where the model is uncertain due to data limitations versus those arising from insufficient model knowledge. Clinically, high epistemic uncertainty can signal the need for expert review or further testing.

## 6. Methodology Insights

### 6.1 Commonly Used Methodologies

Research in thoracic disease classification and uncertainty-aware medical imaging employs a combination of **deep learning architectures, uncertainty quantification methods, and multi-label loss functions**:

* **Architecture Selection:** Transfer learning using ImageNet-pretrained backbones, such as DenseNet-121 and EfficientNet, is standard. DenseNet-121 remains widely used due to its feature reuse efficiency and CheXNet benchmark performance, while attention mechanisms (e.g., CBAM, self-attention) are increasingly incorporated for improved localization.
* **Training Protocols:** Full fine-tuning is preferred over frozen-backbone approaches. Optimizers like AdamW with weight decay and learning rate scheduling (e.g., ReduceLROnPlateau) are commonly used. Patient-level data splitting is critical to prevent data leakage.
* **Uncertainty Quantification:** Probabilistic methods include Monte Carlo Dropout (MCD) and Deep Ensembles (DE). MCD uses multiple stochastic forward passes to approximate predictive distributions, while DE trains diverse models with different architectures and initializations to approximate the posterior over functions. Non-probabilistic methods like Temperature Scaling provide post-hoc calibration.
* **Loss Functions:** Specialized multi-label losses such as Binary Cross Entropy Loss, Focal Loss and ZLPR Loss are used for addressing class imbalance, rare pathologies, and inter-label correlations in datasets like NIH ChestX-ray14.
* **Evaluation Metrics:** AUROC and F1 Score are primary metrics for classification performance, while Expected Calibration Error (ECE), Negative Log-Likelihood (NLL), and Brier Score are used to assess calibration. Uncertainty decomposition metrics—Total Uncertainty (TU), Aleatoric Uncertainty (AU), and Epistemic Uncertainty (EU)—allow granular evaluation of prediction reliability.

### 6.2 Most Promising Approaches

* **High-Diversity Deep Ensembles:** Selecting ensembles with heterogeneous architectures and diverse loss functions improves uncertainty decomposition and out-of-distribution detection compared to ensembles differing only in initialization.
* **Advanced Multi-Label Loss Functions:** ZLPR Loss effectively handles label co-occurrence patterns and rare pathologies, offering advantages over standard Focal Loss.
* **Preprocessing Enhancements:** Techniques like CLAHE (Contrast Limited Adaptive Histogram Equalization) enhance local contrast in chest X-rays, improving feature visibility.
* **Integrated Explainability:** Ensemble Grad-CAM averaging provides more robust visual explanations than single-model attributions, potentially linking uncertainty measures with interpretable regions.
* **Data-Centric Quality Improvements:** Since aleatoric uncertainty dominates in chest X-ray datasets, prioritizing uncertainty-aware labeling protocols where radiologists flag ambiguous cases can improve model reliability beyond purely architectural improvements. 

Overall, **Deep Ensembles combined with specialized multi-label losses and careful preprocessing** represent the most effective methodology for building robust, clinically interpretable thoracic disease classifiers with reliable uncertainty estimates.

## 7. Conclusion

This literature review provides a comprehensive synthesis of research on deep learning for multi-label thoracic disease classification and the role of Uncertainty Quantification (UQ) in medical imaging. Key findings highlight that while foundational models such as CheXNet set strong benchmarks for classification accuracy, they suffer from reproducibility issues and a lack of calibrated confidence measures. Theoretical and empirical studies show that uncertainty can be effectively decomposed into **aleatoric** (data-inherent) and **epistemic** (model-driven) components, which are essential for trustworthy predictions in high-stakes clinical environments.

Methodologically, probabilistic approaches like **Deep Ensembles** outperform approximations such as Monte Carlo Dropout by providing more reliable calibration, robust out-of-distribution detection, and interpretable uncertainty estimates. Specialized multi-label loss functions, including Focal and ZLPR Loss, address class imbalance and label correlations, improving performance in real world datasets like NIH ChestX-ray14. Non-probabilistic calibration methods, such as Temperature Scaling and Conformal Prediction, complement probabilistic techniques but have limitations in handling out-of-distribution cases.

The review identifies critical gaps in reproducibility, uncertainty integration without accuracy degradation, and disentanglement of uncertainty in noisy datasets. These insights guide the research direction toward designing **ensemble-based, uncertainty-aware frameworks** that combine diverse architectures, calibrated outputs, and specialized loss functions. By prioritizing both predictive accuracy and interpretable uncertainty, future work aims to bridge the gap between experimental AI models and clinically reliable diagnostic tools, supporting informed decision-making in thoracic disease diagnosis.

## References

1. D. Strick, C. Garcia, and A. Huang, "Reproducing and Improving CheXNet: Deep Learning for Chest X-ray Disease Classification," May 10, 2025, arXiv: arXiv:2505.06646. doi: 10.48550/arXiv.2505.06646.

2. "NIH Chest X-rays," Kaggle, Feb. 21, 2018. Available at: https://www.kaggle.com/datasets/nih-chest-xrays/data.

3. P. Rajpurkar, J. Irvin, K. Zhu, B. Yang, H. Mehta, T. Duan, D. Ding, A. Bagul, C. Langlotz, K. Shpanskaya, M. P. Lungren, and A. Y. Ng, "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning," Dec. 25, 2017, arXiv: arXiv:1711.05225. doi: 10.48550/arXiv.1711.05225.

4. A. Kendall and Y. Gal, "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?," in Advances in Neural Information Processing Systems, Curran Associates, Inc., 2017. Accessed: Aug. 24, 2025. [Online]. Available: https://papers.nips.cc/paper files/paper/2017/hash/2650d6089a6d640c5e85b2b88265dc2b-Abstract.html.

5. S. Baur, W. Samek, and J. Ma, "Benchmarking Uncertainty and its Disentanglement in multi-label Chest X-Ray Classification," Aug. 06, 2025, arXiv: arXiv:2508.04457. doi: 10.48550/arXiv.2508.04457.

6. M. A. Chan, M. J. Molina, and C. A. Metzler, "Estimating Epistemic and Aleatoric Uncertainty with a Single Model."

7. T. Papamarkou, J. Hinkle, M. T. Young, and D. Womble, "Challenges in Markov Chain Monte Carlo for Bayesian Neural Networks," Stat. Sci., vol. 37, no. 3, Aug. 2022, doi: 10.1214/21-STS840.

8. C. C. Margossian, L. Pillaud-Vivien, and L. K. Saul, "Variational Inference for Uncertainty Quantification: an Analysis of Trade-offs," May 06, 2025, arXiv: arXiv:2403.13748. doi: 10.48550/arXiv.2403.13748.

9. M. Hasan, A. Khosravi, I. Hossain, A. Rahman, and S. Nahavandi, "Controlled Dropout for Uncertainty Estimation," May 06, 2022, arXiv: arXiv:2205.03109. doi: 10.48550/arXiv.2205.03109.

10. "Deep ensembles - AWS Prescriptive Guidance." Accessed: Aug. 24, 2025. [Online]. Available: https://docs.aws.amazon.com/prescriptive-guidance/latest/ml-quantifying-uncertainty/deep-ensembles.html

11. A. Whata, K. Dibeco, K. Madzima, and I. Obagbuwa, "Uncertainty quantification in multi-class image classification using chest X-ray images of COVID-19 and pneumonia," Front. Artif. Intell., vol. 7, Sept. 2024, doi: 10.3389/frai.2024.1410841.

12. S. Lee, "Advanced Uncertainty Estimation Methods." Accessed: Aug. 24, 2025. [Online]. Available: https://www.numberanalytics.com/blog/advanced-uncertainty-estimation-methods-medical-imaging

13. B. Lakshminarayanan, A. Pritzel, and C. Blundell, "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles," in Advances in Neural Information Processing Systems, Curran Associates, Inc., 2017. Accessed: Aug. 24, 2025. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html

14. "Estimating Epistemic and Aleatoric Uncertainty with a Single Model." Accessed: Aug. 24, 2025. [Online]. Available: https://arxiv.org/html/2402.03478v2

15. C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On Calibration of Modern Neural Networks," Aug. 03, 2017, arXiv: arXiv:1706.04599. doi: 10.48550/arXiv.1706.04599.

16. X. Zeng, H. Wang, L. Zhao, Y. Cheng, D. Zhou, and S. Shi, "Uncertainty Quantification and Temperature Scaling Calibration for Protein-RNA Binding Site Prediction," J. Chem. Inf. Model., vol. 65, no. 12, pp. 6310–6321, June 2025, doi: 10.1021/acs.jcim.5c00556.

17. Y. N. Kunang, S. Nurmaini, D. Stiawan, and B. Y. Suprapto, "Deep learning with focal loss approach for attacks classification," TELKOMNIKA (Telecommunication Computing Electronics and Control), vol. 19, no. 4, p. 1407, Aug. 2021, doi: 10.12928/telkomnika.v19i4.18772.

18. J. Su, M. Zhu, A. Murtadha, S. Pan, B. Wen, and Y. Liu, "ZLPR: A novel loss for multi-label classification," arXiv.org, Aug. 05, 2022.

---
