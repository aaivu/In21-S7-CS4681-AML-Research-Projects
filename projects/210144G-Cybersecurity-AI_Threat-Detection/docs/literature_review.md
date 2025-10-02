# Literature Review: Cybersecurity AI:Threat Detection

**Student:** 210144G
**Research Area:** Cybersecurity AI:Threat Detection
**Date:** 2025-10-02

## Abstract

This review synthesizes AI-driven intrusion detection with emphasis on anomaly-based methods, deep learning architectures, and ensemble fusion for network threat detection. We examine the shift from signature-based to behavior-centric IDS; compare supervised, unsupervised, and hybrid deep models; evaluate ensemble/stacking strategies; and discuss metrics and datasets (notably NSL-KDD). Key findings: (i) attack-specialized submodels fused via a meta-classifier can improve minority-class recall (U2R, R2L) without inflating false alarms; (ii) precision–recall–oriented evaluation and threshold optimization are essential under severe class imbalance; (iii) reliance on aging datasets limits external validity; and (iv) practical deployment needs real-time efficiency, drift handling, and explainability. These insights inform the project’s focus on imbalance-aware learning, robust evaluation, and deployment-conscious design.

## 1. Introduction

The proliferation and complexity of cyber threats render purely signature-based defenses insufficient for detecting novel or morphing attacks. Intrusion Detection Systems (IDS) increasingly leverage machine learning (ML) and deep learning (DL) to learn normal behavior and identify anomalies, enabling zero-day detection [1], [3]. Practitioner reports also highlight AI’s role in adaptive threat detection and response in operational settings [8], [9], while public resources summarize IDS functions and deployment patterns [10], [11]. One promising paradigm trains attack-specialized deep models for categories such as DoS, Probe, R2L, and U2R and fuses their outputs via a meta-classifier to improve robustness—particularly for minority classes—on NSL-KDD. This review situates such approaches within broader research on anomaly-based IDS, representation learning (autoencoders/VAEs, CNN/RNN/transformers, tabular DL), ensembles, and class-imbalance strategies, along with dataset and metric practices.

## 2. Search Methodology

### Search Terms Used
- intrusion detection, anomaly-based IDS, network anomaly detection
- deep learning, autoencoder, VAE, LSTM, CNN, transformer, tabular DL
- ensemble learning, stacking, meta-classifier, random forest
- NSL-KDD, KDD Cup 99, Kitsune, IoT IDS
- class imbalance, minority class, U2R, R2L, threshold optimization

### Databases Searched
- [x] IEEE Xplore
- [x] ACM Digital Library
- [x] Google Scholar
- [x] ArXiv
- [ ] Other: —

### Time Period
2018–2025, prioritizing recent developments on Hybrid CNN-LSTM based anormally/threat detection.

## 3. Key Areas of Research

### 3.1 IDS Paradigms: Signature vs Anomaly
Signature-based IDS excel at known threats but struggle with novel patterns. Anomaly-based IDS model normal traffic and flag deviations, enabling zero-day detection but requiring careful thresholding to limit false alarms [1], [3]. Hybrid deployments often combine both, with standard definitions and roles documented in public guidance [10], [11].

**Key Papers:**
- Okoli et al., 2024 [1] – Survey of ML for threat detection and defenses.
- Mirsky et al., 2018 (Kitsune) [3] – Online ensemble of autoencoders for streaming network anomaly detection.

### 3.2 Deep Learning for Anomaly-Based IDS
Unsupervised/semi-supervised models (autoencoders, VAEs) learn normal behavior; supervised DL (CNN/LSTM/MLP/transformers) classifies labeled attacks; hybrids pair representation learning with classic ML. Handling class imbalance and optimizing thresholds are critical. Imbalance-aware objectives such as focal loss can elevate performance on rare events [6], [7]. Recent hybrid CNN–LSTM and attention-augmented architectures report improved intrusion detection across diverse settings, including IoT and industrial networks [12]–[16].

**Key Papers:**
- Khanam et al., 2022 [6] – Focal-loss VAE for IoT IDS to elevate rare-event detection.
- Mulyanto et al., 2020 [7] – Demonstrates focal loss efficacy for minority-class detection in NIDS.
- Bamber et al., 2025 [14] – Hybrid CNN–LSTM for intelligent cyber IDS.
- Alashjaee, 2025 [15] – Attention–CNN–LSTM for accurate intrusion detection.
- Li et al., 2018 [5] – Comparative analysis across ML algorithms for intrusion detection.

### 3.3 Ensembles and Meta-Learning
Ensembles (bagging/boosting/stacking) improve robustness and minority-class recall by aggregating diverse learners. Attack-specialized submodels fused via a meta-classifier capitalize on class-tailored decision boundaries while controlling variance through fusion, and can be combined with data-level balancing and meta-heuristic optimization to further enhance performance [2], [3]. Such ensemble strategies are also explored in IoT/IIoT contexts alongside hybrid DL models [12], [13], [14].

**Key Papers:**
- Shana et al., 2025 [2] – Anomaly IDS combining SMOTE-IPF, whale optimization, and ensembles.
- Mirsky et al., 2018 [3] – Unsupervised ensemble perspective for online IDS.

### 3.4 Datasets and Evaluation Protocols
KDD Cup 99/NSL-KDD are standard but dated; per-class metrics (precision, recall, F1), PR curves, confusion matrices, and threshold tuning offer more truthful assessment under imbalance than accuracy alone [1], [4]. Real-time oriented studies (e.g., Kitsune) emphasize streaming feasibility [3]. Emerging IoT/IIoT applications stress the need for dataset realism and evaluation beyond legacy benchmarks [12], [13], [14], [15].

**Key Papers:**
- Tavallaee et al., 2009 [4] – Critique of KDD Cup 99; rationale for NSL-KDD.
- Mirsky et al., 2018 [3] – Online anomaly detection with lightweight feature extraction.

## 4. Research Gaps and Opportunities

The following gaps emerge from recent literature:

### Gap 1: Reliance on aging datasets (external validity)
**Why it matters:** NSL-KDD may not reflect encrypted, cloud-native, and IoT-dense traffic, risking limited generalization.
**How our project addresses it:** Where feasible, include complementary datasets; otherwise, conduct rigorous per-class and ablation analyses, plus sensitivity to drift and thresholding; reference emerging IoT/IIoT-focused evaluations as design signals [12]–[15].

### Gap 2: Severe class imbalance (U2R/R2L)
**Why it matters:** Minority classes are high impact yet poorly recalled by monolithic models.
**How our project addresses it:** Attack-specialized submodels with PR-based threshold tuning; explore focal/cost-sensitive losses or curated resampling while monitoring precision–recall trade-offs [2], [6], [7].

### Gap 3: Real-time operation and concept drift
**Why it matters:** IDS must adapt to non-stationary traffic and evolving attacker behavior.
**How our project addresses it:** Evaluate latency/throughput; periodic calibration/retraining; drift detection triggers.

## 5. Theoretical Framework

We frame the approach as specialization plus ensembling. Specialized learners per attack type create class-tailored inductive biases; stacking via a meta-classifier aggregates heterogeneous signals to reduce variance and improve overall calibration. Under imbalance, decision theory motivates optimizing cost-sensitive objectives and selecting thresholds based on PR curves to maximize recall for minority classes subject to acceptable precision. This mirrors established ensemble learning theory applied to anomaly detection in high-dimensional tabular data.

## 6. Methodology Insights

- Modeling approaches
	- Unsupervised/semi-supervised anomaly detection (e.g., AEs/VAEs) to model “normal” traffic and flag deviations [3], [6].
	- Supervised deep classifiers (MLP/CNN/LSTM/transformers) for labeled multi-class attack detection [5], [12]–[16].
	- Hybrid DL + classic ML pipelines (representation learning followed by tree-based classifiers) for tabular features.

- Ensemble and stacking strategies
	- Combine complementary detectors (bagging/boosting/stacking) to improve robustness and minority-class recall [2], [3].
	- Use attack-specialized submodels (per DoS/Probe/R2L/U2R) and fuse via a meta-classifier to aggregate evidence.

- Imbalance-aware learning
	- Loss-level: focal or class-weighted/cost-sensitive losses to prioritize rare classes [6], [7].
	- Data-level: curated resampling or synthetic variants (apply cautiously to avoid overfitting and distribution shift).

- Thresholding, calibration, and metrics
	- Optimize decision thresholds using precision–recall (PR) curves; calibrate outputs for reliable probabilities.
	- Report per-class Precision/Recall/F1 and confusion matrices; avoid accuracy-only reporting under imbalance [1], [4].

- Operational readiness
	- Track latency and throughput; prefer lightweight feature sets and efficient inference paths for near real time.
	- Plan for drift-aware maintenance (periodic recalibration/retraining; drift detection triggers) [12]–[16].

- Recommended configuration for this project
	- Train attack-specialized deep submodels (per class family) and stack them with a meta-classifier.
	- Use imbalance-aware losses and PR-based threshold selection; add probability calibration.
	- Evaluate with per-class metrics and confusion matrices on NSL-KDD; document latency and conduct drift checks [1]–[7], [12]–[16].

## 7. Conclusion

Ensemble-fused, attack-specialized deep models advance anomaly-based IDS, particularly for rare classes when paired with PR-oriented thresholding. Recent hybrid CNN–LSTM and attention mechanisms provide complementary modeling capacity, while optimization and resampling techniques mitigate imbalance [2], [6], [7], [12]–[16]. To move toward practical applicability, future work should address dataset realism, latency and drift, and explainability—directions that inform the project’s methodology and evaluation plan.

## References

[1] U. I. Okoli, O. C. Obi, A. O. Adewusi, and T. O. Abrahams, “Machine learning in cybersecurity: A review of threat detection and defense mechanisms,” World Journal of Advanced Research and Reviews, vol. 21, no. 1, pp. 2286–2295, Jan. 2024, doi: 10.30574/wjarr.2024.21.1.0315.

[2] T. B. Shana, N. Kumari, M. Agarwal, S. Mondal, and U. Rathnayake, “Anomaly-based intrusion detection system based on SMOTE-IPF, whale optimization algorithm, and ensemble learning,” Intelligent Systems with Applications, vol. 27, p. 200543, Sep. 2025, doi: 10.1016/j.iswa.2025.200543.

[3] Y. Mirsky, T. Doitshman, Y. Elovici, and A. Shabtai, “Kitsune: An ensemble of autoencoders for online network intrusion detection,” in Proc. NDSS, 2018, doi: 10.14722/ndss.2018.23204.

[4] M. Tavallaee, E. Bagheri, W. Lu, and A. A. Ghorbani, “A detailed analysis of the KDD CUP 99 data set,” in 2009 IEEE Symposium on Computational Intelligence for Security and Defense Applications, pp. 1–6, 2009, doi: 10.1109/CISDA.2009.5356528.

[5] Z. Li, P. Batta, and L. Trajkovic, “Comparison of machine learning algorithms for detection of network intrusions,” in 2018 IEEE Int. Conf. on Systems, Man, and Cybernetics (SMC), pp. 4248–4253, 2018, doi: 10.1109/SMC.2018.00719.

[6] S. Khanam, I. Ahmedy, M. Y. I. Idris, and M. H. Jaward, “Towards an effective intrusion detection model using focal loss variational autoencoder for IoT,” Sensors, vol. 22, no. 15, p. 5822, 2022.

[7] M. Mulyanto, M. Faisal, S. W. Prakosa, and J.-S. Leu, “Effectiveness of focal loss for minority classification in network intrusion detection systems,” Symmetry, vol. 13, no. 1, p. 4, 2020, doi: 10.3390/sym13010004.

[8] SailPoint, “How AI and machine learning are improving cybersecurity,” 2025. [Online]. Available: https://www.sailpoint.com/identity-library/how-ai-and-machine-learning-are-improving-cybersecurity

[9] Comparitech, “Machine learning enhances threat detection by analyzing network traffic, identifying anomalies, and improving security with adaptive, real-time responses,” 2025. [Online]. Available: https://www.comparitech.com/net-admin/machine-learning-threat-detection/

[10] Wikipedia, “Intrusion detection system,” 2025. [Online]. Available: https://en.wikipedia.org/wiki/Intrusion_detection_system

[11] U.S. Department of Homeland Security (DHS), “Intrusion detection and prevention systems,” 2025. [Online]. Available: https://www.dhs.gov/publication/intrusion-detection-and-prevention-systems

[12] A. M. Alashjaee, “A hybrid CNN+LSTM-based intrusion detection system for industrial IoT networks,” 2023. [Online]. Available: https://www.researchgate.net/publication/366919487_A_hybrid_CNN_LSTM-based_intrusion_detection_system_for_industrial_IoT_networks

[13] M. Aljanabi, “Effective intrusion detection through hybrid CNN-LSTM and Grey Wolf Optimization,” Sensors, vol. 23, no. 18, p. 7856, 2023. [Online]. Available: https://www.mdpi.com/1424-8220/23/18/7856

[14] S. S. Bamber, A. V. R. Katkuri, S. Sharma, and M. Angurala, “A hybrid CNN–LSTM approach for intelligent cyber intrusion detection system,” Computers & Security, vol. 148, p. 104146, 2025, doi: 10.1016/j.cose.2024.104146.

[15] A. M. Alashjaee, “Deep learning for network security: An attention–CNN–LSTM model for accurate intrusion detection,” Scientific Reports, vol. 15, no. 1, p. 21856, 2025, doi: 10.1038/s41598-025-07706-y.

[16] K. T. V. Nguyen, A. V. T. Le, and K. M. T. Vo, “Attention mechanism in CNN–LSTM for IDS,” arXiv, 2025.
