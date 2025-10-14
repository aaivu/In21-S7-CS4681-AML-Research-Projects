# Literature Review: CV:Semantic Segmentation

**Student:** 210520G
**Research Area:** CV:Semantic Segmentation
**Date:** 2025-09-01

## Abstract

This literature review covered recent research in multi-object tracking and segmentation, ranging from traditional detection-based methods to modern foundation-model approaches. Early works like ByteTrack and MOTChallenge provided strong baselines for tracking with detection boxes, while newer studies such as SAM2MOT, Masks and Boxes, and Cutie emphasized the use of segmentation for more accurate and robust tracking. With the introduction of the Segment Anything Model (SAM) and its extension SAM 2, methods like SAM-Track, Track Anything (TAM), and SAMURAI have expanded tracking to interactive and zero-shot settings. Hybrid approaches such as combining CSRT with Faster R-CNN also highlight the benefits of merging classical trackers with deep learning. Overall, these works reveal a shift towards segmentation-driven and foundation-model-based tracking frameworks that improve accuracy, adaptability, and usability across applications.

## 1. Introduction

This literature review examines recent developments in the field of object tracking, with a particular focus on approaches that combine segmentation and tracking. While traditional methods relied heavily on detection-based tracking using bounding boxes, recent advances have explored the use of segmentation masks and foundation models such as the Segment Anything Model (SAM) to achieve more accurate and flexible results. The review covers both multi-object and single-object tracking frameworks, highlighting the growing trend of integrating segmentation with tracking algorithms. This scope aligns with my research direction, which focuses on single object tracking using SAM-generated masks in combination with a tracking algorithm.

## 2. Search Methodology

### Search Terms Used

- Single Object Tracking
- Mask oriented tracking
- SAM
- Object Tracking

### Databases Searched

- IEEE Xplore
- ACM Digital Library
- Google Scholar
- ArXiv
- ResearchGate

### Time Period

2020 - 2025

## 3. Key Areas of Research

### 3.1 [Topic Area 1]

**Key Papers:**

- ByteTrack: Multi-Object Tracking by Associating Every Detection Box
  Yifu Zhang, Peize Sun, Yi Jiang, Xinggang Wang 2022 (published Oct 2022 in LNCS)
  Proposes a simple and effective tracking-by-detection association method that uses almost every detection box (including low-confidence ones) rather than only high-scoring detections, to reduce missed tracks and fragmentation.
- MOTChallenge: A Benchmark for Single-Camera Multiple Target Tracking
  Patrick Dendorfer, Aljoša Ošep, Anton Milan, Konrad Schindler, Daniel Cremers, Ian Reid, Stefan Roth, Laura Leal-Taixé 2020 (published online Dec 2020)
  Introduces the MOTChallenge benchmark (collecting MOT15, MOT16, MOT17) for standardized evaluation of multi-object (mostly pedestrian) tracking on single-camera setups. Provides datasets, evaluation metrics, and categorization of trackers.
- SAM2MOT: A Novel Paradigm of Multi-Object Tracking by Segmentation
  Junjie Jiang, Zelin Wang, Manqi Zhao, Yin Li, Dongsheng Jiang 2025 (preprint, Apr 2025)
  Proposes a “tracking by segmentation” paradigm: rather than relying on detection boxes, SAM2MOT directly generates tracking boxes from segmentation masks (via SAM2) and manages object association, addition/removal, and cross-object interactions. It claims strong zero-shot generalization and improved performance on benchmarks like DanceTrack, UAVDT, BDD100K.
- Masks and Boxes: Combining the Best of Both Worlds for Multi-Object Tracking
  Tomasz Stanczyk, François Bremond 2024 (preprint, Sep 2024)
  Proposes McByte, an MOT method that leverages both segmentation masks (propagated temporally) and bounding boxes as cues in the association process. The idea is to use masks as strong guidance to complement box-based association, improving robustness without per-sequence tuning.
- SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory
  Cheng-Yen Yang, Hsiang-Wei Huang, Wenhao Chai, Zhongyu Jiang, Jenq-Neng Hwang 2024 (preprint)
  Enhances SAM 2 for visual object tracking by incorporating motion-aware memory selection: using temporal motion cues to decide which memory frames to keep for conditioning, refining mask selection, and improving robustness in crowded or occluded scenes—without retraining or fine-tuning.
- Putting the Object Back into Video Object Segmentation
  Ho Kei Cheng, Seoung Wug Oh, Brian Price, Joon-Young Lee, Alexander Schwing 2024
  Introduces Cutie, a Video Object Segmentation (VOS) network that reintroduces object-level memory reading into the VOS pipeline. Unlike prior approaches that rely mainly on pixel-level memory matching (which is prone to noise and confusion in cluttered scenes), Cutie uses top-down object queries that summarize the target object. These queries interact with bottom-up pixel features via a query-based object transformer, enabling more robust segmentation. Cutie also employs foreground-background masked attention to clearly separate object and background semantics.
- SAM 2: Segment Anything in Images and Videos
  Nikhila Ravi*,†, Valentin Gabeur*, Yuan-Ting Hu*, Ronghang Hu*, Chaitanya Ryali*, Tengyu Ma*, Haitham Khedr*, Roman Rädle*, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár†, Christoph Feichtenhofer 2024
  SAM 2 extends the original Segment Anything Model to both images and videos, introducing a data engine for large-scale video segmentation and a transformer with streaming memory for real-time performance. It achieves higher accuracy with fewer interactions in video segmentation and is faster and more accurate than SAM for images, establishing a foundation model for promptable segmentation tasks.
- Object Tracking using CSRT Tracker and RCNN
  Khurshedjon Farkhodov, Suk-Hwan Lee, Ki-Ryong Kwon 2021
  This work integrates the CSRT tracker with a Faster R-CNN object detector to improve robustness in object tracking under challenges such as occlusion, appearance change, and camera motion. The CSRT tracker, a C++ implementation of CSR-DCF in OpenCV, benefits from the strong feature extraction of the CNN-based detector, leading to more accurate identification of object features, classes, and locations. Experiments show that the hybrid approach outperforms standalone tracking algorithms or filters.
- Segment and Track Anything
  Yangming Cheng, Liulei Li, Yuanyou Xu, Xiaodi Li, Zongxin Yang, Wenguan Wang, Yi Yang 2021
  This paper introduces SAM-Track, a framework that combines the Segment Anything Model (SAM) with an AOT-based tracking model (DeAOT) and Grounding-DINO to enable multimodal and interactive video object segmentation. Users can segment and track objects in videos through clicks, strokes, or text prompts, supporting flexible and precise multi-object tracking. SAM-Track achieves strong benchmark results, including 92.0% on DAVIS-2016 Val and 79.2% on DAVIS-2017 Test, and demonstrates applicability across domains such as autonomous driving, drone imaging, medicine, augmented reality, and biology.
  -Track Anything: Segment Anything Meets Videos
  Jinyu Yang, Mingqi Gao, Zhe Li, Shang Gao, Fangjing Wang, Feng Zheng 2023
  This work presents the Track Anything Model (TAM), which extends the Segment Anything Model (SAM) to videos by enabling interactive tracking and segmentation with minimal human input. With just a few clicks, users can track arbitrary objects of interest throughout a video in a single inference pass, without requiring extra training. TAM delivers strong video object tracking and segmentation performance, making it a practical and general-purpose framework for real-world applications.

### 3.2 [Topic Area 2]

[Continue with other relevant areas]

## 4. Research Gaps and Opportunities

[Identify gaps in current research that your project could address]

### Gap 1: [Description]

**Why it matters:** [Explanation]
**How your project addresses it:** [Your approach]

### Gap 2: [Description]

**Why it matters:** [Explanation]
**How your project addresses it:** [Your approach]

## 5. Theoretical Framework

[Describe the theoretical foundation for your research]

## 6. Methodology Insights

[What methodologies are commonly used? Which seem most promising for your work?]

## 7. Conclusion

[Summarize key findings and how they inform your research direction]

## References

[Use academic citation format - APA, IEEE, etc.]

1. [Reference 1]
2. [Reference 2]
3. [Reference 3]
   ...

---

**Notes:**

- Aim for 15-20 high-quality references minimum
- Focus on recent work (last 5 years) unless citing seminal papers
- Include a mix of conference papers, journal articles, and technical reports
- Keep updating this document as you discover new relevant work
