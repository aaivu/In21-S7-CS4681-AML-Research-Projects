# Research Proposal: Team-Aware Football Player Tracking with SAM: An Appearance-Based Approach to Occlusion Recovery

**Student:** 210520G
**Research Area:** CV:Semantic Segmentation
**Date:** 2025-09-01

## Abstract

This research investigates team-aware football player tracking using the Segment Anything Model (SAM) combined with appearance-based re-identification techniques. Football player tracking presents unique challenges including frequent occlusions, similar player appearances, and rapid movements. We propose a hybrid approach that leverages SAM's precise segmentation capabilities for initial player identification and combines it with classical tracking methods enhanced by jersey color-based appearance models. Our system is compared against state-of-the-art methods including SAM 2 and SAM-Track across three key metrics: processing speed (FPS), tracking accuracy, and robustness to occlusions. The research aims to provide practical insights for resource-constrained real-time sports analytics applications by evaluating the trade-offs between computational efficiency and tracking performance in crowded football scenarios.

## 1. Introduction

Video object tracking is a fundamental task in computer vision with significant applications in sports analytics, particularly in football where player tracking enables tactical analysis, performance evaluation, and automated highlight generation. The recent introduction of Meta's Segment Anything Model (SAM) has revolutionized image segmentation with its zero-shot capabilities and precise boundary detection. However, extending SAM to video tracking, especially in challenging sports scenarios, remains an active research area.
Football presents unique tracking challenges: (1) players frequently occlude each other in crowded penalty boxes, (2) team uniforms create visual similarity among players, (3) rapid directional changes and sprinting introduce motion blur, and (4) camera movements complicate spatial consistency. Traditional tracking methods struggle with these conditions, while deep learning approaches often require extensive computational resources unsuitable for real-time applications.
This research explores a novel team-aware tracking approach that combines SAM's segmentation precision with lightweight classical trackers and appearance-based re-identification using jersey color analysis. By focusing specifically on football player tracking, we address domain-specific challenges while providing a comparative analysis of different SAM-based tracking paradigms.

## 2. Problem Statement

Existing football player tracking systems face three critical challenges:

Occlusion Recovery: When players overlap or exit the frame, maintaining identity consistency upon re-appearance is difficult. Current methods either lose track entirely or suffer from identity switches.
Computational Efficiency vs. Accuracy Trade-off: State-of-the-art tracking methods (e.g., SAM 2, transformer-based trackers) achieve high accuracy but require substantial computational resources, limiting real-time deployment. Conversely, lightweight trackers sacrifice accuracy in complex scenarios.
Team-Level Discrimination: Most generic trackers do not leverage domain knowledge such as team membership, which could improve re-identification after occlusions by incorporating jersey color information.

Research Question: Can a hybrid approach combining SAM's precise segmentation with classical trackers and appearance-based re-identification provide a practical balance between tracking accuracy and computational efficiency for football player tracking, particularly in handling occlusions?

## 3. Literature Review Summary

Segment Anything Model (SAM)

SAM (Kirillov et al., 2023) introduced a foundation model for image segmentation capable of zero-shot segmentation with various prompts (points, boxes, masks). While powerful for static images, SAM lacks temporal consistency needed for video tracking.

Video Object Segmentation

SAM 2 (Ravi et al., 2024) extended SAM to video with built-in temporal propagation using a memory-based architecture. SAM-Track (Cheng et al., 2023) combined SAM with AOT (Associating Objects with Transformers) for efficient mask propagation. Both approaches demonstrate strong performance but with significant computational overhead.

Sports Tracking

Traditional sports tracking relies on detection-based methods (YOLO + DeepSORT, ByteTrack) which struggle with occlusions. Recent work has explored domain-specific features like jersey numbers and team colors, but integration with foundation models remains limited.

Research Gap

While SAM-based tracking methods show promise, few studies have: (1) specifically addressed football's unique occlusion patterns, (2) leveraged domain knowledge (team colors) for appearance-based re-identification, (3) systematically compared lightweight vs. heavy tracking approaches for resource-constrained scenarios, or (4) provided comprehensive evaluation across multiple dimensions (speed, accuracy, robustness).

## 4. Research Objectives

### Primary Objective

To develop and evaluate a team-aware football player tracking system that combines SAM segmentation with appearance-based re-identification, providing improved occlusion recovery while maintaining computational efficiency suitable for real-time applications.

### Secondary Objectives

- Compare the performance of SAM + classical trackers (CSRT/KCF) against SAM 2 and SAM-Track in football-specific scenarios
- Evaluate the effectiveness of jersey color-based appearance models for player re-identification after occlusions
- Quantify the trade-offs between tracking accuracy, processing speed, and memory usage across different approaches
- Provide practical recommendations for football tracking system deployment based on resource constraints and accuracy requirements
- Analyze robustness to occlusions in crowded scenarios (penalty boxes, corner kicks)

## 5. Methodology

5.1 Proposed Approach

System Architecture:

Initial Segmentation: Use SAM with point prompts to segment selected players in the first frame
Appearance Model Extraction: Extract HSV color histograms from segmented jersey regions (upper 60% of mask)
Tracking: Initialize classical trackers (CSRT) for each player using SAM-derived bounding boxes
Appearance Monitoring: Maintain running appearance history for each player
Occlusion Detection: Monitor tracker confidence and bbox consistency
Recovery Mechanism: When players are lost, use velocity prediction and SAM re-segmentation at predicted locations, validating identity through appearance similarity

Baseline Implementations:

SAM 2: Native video segmentation with temporal memory
SAM-Track: SAM + AOT mask propagation

5.2 Evaluation Framework
Three-Dimensional Evaluation:

Performance Metrics

Average FPS (frames per second)
Frame processing time (mean, min, max)
Memory usage (average and peak)
Real-time capability assessment

Accuracy Metrics

Tracking success rate (% frames with active tracks)
Bounding box stability (frame-to-frame consistency)
Track fragmentation (average fragments per player)
IoU with ground truth (if available)

Robustness Metrics

Occlusion recovery rate (% successful recoveries)
Average recovery time (frames)
Identity switch rate
Track persistence (average track duration)
Overall robustness score (0-100)

5.3 Experimental Design
Test Scenarios:

Light occlusion: 1-2 players briefly overlap
Heavy occlusion: Crowded penalty box (5+ players)
Long-term occlusion: Player temporarily off-screen
Similar appearance: Same-team players in proximity

Datasets:

Custom football footage: 5-10 video clips (10-30 seconds each)
SoccerNet dataset (if accessible)
Manual annotation for ground truth validation

5.4 Implementation Tools

Python 3.10+
PyTorch 2.1+
OpenCV for classical trackers
SAM (segment-anything)
SAM 2 (segment-anything-2)
SAM-Track (Segment-and-Track-Anything)

## 6. Expected Outcomes

Technical Contributions

A working team-aware football player tracking system with appearance-based re-identification
Comprehensive comparative analysis of SAM-based tracking approaches in football contexts
Quantified trade-offs between computational efficiency and tracking accuracy
Open-source implementation and evaluation framework for reproducibility

Expected Findings

Hypothesis 1: Classical trackers with SAM initialization and appearance models will achieve 60-80% of the accuracy of SAM 2/SAM-Track while running 2-3x faster
Hypothesis 2: Jersey color-based appearance models will improve occlusion recovery rates by 20-40% compared to position-only tracking
Hypothesis 3: Different tracking approaches will excel in different scenarios (e.g., lightweight for real-time, heavy for accuracy)

Practical Impact

Guidelines for selecting appropriate tracking methods based on deployment constraints
Insights into when appearance-based re-identification provides significant benefits
Foundation for future sports analytics applications

## 7. Timeline

| Week | Task |
|------|------|
| 1-2  | Literature Review and Environment Setup: Review SAM, SAM 2, SAM-Track papers; survey football tracking methods; setup development environment with required libraries and models |
| 3-4  | Methodology Development: Design team-aware tracking architecture; develop appearance-based re-identification approach; design evaluation framework with three metrics |
| 5-8  | Implementation: Implement SAM + CSRT tracker with jersey color analysis (Week 5-6); Implement SAM 2 tracking system (Week 6-7); Implement SAM-Track system (Week 7-8); Develop evaluation scripts for performance, accuracy, and robustness |
| 9-12 | Experimentation: Collect and prepare football video clips and annotations (Week 9); Run all three tracking systems on test scenarios (Week 10-11); Collect performance, accuracy, and robustness metrics (Week 11-12); Generate comparison tables and visualizations |
| 13-15| Analysis and Writing: Analyze results and identify trade-offs between methods (Week 13); Write methodology, results, and discussion sections (Week 14); Complete full draft with introduction, conclusion, and abstract (Week 15) |
| 16   | Final Submission: Revisions and proofreading; prepare presentation slides; final submission |

## 8. Resources Required

Software & Libraries

Python 3.10+ with PyTorch 2.1+
OpenCV 4.5+ (opencv-contrib-python for trackers)
segment-anything (SAM)
segment-anything-2 (SAM 2)
Segment-and-Track-Anything repository
NumPy, Matplotlib, Pandas for analysis
Jupyter Notebook for experimentation

Hardware

GPU with 8GB+ VRAM (NVIDIA RTX 3060 or better) for SAM/SAM 2
16GB+ RAM
50GB+ storage for models and videos

Datasets

Custom football video clips (5-10 clips, 10-30 seconds each)
Optional: SoccerNet dataset for additional validation
Manual annotations for ground truth (minimal - focus on qualitative analysis)

Pre-trained Models

SAM checkpoints: vit_b (358MB) or vit_h (2.4GB)
SAM 2 checkpoints: sam2_hiera_large.pt (2.4GB)
DeAOT checkpoints for SAM-Track

Development Tools

Git for version control
Google Colab/Kaggle (backup if local GPU insufficient)
LaTeX for paper writing

## References

Cheng, Y., Li, L., Xu, Y., Li, X., Yang, Z., Wang, W., & Yang, Y. (2023). Segment and Track Anything. arXiv preprint arXiv:2305.06558.

Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., ... & others. (2023). Segment Anything. arXiv preprint arXiv:2304.02643.

Ravi, N., Gabeur, V., Hu, Y. T., Hu, R., Ryali, C., Ma, T., ... & others. (2024). SAM 2: Segment Anything in Images and Videos. arXiv preprint arXiv:2408.00714.

Yang, Z., Wei, Y., & Yang, Y. (2021). Associating Objects with Transformers for Video Object Segmentation. In Advances in Neural Information Processing Systems (NeurIPS).

Yang, Z., & Yang, Y. (2022). Decoupling Features in Hierarchical Propagation for Video Object Segmentation. In Advances in Neural Information Processing Systems (NeurIPS).

Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., ... & Wang, X. (2022). ByteTrack: Multi-Object Tracking by Associating Every Detection Box. In European Conference on Computer Vision (ECCV).

---
