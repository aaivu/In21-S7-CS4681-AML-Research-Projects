# Methodology: CV:Semantic Segmentation

**Student:** 210520G
**Research Area:** CV:Semantic Segmentation
**Date:** 2025-01-11

## 1. Overview

This research investigates a team-aware football player tracking system that combines the Segment Anything Model (SAM) for precise segmentation with classical CSRT trackers enhanced by jersey color-based appearance models. The methodology focuses on evaluating the system's performance across three dimensions: processing speed, tracking accuracy, and robustness to occlusions in crowded football scenarios. The approach leverages SAM's zero-shot segmentation capabilities for initialization while maintaining computational efficiency through lightweight classical trackers and appearance-based re-identification.

## 2. Research Design

This is a **comparative evaluation study** using an empirical research design. The research follows these stages:

1. **System Development**: Implement a team-aware tracking system combining SAM initialization, CSRT tracking, and jersey color-based re-identification
2. **Experimental Evaluation**: Test the system on football video sequences with varying levels of occlusion
3. **Multi-dimensional Analysis**: Evaluate performance across speed, accuracy, and robustness metrics
4. **Comparative Assessment**: Compare against baseline (pure CSRT without SAM initialization or appearance models)
5. **Qualitative Analysis**: Analyze failure cases and system limitations

The research emphasizes **practical deployment considerations**, focusing on the trade-offs between computational efficiency and tracking performance for real-time sports analytics applications.

## 3. Data Collection

### 3.1 Data Sources

- **Primary Source**: Football match videos from YouTube (publicly available broadcast footage)
- **Alternative Sources**: 
  - SoccerNet dataset (if accessible for validation)
  - Custom recorded footage (if needed for specific scenarios)
- **Selection Criteria**: 
  - HD quality (minimum 720p, preferably 1080p)
  - Clear visibility of players
  - Diverse camera angles (wide shots, tactical camera)
  - Various crowding levels

### 3.2 Data Description

**Dataset Composition**:
- **Number of clips**: 8-12 video sequences
- **Duration per clip**: 10-30 seconds
- **Total footage**: ~3-5 minutes
- **Resolution**: 1920×1080 (1080p) or 1280×720 (720p)
- **Frame rate**: 25-30 FPS
- **Scenarios**:
  - Light occlusion: 1-2 players briefly overlapping (3-4 clips)
  - Heavy occlusion: Crowded penalty box, corner kicks, 5+ players (3-4 clips)
  - Long-term occlusion: Players temporarily off-screen (2-3 clips)
  - Clean tracking: Open field play for baseline comparison (2-3 clips)

**Player characteristics**:
- Players from both teams (different jersey colors)
- Mix of static and moving players
- Various player positions and roles

### 3.3 Data Preprocessing

**Video Preprocessing**:
1. **Format Standardization**: Convert all videos to MP4 format with H.264 codec
2. **Resolution Normalization**: Resize to consistent resolution (1280×720) if needed
3. **Frame Rate Adjustment**: Ensure consistent 30 FPS across all clips
4. **Clip Extraction**: Extract relevant segments showing target scenarios
5. **Quality Check**: Verify video quality, remove corrupted frames

**Annotation (Minimal)**:
- Manual selection of initial player positions in first frame
- Team membership assignment for each player
- Optional: Mark occlusion events for detailed robustness analysis
- No frame-by-frame annotation required (system-generated tracking serves as data)

**Storage Organization**:
```
data/
├── videos/
│   ├── light_occlusion/
│   ├── heavy_occlusion/
│   ├── long_term_occlusion/
│   └── clean_tracking/
└── annotations/
    └── player_selections.json
```

## 4. Model Architecture

Our system architecture consists of three main components:

### 4.1 SAM-based Initialization Module

**Component**: Segment Anything Model (SAM)
- **Model**: ViT-B (Vision Transformer Base)
- **Purpose**: Generate precise segmentation masks from user-specified points
- **Input**: First frame + point prompts (x, y coordinates)
- **Output**: Binary segmentation masks for each player
- **Processing**: Run once per video (first frame only)

### 4.2 Appearance Feature Extractor

**Purpose**: Extract and maintain jersey color signatures for re-identification

**Process**:
1. Isolate jersey region (top 60% of SAM mask)
2. Convert frame to HSV color space
3. Compute normalized histograms (32 bins for H and S channels)
4. Maintain sliding window of last 10 frames for robust averaging

**Representation**: 64-dimensional feature vector (32 + 32)

### 4.3 Tracking and Recovery Module

**Primary Tracker**: CSRT (Channel and Spatial Reliability Tracker)
- Discriminative correlation filter-based
- Handles partial occlusions
- Provides confidence scores

**Recovery Mechanism**:
1. **Occlusion Detection**: Monitor confidence drops, position jumps, size changes
2. **Position Prediction**: Extrapolate using velocity from recent history
3. **SAM Re-segmentation**: Apply SAM at predicted location
4. **Appearance Matching**: Compare jersey colors using Bhattacharyya distance
5. **Re-initialization**: Restart tracker if similarity > 0.6

### 4.4 Pipeline Flow
```
First Frame → SAM Segmentation → Appearance Extraction → CSRT Initialization
                                                              ↓
Frame N ← Recovery Module ← Occlusion Detection ← CSRT Tracking ← Frame 2..N
```

## 5. Experimental Setup

### 5.1 Evaluation Metrics

**Dimension 1: Performance (Speed & Efficiency)**
- Average FPS (Frames Per Second)
- Mean frame processing time (ms)
- Min/Max frame processing time (ms)
- Average memory usage (MB)
- Peak memory consumption (MB)

**Dimension 2: Accuracy (Tracking Quality)**
- Tracking Success Rate (TSR): % of frames with active tracking
- Bounding Box Stability: Frame-to-frame area consistency
- Track Fragmentation: Average interruptions per player
- IoU with ground truth: If manual annotations available (optional)

**Dimension 3: Robustness (Occlusion Handling)**
- Occlusion Recovery Rate (ORR): % of lost players recovered
- Average Recovery Time: Frames needed to recover
- Identity Switches: Number of ID swaps
- Track Persistence: Average continuous tracking duration
- Overall Robustness Score: Weighted combination (0-100)

### 5.2 Baseline Models

**Primary Baseline**: Pure CSRT tracker without SAM
- Manual bounding box initialization (no SAM segmentation)
- No appearance-based re-identification
- No recovery mechanism

**Comparison Points**:
- Speed: CSRT-only should be faster (no SAM processing)
- Accuracy: SAM initialization expected to improve initial tracking
- Robustness: Appearance-based recovery expected to significantly improve

**Ablation Studies**:
1. SAM + CSRT (no appearance model)
2. SAM + CSRT + appearance (no recovery)
3. SAM + CSRT + appearance + recovery (full system)

### 5.3 Hardware/Software Requirements

**Hardware**:
- GPU: NVIDIA RTX 3060 (12GB VRAM) or better
- CPU: Intel i7-12700K or equivalent
- RAM: 16GB minimum, 32GB recommended
- Storage: 50GB for models, videos, and results

**Software Environment**:
- Operating System: Windows 11 / Ubuntu 20.04+
- Python: 3.10+
- CUDA: 11.8
- Key Libraries:
  - PyTorch 2.1.0
  - torchvision 0.16.0
  - opencv-contrib-python 4.8.0
  - segment-anything 1.0
  - NumPy 1.24.0
  - Matplotlib 3.7.0
  - psutil 5.9.0

**Model Checkpoints**:
- SAM ViT-H: 2.38GB (`sam_vit_h_4b8939.pth`)

## 6. Implementation Plan

| Phase | Tasks | Duration | Deliverables |
|-------|-------|----------|--------------|
| **Week 1** | Environment setup, SAM integration, basic tracking implementation | 1 week | Working SAM + CSRT baseline |
| **Week 2** | Implement appearance model extraction and matching, develop recovery mechanism | 1 week | Complete tracking system with re-ID |
| **Week 3** | Develop evaluation framework (performance, accuracy, robustness metrics) | 1 week | Evaluation scripts |
| **Week 4** | Data collection and preparation (video clips, player selection) | 1 week | Test dataset ready |
| **Week 5-6** | Run experiments on all scenarios, collect metrics, ablation studies | 2 weeks | Raw experimental data |
| **Week 7** | Analyze results, generate visualizations, comparison tables | 1 week | Figures and tables |
| **Week 8-9** | Write methodology, results, discussion sections | 2 weeks | Complete draft paper |
| **Week 10** | Revisions, finalize paper, prepare presentation | 1 week | Final submission |

**Critical Milestones**:
- End of Week 2: Full system implementation complete
- End of Week 4: Dataset ready and validated
- End of Week 6: All experiments completed
- End of Week 9: Paper draft complete

## 7. Risk Analysis

### 7.1 Technical Risks

| Risk | Probability | Impact | Mitigation Strategy |
|------|------------|--------|---------------------|
| SAM model too slow for real-time | Medium | High | Use ViT-B (smaller model), optimize code, only run SAM for initialization and recovery |
| Appearance model fails with similar colors | High | Medium | Add spatial constraints, use texture features as backup, test on diverse jerseys |
| CSRT tracker drifts significantly | Medium | High | Implement stricter confidence thresholds, more frequent recovery attempts |
| Limited GPU memory | Low | High | Use smaller batch sizes, process videos sequentially, reduce memory footprint |
| Insufficient video data quality | Medium | Medium | Collect from multiple sources, ensure HD quality, validate before experiments |

### 7.2 Methodological Risks

| Risk | Probability | Impact | Mitigation Strategy |
|------|------------|--------|---------------------|
| Baseline comparison unfair | Low | Medium | Ensure identical test conditions, same videos, document all parameters |
| Evaluation metrics don't capture key behaviors | Medium | High | Use multiple metrics, qualitative analysis, validate with domain experts |
| Results not generalizable | Medium | Medium | Test on diverse scenarios, different teams/jerseys, multiple camera angles |
| Time constraints prevent full evaluation | Medium | High | Prioritize core experiments, reduce dataset size if needed, focus on key metrics |

### 7.3 Data Risks

| Risk | Probability | Impact | Mitigation Strategy |
|------|------------|--------|---------------------|
| Copyright issues with video footage | Low | High | Use publicly available content, cite sources, educational fair use |
| Insufficient occlusion scenarios | Medium | Medium | Specifically search for crowded scenes, use corner kicks and penalty boxes |
| Video quality inconsistencies | Medium | Low | Standardize preprocessing, filter poor quality clips |

## 8. Expected Outcomes

### 8.1 Technical Contributions

1. **Working Implementation**: A functional team-aware football player tracking system with:
   - SAM-based initialization for precise player segmentation
   - Appearance-based re-identification using jersey colors
   - Automatic occlusion recovery mechanism
   - Open-source code with comprehensive documentation

2. **Evaluation Framework**: Three-dimensional evaluation system measuring speed, accuracy, and robustness

3. **Empirical Insights**: Quantified trade-offs between different tracking components

### 8.2 Expected Findings

**Hypothesis 1**: SAM initialization improves tracking accuracy by 10-20% compared to manual bbox initialization due to more precise object boundaries.

**Hypothesis 2**: Jersey color-based appearance models improve occlusion recovery rate by 25-40% compared to position-only tracking.

**Hypothesis 3**: The lightweight approach (SAM + CSRT) processes 18-25 FPS on typical hardware, making it suitable for near-real-time applications.

**Hypothesis 4**: The system shows degraded performance when:
- Teams wear very similar jersey colors
- Extreme lighting changes occur
- Occlusions exceed 3-5 seconds

### 8.3 Practical Outcomes

1. **Deployment Guidelines**: Recommendations for when to use SAM-based tracking vs. simpler alternatives based on:
   - Computational budget
   - Required accuracy level
   - Occlusion frequency
   - Jersey distinctiveness

2. **Best Practices**: Documented optimal hyperparameters and configurations for football tracking

3. **Limitation Analysis**: Clear understanding of system boundaries and failure modes

### 8.4 Academic Contributions

1. **Research Paper**: Conference/workshop paper documenting methodology, experiments, and findings
2. **Code Repository**: GitHub repository with reproducible implementation
3. **Dataset**: Annotated test sequences (if permissible) for future research

### 8.5 Success Criteria

**Minimum Viable Success**:
- System tracks 3+ players simultaneously
- Achieves >80% tracking success rate on clean scenarios
- Recovers >50% of occlusions in crowded scenarios
- Processes at >15 FPS

**Target Success**:
- Tracks 6+ players simultaneously
- Achieves >85% tracking success rate overall
- Recovers >65% of occlusions
- Processes at >20 FPS
- Identity switches <5 per 100 frames

---

**Note:** This methodology document will be updated based on preliminary results and challenges encountered during implementation. All changes will be documented with version control.

**Version**: 1.0  
**Last Updated**: 2025-01-11