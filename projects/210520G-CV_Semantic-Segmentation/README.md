# CV:Semantic Segmentation - 210520G

## Student Information

- **Index Number:** 210520G
- **Research Area:** CV:Semantic Segmentation
- **GitHub Username:** @chamath-ranasinghe
- **Email:** chamathranasinghe.21@cse.mrt.ac.lk

# Team-Aware Football Player Tracking with SAM

A lightweight football player tracking system that combines the Segment Anything Model (SAM) with classical CSRT trackers, enhanced by jersey color-based appearance models for improved occlusion recovery and player re-identification.

## Overview

This project addresses the challenge of tracking football players in crowded scenarios with frequent occlusions. By leveraging SAM's precise segmentation for initialization and incorporating appearance-based re-identification using jersey colors, the system maintains player identities even after temporary occlusions.

### Key Features

- 🎯 **Precise Initialization**: Uses SAM for high-quality player segmentation in the first frame
- 👕 **Jersey Color Recognition**: Extracts and tracks HSV color histograms for player re-identification
- 🏃 **Lightweight Tracking**: Employs CSRT trackers for efficient frame-to-frame tracking
- 🔄 **Occlusion Recovery**: Automatically detects and recovers lost players using appearance matching
- ⚽ **Team-Aware**: Supports tracking multiple players with team membership
- 📊 **Comprehensive Evaluation**: Three-dimensional metrics (speed, accuracy, robustness)

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- 16GB+ RAM

### Setup

1. Clone the repository:
```bash
git clone https://github.com/chamath-ranasinghe/SAMCSRT.git
cd SAMCSRT
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download SAM checkpoint:
```bash
mkdir checkpoints
# Download SAM ViT-H checkpoint (2.38GB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P checkpoints/
```

## Usage

### Basic Tracking

```python
from team_aware_tracker import TeamAwareTracker

# Initialize tracker with SAM checkpoint
tracker = TeamAwareTracker(
    sam_checkpoint="checkpoints/sam_vit_h_4b8939.pth",
    model_type="vit_h"
)

# Run tracking
tracker.track_video(
    video_path="football_match.mp4",
    output_path="output_tracked.mp4"
)
```

### Tracking with Evaluation

```python
# Run tracking with comprehensive evaluation metrics
tracker.track_video_with_evaluation_multiPlayer(
    video_path="football_match.mp4",
    output_path="output_tracked.mp4",
    results_dir="results"
)
```

### Interactive Controls

During tracking:
- **Left Click**: Select player (positive point on player)
- **Right Click**: Add negative point (background)
- **1, 2, 3 Keys**: Switch team selection (Team 0, Team 1, Referee)
- **SPACE**: Start tracking with selected players
- **R**: Reset selections
- **Q**: Quit

## Evaluation Metrics

The system provides three-dimensional evaluation:

### 1. Performance Metrics
- **Average FPS**: Processing speed
- **Frame Time**: Mean/min/max processing time per frame
- **Memory Usage**: Average and peak RAM consumption

### 2. Accuracy Metrics
- **Tracking Success Rate**: Percentage of frames with successful tracking
- **Bounding Box Stability**: Frame-to-frame consistency
- **Track Fragmentation**: Average interruptions per player

### 3. Robustness Metrics
- **Occlusion Recovery Rate**: Percentage of lost players recovered
- **Average Recovery Time**: Frames needed to recover after occlusion
- **Identity Switches**: Incorrect player ID swaps
- **Overall Robustness Score**: Weighted combination (0-100)

## Results

Results are automatically saved in the `results/` directory after running evaluation:

```bash
results/
├── performance_CSRT_TeamAware_Tracker.json    # Speed and memory metrics
├── accuracy_CSRT_TeamAware_Tracker.json       # Tracking quality metrics
├── robustness_CSRT_TeamAware_Tracker.json     # Occlusion handling metrics
└── robustness_CSRT_TeamAware_Tracker.txt      # Detailed text report
```

### Example Results

```
Performance: 22.5 FPS, 450 MB peak memory
Accuracy: 87.3% success rate, 0.94 bbox stability
Robustness: 64.2% recovery rate, 75.3/100 robustness score
```

## Methodology

### Pipeline Overview

1. **Interactive Selection**: User selects players in first frame with team assignment
2. **SAM Segmentation**: Generates precise masks for each selected player
3. **Appearance Extraction**: Computes HSV color histograms from jersey regions (top 60% of mask)
4. **CSRT Tracking**: Initializes lightweight trackers for frame-to-frame tracking
5. **Occlusion Recovery**: Monitors tracker confidence and uses appearance matching to recover lost players

### Appearance-Based Re-identification

The system extracts jersey color features using:
- HSV color space for better color discrimination
- 32-bin histograms for hue and saturation channels
- Sliding window (10 frames) for robust appearance modeling
- Bhattacharyya distance for similarity matching (threshold: 0.6)

## Configuration

Key parameters can be adjusted in `team_aware_tracker.py`:

```python
# Appearance model
HISTOGRAM_BINS = 32              # Color histogram bins
APPEARANCE_WINDOW = 10           # Frames to average for appearance model
JERSEY_REGION = 0.6              # Top 60% of mask (jersey area)

# Tracking
CONFIDENCE_THRESHOLD = 0.3       # Minimum tracker confidence
RECOVERY_INTERVAL = 10           # Frames before attempting recovery
SIMILARITY_THRESHOLD = 0.6       # Appearance matching threshold
```

## Requirements

```
torch>=2.1.0
torchvision>=0.16.0
opencv-contrib-python>=4.8.0
segment-anything>=1.0
numpy>=1.24.0
matplotlib>=3.7.0
psutil>=5.9.0
```

## Limitations

- Requires manual player selection in the first frame
- Performance depends on jersey color distinctiveness
- May struggle with extreme lighting changes
- Single camera view (no multi-view fusion)

## Future Work

- Automatic player detection (remove manual selection)
- Multi-camera tracking and fusion
- Real-time optimization for live broadcasts
- Integration with tactical analysis tools
- Deep learning-based re-identification

## Citation

If you use this code in your research, please cite:

```bibtex
@article{yourname2025football,
  title={Team-Aware Football Player Tracking with SAM: An Appearance-Based Approach to Occlusion Recovery},
  author={Your Name},
  journal={Conference/Journal Name},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) by Meta AI
- [OpenCV](https://opencv.org/) for computer vision tools
- CSRT tracker implementation from OpenCV contrib

## Contact

For questions or issues, please open an issue on GitHub or contact [chamathranasinghe.21@cse.mrt.ac.lk]

---

**Note**: This is a research project developed for academic purposes. For production deployments, additional optimization and testing are recommended.
