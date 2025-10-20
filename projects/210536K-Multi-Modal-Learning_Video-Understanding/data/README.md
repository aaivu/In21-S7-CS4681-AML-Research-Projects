# Datasets

## Overview

This research primarily focuses on **temporal action localization** in untrimmed videos, a challenging task in video understanding that requires models to not only recognize actions but also precisely identify when they occur within long video sequences.

## Primary Dataset: THUMOS14

Our main target dataset for this research is **[THUMOS14](./THUMOS14.md)**, which serves as the primary benchmark for evaluating our proposed model's performance.

### Why THUMOS14?

THUMOS14 was chosen as our primary dataset for several key reasons:

- **Temporal Action Localization Focus:** THUMOS14 is specifically designed for temporal action localization tasks, which aligns perfectly with our research objectives.
- **Untrimmed Videos:** The dataset consists of temporally untrimmed videos where actions of interest occupy only small portions of the entire video duration, making it ideal for testing temporal localization capabilities.
- **Standard Benchmark:** THUMOS14 is a widely-adopted benchmark in the temporal action localization community, allowing for direct comparison with state-of-the-art methods.
- **Challenging Scenarios:** With background clutter and varying action durations, THUMOS14 provides a realistic and challenging testbed for developing robust video understanding models.
- **Moderate Scale:** The dataset's size (20 action classes with over 1,000 validation videos and 1,500 test videos) provides sufficient data for training while remaining computationally manageable.

## Additional Datasets for Robustness Validation

To demonstrate the generalizability and robustness of our proposed model across diverse video understanding scenarios, we additionally evaluate on:

### ActivityNet

**[ActivityNet](./ActivityNet.md)** is used to validate our model's performance on:

- **Large-scale scenarios:** With over 200 activity classes, ActivityNet tests our model's scalability and ability to handle a diverse range of human activities.
- **Hierarchical activity understanding:** The dataset's semantic hierarchy allows us to evaluate how well our model captures relationships between different action categories.
- **Complex activities:** ActivityNet includes more complex, longer-duration activities compared to THUMOS14, testing our model's robustness to temporal variations.

### EPIC Kitchens 100

**[EPIC Kitchens 100](./EPIC_Kitchens_100.md)** is employed to assess our model's robustness on:

- **Egocentric viewpoint:** The first-person perspective provides a distinctly different visual domain, testing our model's ability to generalize across viewpoints.
- **Fine-grained actions:** With detailed action classes like "take knife from block," this dataset evaluates our model's precision in distinguishing subtle action variations.
- **Real-world, unscripted scenarios:** The naturalistic kitchen activities test our model's performance on authentic, unconstrained human behaviors.
- **Dense temporal annotations:** The dataset's rich annotations allow for comprehensive evaluation of temporal localization accuracy.

## Summary

By primarily developing and optimizing our approach on **THUMOS14**, then validating on **ActivityNet** and **EPIC Kitchens 100**, we demonstrate that our model:

1. Achieves strong performance on the standard temporal action localization benchmark (THUMOS14)
2. Scales effectively to large-scale datasets with diverse action categories (ActivityNet)
3. Generalizes across different viewpoints and fine-grained action classes (EPIC Kitchens 100)

This multi-dataset evaluation strategy provides comprehensive evidence of our model's robustness and effectiveness across various video understanding challenges.

## Dataset Details

For detailed information about each dataset, including download links, features, and specifications, please refer to:

- **[THUMOS14.md](./THUMOS14.md)** - Primary benchmark dataset
- **[ActivityNet.md](./ActivityNet.md)** - Large-scale validation dataset
- **[EPIC_Kitchens_100.md](./EPIC_Kitchens_100.md)** - Egocentric robustness validation dataset
