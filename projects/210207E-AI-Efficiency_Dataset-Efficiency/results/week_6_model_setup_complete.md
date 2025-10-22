# Progress Report: Week 6 - Baseline Model and Methodology Setup

**Student Index Number:** 210207E

**Milestone:** Baseline Model Setup and Variant Implementation

## 1. Objectives for the Week

1.  Finalize the 1.6M parameter 'nanoGPT' architecture implementation.
2.  Implement the three experimental variants: Baseline (None), LayerNorm (LN), and RMSNorm (RMS).
3.  Establish the data pipeline for the character-level Shakespeare dataset.
4.  Verify all key experimental controls (hyperparameters, optimizer, weight tying, Pre-LN blocks).

## 2. Progress Achieved

*   **Architecture Implementation:** Completed the custom, decoder-only Transformer model. The architecture is a faithful replica of the GPT-style Pre-LN configuration (8 layers, 128 dim, 8 heads) and confirms a final parameter count of approximately 1.6 million.
*   **Variant Implementation:** The normalization modules (`LayerNorm` and custom `RMSNorm`) have been successfully integrated immediately following the summed token and positional embeddings (the critical point of the study).
*   **Diagnostic Implementation:** The `Mean Embedding Norm` metric is implemented and confirmed to be operational to verify the layer function.
*   **Data Pipeline:** The character-level Shakespeare dataset (1.1M characters) has been preprocessed, and the `train.bin` and `val.bin` files are generated. The data loading utilities are stable.
*   **Controls Verified:** Hyperparameters (e.g., L=128, B=16, LR=3e-4) and architectural constraints (weight tying, fixed initialization) are locked in across all variants.

## 3. Preliminary Verification

A single-seed, 50-iteration test run for all three variants was conducted to confirm the stability of the training loop and the function of the new layers.

*   The Baseline model is stable and converges.
*   The LN and RMS variants successfully report a significantly higher mean embedding norm compared to the Baseline, confirming that the normalization layers are actively rescaling the vectors as intended.

## 4. Plan for Next Week (Mid-Evaluation)

*   Execute the Phase 1 preliminary experiment: **150 iterations** across **3 distinct random seeds** for all three variants.
*   Collect the four core metrics (PPL, Mock Accuracy, Final Grad Norm, Embedding Norm) for the short paper submission (Mid-Evaluation).
*   Begin drafting the short paper (Mid-Evaluation Submission).

## 5. Potential Issues

*   None. The current 1.6M parameter size fits within the allocated GPU resources, and the codebase is stable.

---
