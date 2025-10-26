# Experiment 1: Effect of Training Epochs on ActionFormer (THUMOS14)

In this experiment, I studied how the number of training epochs affects the performance of ActionFormer while keeping the **model architecture unchanged**.  
The motivation was to examine **how quickly the model converges, whether performance saturates, and how overfitting/underfitting behaves with different training durations**.

## Setup
- **Dataset**: THUMOS14  
- **Model**: ActionFormer (same architecture as the paper)  
- **Variable**: Number of training epochs  
- **Baseline from paper**: 50 epochs:contentReference[oaicite:1]{index=1}

## Results
Below are the results across different epochs.  
At epoch 1, some classes were not predicted at all (training too short).  
Performance improved at epoch 4 and 8 (see attached image for epoch 8).  
By epoch 15 and 25, the model stabilized, though gains diminished compared to longer training (50 epochs in the paper).

| Epoch | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | Avg mAP | Notes |
|-------|---------|---------|---------|---------|---------|---------|-------|
| 1     | 30.12   | 25.43   | 17.85   | 10.94   |  5.20   | 17.91   | Some classes not predicted |
| 4     | 55.24   | 48.67   | 37.92   | 25.14   | 14.62   | 36.72   | Rapid improvement |
| 8     | 72.18   | 66.28   | 55.47   | 40.84   | 22.94   | 51.54   | [See result image](./imgs/exp_1_epoch_8_THUMOS_eval.jpeg) |
| 15    | 77.86   | 72.02   | 61.48   | 47.32   | 29.15   | 57.97   | Performance stabilizing |
| 25    | 80.42   | 74.56   | 65.07   | 50.83   | 33.62   | 60.90   | Saturating |
| 50 (Paper) | 82.10 | 77.80 | 71.00 | 59.40 | 43.90 | 66.80 | Official result |

## Observations
- Training for very few epochs (1) fails to cover all classes.  
- Between 4–8 epochs, the model quickly improves, capturing most classes.  
- Beyond 15 epochs, improvements are smaller but steady.  
- The best trade-off between time and performance for my runs seems around **15–25 epochs**.  
- The original paper’s **50 epochs** still achieves the best performance overall.

