# Experiment 6 — CLAHE-enhanced DenseNet-121 with Attention (CBAM)

Summary
-------
DenseNet-121 with CBAM (channel + spatial attention) applied to CLAHE-preprocessed images. 

Key config
----------
- Seed: `42` | Batch size: `8` | LR: `5e-5` | Epochs: `25` | Image size: `224`
- Optimizer: `AdamW` (wd=1e-5) | Loss: `FocalLoss(alpha=1,gamma=2)`

What I did
----------
- Implemented a CBAM (Convolutional Block Attention Module) on top of the CLAHE-enhanced DenseNet-121 replication. The CBAM applies channel attention followed by spatial attention to the DenseNet convolutional feature maps before pooling and classification.

Why I used attention
--------------------
- Attention (CBAM) can help the model focus on the most informative channels and spatial locations. For chest x-rays this can: (a) emphasize lesion-related feature channels, and (b) spatially weight regions likely to contain pathology (e.g., localized masses, pneumothorax). The goal was a lightweight inductive bias to improve AUROC on selected classes without changing the encoder backbone.

How attention is integrated (complete overview)
----------------------------------------------
1. Backbone: DenseNet-121 pretrained on ImageNet; features are taken from `densenet.features` and classifier head removed.
2. CBAM module: a ChannelAttention block followed by a SpatialAttention block. ChannelAttention computes global avg and max pooled descriptors, passes them through a small MLP (1x1 convs) and combines via sigmoid gating. SpatialAttention concatenates channel-wise avg/max maps and applies a conv to produce spatial attention weights.
3. Integration point: CBAM is applied to the final DenseNet convolutional feature map tensor (the output of `densenet.features`) and multiplies the feature map in-place before adaptive pooling.
4. Head: After CBAM, an `AdaptiveAvgPool2d(1)` reduces spatial dims, the vector is flattened, passed through `Dropout(0.2)` and a final `Linear(num_features, 14)` classifier.

Reproducibility
---------------
- Same random seed `42` and `25` epochs were used for this run (matching other experiments in this series).

===== TEST RESULTS =====
Loss      : 0.0419
Avg AUROC : 0.8480
Avg F1    : 0.3787

=== Per-Class Metrics ===
Class                AUROC      F1         Threshold 
-------------------------------------------------------
Atelectasis          0.8138     0.3972     0.3822    
Cardiomegaly         0.9364     0.4912     0.3991    
Consolidation        0.7774     0.2374     0.3048    
Edema                0.8950     0.2706     0.3552    
Effusion             0.8991     0.6050     0.4112    
Emphysema            0.9606     0.5325     0.3138    
Fibrosis             0.8379     0.1491     0.2836    
Hernia               0.9973     0.7500     0.4902    
Infiltration         0.7058     0.4007     0.3614    
Mass                 0.9077     0.4833     0.4356    
Nodule               0.7648     0.3276     0.4054    
Pleural_Thickening   0.7932     0.2411     0.3204    
Pneumonia            0.7054     0.0621     0.2181    
Pneumothorax         0.8781     0.3539     0.4121   

Overall comparison

| Metric    | CLAHE DenseNet-121 (baseline) | DenseNet-121 + CBAM |
|----------:|-------------------------------:|--------------------:|
| Loss      | 0.0415                         | 0.0419              |
| Avg AUROC | 0.8514                         | 0.8480              |
| Avg F1    | 0.3803                         | 0.3787              |

Per-class AUROC comparison (baseline vs CBAM) with delta (CBAM - baseline)

| Class                | Baseline | CBAM   | Delta   | Winner |
|----------------------|:--------:|:------:|:-------:|:------:|
| Atelectasis          | 0.8146   | 0.8138 | -0.0008 | Baseline |
| Cardiomegaly         | 0.9325   | 0.9364 |  0.0039 | CBAM |
| Consolidation        | 0.7871   | 0.7774 | -0.0097 | Baseline |
| Edema                | 0.8841   | 0.8950 |  0.0109 | CBAM |
| Effusion             | 0.9015   | 0.8991 | -0.0024 | Baseline |
| Emphysema            | 0.9656   | 0.9606 | -0.0050 | Baseline |
| Fibrosis             | 0.8207   | 0.8379 |  0.0172 | CBAM |
| Hernia               | 0.9936   | 0.9973 |  0.0037 | CBAM |
| Infiltration         | 0.7044   | 0.7058 |  0.0014 | CBAM |
| Mass                 | 0.9122   | 0.9077 | -0.0045 | Baseline |
| Nodule               | 0.7780   | 0.7648 | -0.0132 | Baseline |
| Pleural_Thickening   | 0.8124   | 0.7932 | -0.0192 | Baseline |
| Pneumonia            | 0.7229   | 0.7054 | -0.0175 | Baseline |
| Pneumothorax         | 0.8902   | 0.8781 | -0.0121 | Baseline |

Observations & notes
---------------------
- Summary of result: the CBAM run produced a slightly lower Avg AUROC (0.8480 vs baseline 0.8514) and similar Avg F1; several classes improved modestly (e.g., Fibrosis, Edema, Cardiomegaly, Hernia) while others degraded slightly.
- Memory & runtime: CBAM adds a small parameter and compute overhead; this run could not complete on Kaggle due to GPU OOM and was executed locally (25 epochs ≈ 1 day on local GPU).
- Suggested follow-ups:
	- Tune attention capacity (reduction ratio in ChannelAttention), add attention dropout, or apply CBAM selectively to intermediate DenseNet blocks.
	- Try ensembling baseline + CBAM models — CBAM helps some classes and hurts others, an ensemble may capture complementary strengths.
	- Check per-class threshold selection method (precision-recall based) if F1 behaviour is unexpected for specific classes.

Conclusions
-----------
- CBAM provides per-class gains on a subset of pathologies (notably Fibrosis, Edema, Hernia, Cardiomegaly) but did not improve global Avg AUROC in this run. Further tuning or selective application is advisable.
- CBAM increases parameter & memory usage; check the notebook for exact parameter counts.
- If Avg AUROC improves: CBAM is promising; otherwise tune regularization or attention capacity.

Notes
-----
- Could not complete on Kaggle due to GPU OOM. Run completed locally on a CUDA-enabled GPU (~1 day for 25 epochs).
