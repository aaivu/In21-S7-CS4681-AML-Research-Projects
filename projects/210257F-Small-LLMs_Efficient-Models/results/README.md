# 📊 Results Directory — EdgeMIN Project

This folder contains the **evaluation outputs, trained model artifacts, and performance summaries** generated throughout the EdgeMIN model compression pipeline.

EdgeMIN is a three-stage systematic compression framework combining **relational knowledge distillation**, **structured pruning**, and **post-training quantization** to make Transformer models efficient for **edge device deployment**.

---

## 📁 Directory Overview

```bash
EdgeMIN_Project/
│
├── distilled_student/              # Stage 1 output: Distilled MiniLMv2 student model
├── figures/                        # Visual results and plots (e.g., size vs. accuracy, latency comparisons)
├── models/                         # Final trained and compressed model weights
├── results/                        # (You are here) — all evaluation metrics and outputs
├── tmp_pruning_finetune/           # Intermediate checkpoints during pruning fine-tuning
├── tmp_qat_training/               # Temporary artifacts from quantization-aware training
│
├── accuracy_results.csv            # Accuracy metrics for all models across compression stages
├── baseline_results.csv            # Baseline (teacher model) evaluation before compression
├── evaluation_results.csv          # Combined accuracy, latency, FLOPs, and model size results
├── evaluation_results.xlsx          # Excel version of full evaluation metrics for easy analysis
├── evaluation_summary.txt          # Summary of compression performance and key observations
├── quantized_model.pth             # Final quantized and pruned EdgeMIN model (INT8 weights)
└── throughput_results.csv          # CPU throughput and latency measurements (samples/sec)
```

## 🔗 Access Trained Models

All trained and compressed models are also available in the following shared Google Drive folder:

👉 **[Access Trained Models on Google Drive](https://drive.google.com/drive/u/0/folders/1EabMrsx7YFVIBq4K9B4Z6sIbHRfwhSxs)**

This includes:
- Distilled student model checkpoints  
- Pruned and fine-tuned models  
- Quantized INT8 deployment models  
- Evaluation results and logs  
