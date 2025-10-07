# WavLM Large Fine-Tuning for ASR: Experiments on LibriSpeech

This repository contains experiments fine-tuning **WavLM-large** with a **CTC head** for automatic speech recognition (ASR) on **LibriSpeech**.  
The goal is to explore the performance of WavLM and aim to **outperform WeNet**’s benchmark WER of 2.7%.

---

## Objective
- Fine-tune **WavLM-large** on subsets of LibriSpeech and analyze performance.  
- Evaluate **Word Error Rate (WER)** improvement with larger datasets and optimized training strategies.  
- Establish a baseline and scaling strategy for ASR tasks.

---

## Experiments Overview

### Experiment 1: Fine-Tuning on 60% of LibriSpeech Clean-100
- **Purpose**: Initial experiment to test WavLM-large with a smaller dataset.  
- **Dataset**: 60% of LibriSpeech clean-100 (~60 hours).  
- **Epochs**: 3  
- **Training Time**: ~10 hours  

**Training Arguments:**

```python
training_args = TrainingArguments(
    output_dir="./wavlm-ctc-ex-1",
    group_by_length=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    eval_strategy="steps",
    num_train_epochs=3,
    fp16=True,
    save_steps=100,
    eval_steps=100,
    logging_steps=100,
    learning_rate=3e-4,
    weight_decay=0.005,
    warmup_steps=100,
    lr_scheduler_type="linear",
    adam_beta1= 0.9,
    adam_beta2= 0.999,
    adam_epsilon= 1e-08,
    max_grad_norm= 1.0,
    save_total_limit=2,
    push_to_hub=False,
    report_to="none"
)
```


### Experiment 2: Full LibriSpeech Clean-100 with Optimized Training
- **Purpose**: Improve WER by training on the full dataset and accelerating learning with optimized hyperparameters.  
- **Dataset**: Full LibriSpeech clean-100 (~100 hours).  
- **Epochs**: 4  
- **Training Time**: ~12 hours  

**Training Arguments:**

```python
training_args = TrainingArguments(
    output_dir="./wavlm-ctc-ex-2",
    group_by_length=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    eval_strategy="steps",
    num_train_epochs=4,
    fp16=True,
    save_steps=400,
    eval_steps=400,
    logging_steps=400,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=1000,
    lr_scheduler_type="cosine",
    adam_beta1= 0.9,
    adam_beta2= 0.999,
    adam_epsilon= 1e-08,
    max_grad_norm= 1.0,
    save_total_limit=2,
    gradient_checkpointing=True,
    push_to_hub=False,
    report_to="none"
)
```
--- 

## Training Performance

| Step | Training Loss | Validation Loss | WER |
|------|---------------|----------------|-----|
| 400  | 4.123900      | 2.946558       | 1.000000 |
| 800  | 2.849300      | 2.473579       | 1.000000 |
| 1200 | 1.319600      | 0.560408       | 0.509154 |
| 1600 | 0.650500      | 0.282510       | 0.265192 |
| 2000 | 0.434300      | 0.196894       | 0.173376 |
| 2400 | 0.338100      | 0.156762       | 0.136190 |
| 2800 | 0.290900      | 0.130854       | 0.113194 |
| 3200 | 0.263200      | 0.120677       | 0.102662 |
| 3600 | 0.235000      | 0.111170       | 0.091136 |
| 4000 | 0.204100      | 0.102542       | 0.082166 |
| 4800 | 0.167800      | 0.094883       | 0.075310 |
| 5200 | 0.167000      | 0.093479       | 0.072644 |
| 5600 | 0.156100      | 0.089587       | 0.070861 |
| 6000 | 0.150800      | 0.090331       | 0.068913 |
| 6400 | 0.147400      | 0.088526       | 0.068453 |
| 6800 | 0.146900      | 0.087806       | 0.068214 |


**Final WER**: **~6.8%** on validation after 4 epochs

---

## Key Observations

- Increasing dataset size significantly improves WER.

- Optimized training parameters (larger batch size, cosine scheduler, gradient checkpointing) accelerate convergence.

- Experiment 2 demonstrates a major reduction in WER compared to Experiment 1 (~18% → ~6.8%).

---

## 🔗 References
- [WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900)  
- [LibriSpeech ASR Corpus](https://www.openslr.org/12)  
- [WeNet](https://github.com/wenet-e2e/wenet)  

---

