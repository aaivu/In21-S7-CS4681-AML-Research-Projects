# 🧠 ResQEdge — Research Project (Project ID: SLM002)

**ResQEdge** is a research project focusing on efficient edge deployment of large language models for disaster response.  
This repository contains experiment scripts, datasets, and source code for model training, benchmarking, and evaluation.

---

## 📁 Repository Structure

```
.
├── data/                          # Datasets
│   └── disaster_corpus/
│       └── flood.json
│
├── docs/                          # Project documentation
│   ├── literature_review.md
│   ├── methodology.md
│   ├── progress_reports/
│   ├── research_proposal.md
│   └── usage_instructions.md
│
├── experiments/                   # Experiment scripts and configurations
│   ├── baseline_benchmarking/
│   │   ├── benchmark.py
│   │   ├── benchmark_utils.py
│   │   ├── plotting_utils.py
│   │   └── config.yaml
│   ├── disaster_corpus_benchmarking/
│   │   ├── ptq_benchmark_disaster.py
│   │   ├── qat_train_disaster.py
│   │   ├── benchmark_utils_disaster.py
│   │   └── qat_checkpoints/
│   ├── ptq_benchmarking/
│   │   └── benchmark.py
│   └── qat_benchmarking/
│       ├── qat_train.py
│       └── qat_benchmark.py
│
├── results/                       # Experiment outputs and plots
│
├── src/                           # Model training and inference scripts
│   ├── resqedge_train.py
│   ├── resqedge_inference.py
│   └── model_checkpoints/
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Environment Setup

To ensure reproducibility, it’s recommended to use a **Python virtual environment**.

### 1. Create and activate the virtual environment

#### 🪟 Windows (PowerShell)
```bash
python -m venv .venv
.venv\Scripts\activate
```

#### 🐧 Linux / 🍎 macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🧪 Running Experiments

### 1. Baseline Benchmarking
Run:
```bash
python experiments/baseline_benchmarking/benchmark.py
```

- Configuration can be adjusted in `experiments/baseline_benchmarking/config.yaml`.  
- Results (metrics, plots, and CSV files) will be stored in:
  ```
  results/baseline_benchmarking_results/
  ```

---

### 2. PTQ (Post-Training Quantization) Benchmarking
Run:
```bash
python experiments/ptq_benchmarking/benchmark.py
```

Results will be saved in:
```
results/ptq_benchmarking_results/
```

---

### 3. QAT (Quantization-Aware Training) Benchmarking

#### 🧩 Step 1 — Train (if you don’t have a checkpoint)
Before running the QAT benchmark, make sure you have trained the model.  
If you **don’t** have an existing checkpoint in:
```
experiments/qat_benchmarking/qat_checkpoints/
```
run the following:
```bash
python experiments/qat_benchmarking/qat_train.py
```

This will create a new QAT-trained model checkpoint.

#### 🧪 Step 2 — Run the benchmark
Once the training is complete (or if you already have a checkpoint), run:
```bash
python experiments/qat_benchmarking/qat_benchmark.py
```

Results will be saved in:
```
results/qat_benchmarking_results/
```

> ⚠️ **Note:** Checkpoints are not pushed to Git.  
> You must train the model locally using `qat_train.py` before running `qat_benchmark.py` if no checkpoint exists.

---

### 4. Disaster Corpus Benchmarking

The `flood.json` dataset is located at:
```
data/disaster_corpus/flood.json
```

This dataset is used for both **disaster-specific QAT and PTQ** experiments and also for model training under `src/`.

To reproduce disaster corpus experiments:

#### (a) Run PTQ Benchmark
```bash
python experiments/disaster_corpus_benchmarking/ptq_benchmark_disaster.py
```

#### (b) Train and Run QAT Benchmark
If you don’t have a QAT checkpoint:
```bash
python experiments/disaster_corpus_benchmarking/qat_train_disaster.py
```
Then benchmark:
```bash
python experiments/disaster_corpus_benchmarking/qat_train_disaster.py
```

Results will be stored in:
```
results/disaster_corpus_benchmarking_results/
```

---

## 🧩 Training and Inference (Source Code)

The core ResQEdge model is trained and evaluated from the `src/` directory.

### 1. Train the Model
```bash
python src/resqedge_train.py
```

### 2. Run Inference
```bash
python src/resqedge_inference.py
```

> ⚠️ **Note:** Model checkpoints generated during training (in `src/model_checkpoints/`) are not committed to Git.  
> You must train the model locally before running inference or benchmarking.

---

## 📦 Results and Outputs

Each experiment automatically creates its own results directory under `results/`.  
Typical contents include:
- `*.csv` files (quantitative results)
- `plots/` (visualizations)
- `.zip` archives (compressed experiment summaries)

Example:
```
results/baseline_benchmarking_results/
├── baseline_results.csv
├── plots/
│   ├── bar_latency.png
│   ├── scatter_Perplexity_vs_F1.png
│   └── ...
└── results_YYYYMMDD_HHMMSS.zip
```

---

## 🧭 Notes

- All experiment configurations (paths, models, datasets) are defined in their respective `config.yaml` or script headers.  
- Make sure your working directory is the **project root** when running scripts.
- Before running quantized experiments (PTQ/QAT), ensure the base models have been trained or are available locally.

---

## 🏁 Citation
If you use this repository or related findings in your research, please cite:

```
ResQEdge: Efficient Edge Deployment of Quantized Language Models for Disaster Response
Project ID: SLM002
University of Moratuwa
```
