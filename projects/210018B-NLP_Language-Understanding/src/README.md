# Enhanced One-Shot Learning for LAMBADA

This project implements a hybrid example selection strategy for one-shot learning on the LAMBADA dataset, combining semantic similarity and syntactic compatibility to achieve **73.30% accuracy** - outperforming random selection (70.93%) and the published GPT-3 result (72.5%).

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Installation](#️-installation)
- [Configuration](#️-configuration)
- [Usage](#-usage)
- [Troubleshooting](#-troubleshooting)
- [Citation](#-citation)
- [License](#-license)

---

## 🎯 Overview
The LAMBADA dataset requires models to predict the final word of passages where accurate prediction necessitates understanding long-range dependencies. This project demonstrates that intelligent example selection significantly improves one-shot learning performance through:

- **Semantic Similarity:** Using Sentence-BERT embeddings (20% weight)
- **Syntactic Compatibility:** Using Part-of-Speech matching (80% weight)
- **Hybrid Scoring:** Optimal combination achieving 73.30% accuracy

---

## ✨ Features
- 🎯 Hybrid example selection combining semantic and syntactic similarity
- 📊 Multiple evaluation modes (random baseline, semantic-only, POS-only, hybrid)
- 🔧 Configurable weighting parameter (`α`) for semantic vs. syntactic balance
- 📝 Detailed CSV logging of predictions and results
- 🚀 Efficient precomputation of embeddings and POS tags
- 📈 Grid search support for optimal `α` parameter tuning

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for faster embedding generation)
- OpenAI API key

### Step 1: Clone the Repository
```bash
git clone https://github.com/aaivu/In21-S7-CS4681-AML-Research-Projects.git
cd projects/210018B-NLP_Language-Understanding
```

### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```
**`requirements.txt`:**
```
openai==1.109.1
sentence-transformers==5.1.1
spacy==3.8.7
datasets
torch
numpy
pandas
python-dotenv
```

### Step 4: Download spaCy Model
```bash
python -m spacy download en_core_web_sm
```

---

## ⚙️ Configuration

### Step 1: Create Environment File
Create a `.env` file in the project root:
```bash
touch .env
```

### Step 2: Add API Keys
Edit `.env` and add your credentials:
```env
API_KEY=your_openai_api_key_here
HF_KEY=your_huggingface_token_here  # Optional, for private datasets
```
To get your OpenAI API key:
1. Go to https://platform.openai.com/api-keys
2. Create a new API key
3. Copy and paste it into the `.env` file

> **Note:** Keep your `.env` file secure and never commit it to version control!

---

## 🚀 Usage

### Basic Usage
Run the evaluation with default parameters (1-shot, hybrid selection, α=0.2):
```bash
python src/main.py
```

### Custom Configuration
Edit the parameters in `main.py`:
```python
from lambada_eval import evaluate

if __name__ == "__main__":
    evaluate(
        n_samples=1000,              # Number of test samples (max=5153)
        k=1,                         # Number of shots (1 for one-shot)
        model="gpt-3.5-turbo-instruct",
        mode="cloze",                # Prompt format: "cloze" or "default"
        use_POS_sem=True,            # Enable hybrid selection
        alpha=0.2,                   # Semantic weight (0.0-1.0)
        log_path="lambada_results.csv"
    )
```

### Run Different Experiments

**1. Random Baseline (No Intelligent Selection)**
```python
evaluate(
    n_samples=1000,
    k=1,
    use_POS_sem=False,  # Disable hybrid selection
    alpha=0.2,
    log_path="random_baseline.csv"
)
```

**2. Semantic-Only Selection (α=1.0)**
```python
evaluate(
    n_samples=1000,
    k=1,
    use_POS_sem=True,
    alpha=1.0,  # 100% semantic, 0% syntactic
    log_path="semantic_only.csv"
)
```

**3. POS-Only Selection (α=0.0)**
```python
evaluate(
    n_samples=1000,
    k=1,
    use_POS_sem=True,
    alpha=0.0,  # 0% semantic, 100% syntactic
    log_path="pos_only.csv"
)
```

**4. Optimal Hybrid Selection (α=0.2)**
```python
evaluate(
    n_samples=1000,
    k=1,
    use_POS_sem=True,
    alpha=0.2,  # 20% semantic, 80% syntactic (optimal)
    log_path="hybrid_optimal.csv"
)
```

**5. Zero-Shot (No Examples)**
```python
evaluate(
    n_samples=1000,
    k=0,  # No examples
    use_POS_sem=False,
    log_path="zero_shot.csv"
)
```

### Grid Search for Optimal α
Create a separate script `grid_search.py`:
```python
from lambada_eval import evaluate

alphas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
results = {}

for alpha in alphas:
    print(f"\n{'='*60}")
    print(f"Testing α = {alpha}")
    print(f"{'='*60}\n")
    
    accuracy = evaluate(
        n_samples=1000,
        k=1,
        use_POS_sem=True,
        alpha=alpha,
        log_path=f"results_alpha_{alpha}.csv"
    )
    results[alpha] = accuracy

# Print summary
print("\n" + "="*60)
print("GRID SEARCH RESULTS")
print("="*60)
for alpha, acc in results.items():
    print(f"α = {alpha:.1f}: {acc:.2f}%")
print(f"\nBest α: {max(results, key=results.get)} ({max(results.values()):.2f}%)")
```
Run with:
```bash
python grid_search.py
```

## 🔧 Troubleshooting

### Common Issues

**1. OpenAI API Error: Invalid API Key**
- **Error:** `Incorrect API key provided`
- **Solution:** Check that your `.env` file contains a valid OpenAI API key: `cat .env`

**2. Rate Limit Exceeded**
- **Error:** `Rate limit reached for requests`
- **Solution:** Add delays between requests or reduce `n_samples`.

**3. CUDA Out of Memory**
- **Error:** `RuntimeError: CUDA out of memory`
- **Solution:** Force CPU usage in `lambada_eval.py`: `device = "cpu"`

**4. spaCy Model Not Found**
- **Error:** `OSError: [E050] Can't find model 'en_core_web_sm'`
- **Solution:** Download the model: `python -m spacy download en_core_web_sm`

**5. Dataset Download Error**
- **Error:** `ConnectionError: Couldn't reach https://huggingface.co`
- **Solution:** Check your internet connection or try forcing a re-download.

### Performance Optimization
To speed up evaluation:
- Use a GPU for embedding generation (already implemented).
- Reduce `n_samples` for quick testing.
- For memory optimization, clear the CUDA cache periodically: `torch.cuda.empty_cache()`

---

## 📚 Citation
If you use this code in your research, please cite:
```bibtex
@article{sivakumar2025lambada,
  title={Enhanced One-Shot Learning for LAMBADA Through Semantic and Syntactic Example Selection},
  author={Sivakumar, Abisherk and Thayasivam, Uthayasanker},
  journal={University of Moratuwa},
  year={2025}
}
```

---

## 📄 License
This project is licensed under the MIT License - see the `LICENSE` file for details.

---

## 🙏 Acknowledgments
- OpenAI for providing API access to GPT-3.5-Turbo-Instruct
- The LAMBADA dataset creators for making their data publicly available
- Sentence-BERT and spaCy teams for excellent NLP tools
- University of Moratuwa, Department of Computer Science and Engineering

---

## 📧 Contact
**Abisherk Sivakumar**
- Department of Computer Science and Engineering
- University of Moratuwa
- **Email:** abisherk.21@cse.mrt.ac.lk

> **Note:** This is a research project. API costs may apply when running experiments. Monitor your OpenAI usage at https://platform.openai.com/usage
