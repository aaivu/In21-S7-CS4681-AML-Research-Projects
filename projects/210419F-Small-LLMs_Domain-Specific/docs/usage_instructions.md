#  Project Usage Instructions

This document outlines the step-by-step guide to setting up and running the project pipeline — from environment creation to model training on the **MACCROBAT** dataset.

---

##  1. Environment Setup

### Step 1. Create and activate a virtual environment
```bash
# Create a virtual environment
python3 -m venv venv

# For Windows:
venv\Scripts\activate
```

### Step 2. Install dependencies
Make sure you have the `requirements.txt` file in your project root, then run:
```bash
pip install -r requirements.txt
```

---

## 2. Tokenizer Enhancement

### Step 1. Prepare the required data files
Ensure the following files are present in the directory:
```
data/normalize/pubmed.json
data/normalize/wiki.json
```

### Step 2. Run the tokenizer enhancement script
Execute the following command:
```bash
python enhanced_tokenizer.py
```

This step will enhance the tokenizer using biomedical data and **save the enhanced tokenizer and model** inside the `medtok/` directory.

**Output directory:**
```
medtok/
 ├── tokenizer/
 └── model/
```

---

## 3. Knowledge Distillation

Run the **knowledge distillation** process using the Jupyter Notebook:

```bash
jupyter notebook knowledge_distillation.ipynb
```

This will perform the knowledge distillation step and **save the distilled model** to:

```
artifacts/distilled_model/
```

---

## 4. Model Training on MACCROBAT Dataset

After distillation, train the enhanced model on the **MACCROBAT** dataset using:

```bash
jupyter notebook enhanced_distilled_model.ipynb
```

This step fine-tunes the model for biomedical named entity recognition (NER) on the MACCROBAT dataset.

**The final trained model will be saved in:**
```
maccrobat_ner_model/
```

---

## Summary of Workflow

| Step | Description | Output Folder |
|------|--------------|----------------|
| 1 | Environment setup | — |
| 2 | Tokenizer enhancement (`enhanced_tokenizer.py`) | `medtok/` |
| 3 | Knowledge distillation (`knowledge_distillation.ipynb`) | `artifacts/distilled_model/` |
| 4 | Final model training (`enhanced_distilled_model.ipynb`) | `maccrobat_ner_model/` |

---


