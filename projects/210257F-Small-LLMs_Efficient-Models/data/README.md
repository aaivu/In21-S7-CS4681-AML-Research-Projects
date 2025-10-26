# Datasets

This directory contains information about the datasets used in the EdgeMIN project. The actual dataset files are **not included** in this repository due to size constraints and licensing considerations.

---

## Datasets Used

EdgeMIN is evaluated on three tasks from the **GLUE (General Language Understanding Evaluation) benchmark**:

### 1. SST-2 (Stanford Sentiment Treebank)
- **Task:** Binary sentiment classification
- **Domain:** Movie reviews
- **Samples:** 67,349 training, 872 validation
- **Labels:** 2 (positive, negative)

### 2. MNLI (Multi-Genre Natural Language Inference)
- **Task:** Textual entailment (3-way classification)
- **Domain:** Diverse text genres
- **Samples:** 392,702 training, 9,815 validation (matched)
- **Labels:** 3 (entailment, neutral, contradiction)

### 3. QQP (Quora Question Pairs)
- **Task:** Paraphrase detection
- **Domain:** Question pairs
- **Samples:** 363,846 training, 40,430 validation
- **Labels:** 2 (duplicate, not duplicate)

---

## 🔗 How to Access the Datasets

### Option 1: Using Hugging Face Datasets (Recommended)

The easiest way to load the GLUE datasets is via the Hugging Face `datasets` library:
```python
from datasets import load_dataset

# Load SST-2
sst2_dataset = load_dataset("glue", "sst2")
print(sst2_dataset)
# Output: DatasetDict({
#     train: Dataset({features: ['sentence', 'label', 'idx'], num_rows: 67349})
#     validation: Dataset({features: ['sentence', 'label', 'idx'], num_rows: 872})
#     test: Dataset({features: ['sentence', 'label', 'idx'], num_rows: 1821})
# })

# Load MNLI
mnli_dataset = load_dataset("glue", "mnli")
print(mnli_dataset)
# Output: DatasetDict({
#     train: Dataset({num_rows: 392702})
#     validation_matched: Dataset({num_rows: 9815})
#     validation_mismatched: Dataset({num_rows: 9832})
#     test_matched: Dataset({num_rows: 9796})
#     test_mismatched: Dataset({num_rows: 9847})
# })

# Load QQP
qqp_dataset = load_dataset("glue", "qqp")
print(qqp_dataset)
# Output: DatasetDict({
#     train: Dataset({num_rows: 363846})
#     validation: Dataset({num_rows: 40430})
#     test: Dataset({num_rows: 390965})
# })
```

**Installation:**
```bash
pip install datasets
```

**Official Documentation:** [https://huggingface.co/docs/datasets/](https://huggingface.co/docs/datasets/)

---

### Option 2: Manual Download from GLUE Benchmark

You can download the datasets directly from the official GLUE website:

**GLUE Benchmark:** [https://gluebenchmark.com/](https://gluebenchmark.com/)

**Direct Download Link:** [https://gluebenchmark.com/tasks](https://gluebenchmark.com/tasks)

**Steps:**
1. Visit the GLUE website
2. Navigate to the "Tasks" section
3. Download the specific tasks (SST-2, MNLI, QQP)
4. Extract the files and place them in your working directory

**File Format:**
- TSV (Tab-Separated Values) files
- Columns: `sentence`, `label` (for SST-2)
- Columns: `premise`, `hypothesis`, `label` (for MNLI)
- Columns: `question1`, `question2`, `label` (for QQP)

---

### Option 3: TensorFlow Datasets

If you prefer TensorFlow, you can use `tensorflow_datasets`:
```python
import tensorflow_datasets as tfds

# Load SST-2
sst2_dataset = tfds.load('glue/sst2', split=['train', 'validation'])

# Load MNLI
mnli_dataset = tfds.load('glue/mnli', split=['train', 'validation_matched'])

# Load QQP
qqp_dataset = tfds.load('glue/qqp', split=['train', 'validation'])
```

**Installation:**
```bash
pip install tensorflow-datasets
```

**Official Documentation:** [https://www.tensorflow.org/datasets/catalog/glue](https://www.tensorflow.org/datasets/catalog/glue)
