# Adapting Multimodal Foundation Models for Few-Shot Learning: A Comprehensive Study on Contrastive Captioners - 210407R

## Student Information

- **Index Number:** 210407R
- **Research Area:** Few-Shot Adaptation of Contrastive Captioners (CoCa)
- **GitHub Username:** @luckylukezzz
- **Email:** patalee.21@cse.mrt.ac.lk
- **Supervisor:** Uthayasanker Thayasivam
- **Institution:** Department of Computer Science and Engineering, University of Moratuwa, Sri Lanka

## Project Overview

This research presents a comprehensive empirical study on few-shot adaptation of the Contrastive Captioners (CoCa) model for image classification. We systematically evaluate a hierarchy of adaptation methods ranging from parameter-free hybrid prototyping to parameter-efficient fine-tuning (PEFT) via LoRA. The study addresses the challenge of adapting large-scale multimodal foundation models to downstream tasks with sparse labeled data, avoiding computational costs and overfitting associated with full fine-tuning.

**Key Research Questions:**
- How can CoCa's multimodal nature be leveraged in few-shot scenarios?
- What is the optimal balance between performance and computational efficiency?
- How do different adaptation strategies perform across varying data regimes?

## Project Structure

```
210407R-Few-Shot-CoCa/
├── README.md                          # This file
├── requirements.txt                   # Project dependencies
├── docs/
│   ├── research_proposal.md          # Research proposal
│   ├── methodology.md                # Detailed methodology
│   ├── literature_review.md          # Literature review and analysis
│   └── progress_reports/             # Weekly progress reports
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # Mini-ImageNet data pipeline
│   ├── models.py                     # CoCa model wrapper
│   ├── adaptation/
│   │   ├── hybrid_prototype.py       # Parameter-free prototype method
│   │   ├── linear_probing.py         # Linear probing strategy
│   │   └── lora_finetuning.py        # LoRA fine-tuning implementation
│   ├── losses.py                     # CE, Prototypical, SupCon losses
│   └── utils.py                      # Utility functions
├── experiments/
│   ├── config_hybrid.yaml            # Hybrid prototype config
│   ├── config_linear_probing.yaml    # Linear probing config
│   ├── config_lora.yaml              # LoRA fine-tuning config
│   ├── run_hybrid_prototype.py       # Hybrid prototype experiments
│   ├── run_linear_probing.py         # Linear probing experiments
│   └── run_lora_finetuning.py        # LoRA fine-tuning experiments
├── data/
│   └── mini_imagenet/                # Mini-ImageNet dataset location
├── results/
│   ├── hybrid_prototype/             # Hybrid prototype results
│   ├── linear_probing/               # Linear probing results
│   ├── lora_finetuning/              # LoRA fine-tuning results
│   └── analysis.py                   # Result visualization and analysis
└── notebooks/
    ├── exploratory_analysis.ipynb    # Data exploration
    ├── results_visualization.ipynb   # Results analysis and visualization
    └── ablation_studies.ipynb        # Ablation study notebooks
```

## Getting Started

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/luckylukezzz/210407R-Few-Shot-CoCa.git
cd 210407R-Few-Shot-CoCa

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

```bash
# Download Mini-ImageNet dataset
# Place in data/mini_imagenet/ directory
# Expected structure:
# data/mini_imagenet/
# ├── train/
# ├── val/
# └── test/
```

### 3. Review Documentation

- Start with `docs/research_proposal.md` for project overview
- Read `docs/methodology.md` for detailed methodology
- Check `docs/literature_review.md` for related work

### 4. Run Experiments

```bash
# Hybrid prototype baseline
python experiments/run_hybrid_prototype.py --config experiments/config_hybrid.yaml

# Linear probing
python experiments/run_linear_probing.py --config experiments/config_linear_probing.yaml

# LoRA fine-tuning
python experiments/run_lora_finetuning.py --config experiments/config_lora.yaml

# Analyze results
python results/analysis.py
```

## Key Dependencies

- **PyTorch** (≥1.9.0) - Deep learning framework
- **Hugging Face Transformers** (≥4.20.0) - Pre-trained model access
- **Hugging Face Datasets** - Data loading utilities
- **NumPy, Pandas** - Data manipulation
- **Scikit-learn** - Machine learning utilities
- **Matplotlib, Seaborn** - Visualization
- **peft** - Parameter-efficient fine-tuning library
- **pyyaml** - Configuration management

See `requirements.txt` for complete list and versions.

## Methodology Summary

### Strategy 1: Hybrid Prototype Classification (Parameter-Free)

Combines visual and textual embeddings without any training:
- Visual prototypes: normalized mean of image embeddings
- Textual prototypes: mean of text embeddings with prompt ensembling
- Hybrid fusion: weighted combination with learnable parameter α
- Classification via cosine similarity

### Strategy 2: Linear Probing

Trains only a classification head with frozen visual encoder:
- Frozen CoCa image encoder preserves pre-trained knowledge
- New linear classification head (trainable)
- Strong data augmentation in low-shot scenarios
- Cross-entropy loss with label smoothing

### Strategy 3: LoRA Fine-Tuning

Parameter-efficient adaptation via low-rank decomposition:
- Adaptive rank scaling (4, 8, or 16) based on shot count
- Adaptive target module selection
- Hybrid loss: $L_{total} = L_{metric} + L_{CE}$
- Three loss functions: Cross-Entropy, Prototypical, Supervised Contrastive

## Project Milestones

- [x] **Week 1-2:** Literature review and related work analysis
- [x] **Week 3-4:** Data preparation and experimental framework setup
- [ ] **Week 5-6:** Hybrid prototype implementation and baseline experiments
- [ ] **Week 7-8:** Linear probing implementation and hyperparameter tuning
- [ ] **Week 9-10:** LoRA implementation with multiple loss functions
- [ ] **Week 11-12:** Comprehensive experiments across all shot settings
- [ ] **Week 13-14:** Comparative analysis and visualization
- [ ] **Week 15-16:** Final report writing and documentation
- [ ] **Week 17:** Final submission

## Results Overview

Results will be organized by strategy and shot setting:

```
results/
├── hybrid_prototype/
│   └── accuracy_across_shots.csv
├── linear_probing/
│   ├── augmentation_analysis.csv
│   └── accuracy_across_shots.csv
└── lora_finetuning/
    ├── loss_comparison.csv
    ├── adaptive_config_results.csv
    └── accuracy_across_shots.csv
```

## Key Findings (Expected)

- **Hybrid Prototype:** Strong performance in extremely low-shot scenarios (1-5 shots) without training overhead
- **Linear Probing:** Competitive results with minimal parameters when augmentation is carefully tuned
- **LoRA Fine-Tuning:** 
  - Metric-based losses excel in low-data regimes (1-5 shots)
  - Cross-entropy becomes competitive and superior at higher shots (20+)
  - Adaptive configuration provides consistent improvements across all settings

## Code Usage Example

```python
from src.adaptation.hybrid_prototype import HybridPrototype
from src.data_loader import load_mini_imagenet

# Load data
train_loader, val_loader, test_loader = load_mini_imagenet(
    shots=5,
    ways=5,
    data_path='data/mini_imagenet'
)

# Initialize model
model = HybridPrototype(
    model_name='coca_ViT-L/14',
    fusion_weight=0.7
)

# Evaluate
accuracy = model.evaluate(test_loader)
print(f"Accuracy: {accuracy:.2f}%")
```

## Tracking Progress

Use GitHub Issues with labels for progress tracking:
- `literature-review` - Literature review tasks
- `implementation` - Code implementation
- `experiments` - Experiment execution
- `analysis` - Results analysis
- `documentation` - Writing and documentation
- `bug` - Issues and fixes

Create weekly progress reports in `docs/progress_reports/` documenting:
- Tasks completed
- Challenges encountered
- Next steps
- Code commits and changes

## Contributing Guidelines

1. **Code Quality:**
   - Follow PEP 8 style guidelines
   - Add docstrings to all functions
   - Include type hints where possible
   - Write unit tests for utilities

2. **Experimentation:**
   - Document all hyperparameters in config files
   - Save results with timestamps and configurations
   - Create reproducible random seeds
   - Log all experiments systematically

3. **Documentation:**
   - Keep README updated
   - Document code changes
   - Add comments for complex logic
   - Maintain clear commit messages

4. **Git Workflow:**
   - Create feature branches for new work
   - Commit frequently with clear messages
   - Push to remote regularly
   - Create pull requests for major changes

## Academic Integrity

- All work is original and properly attributed
- External code or algorithms are clearly cited
- Collaboration is acknowledged and documented
- University academic integrity policies are followed
- All references are properly formatted in academic style

## Resources

- **Research Paper:** Base paper on CoCa few-shot adaptation
- **CoCa Model:** https://github.com/lucidrains/CoCa-pytorch
- **Hugging Face:** https://huggingface.co/docs
- **Mini-ImageNet:** https://github.com/rpmignani/mini-imagenet
- **LoRA Paper:** https://arxiv.org/abs/2106.09685

## Contact & Support

For questions or issues, contact:
- **Student:** Narasinghe N.K.B.M.P.K.B (patalee.21@cse.mrt.ac.lk)
- **Supervisor:** Uthayasanker Thayasivam (rtuthaya@cse.mrt.ac.lk)
- **Institution:** University of Moratuwa, Department of Computer Science and Engineering

---

**Last Updated:** October 2025  
**Status:** Active Development  
**Regular commits and clear documentation are essential for project success!**