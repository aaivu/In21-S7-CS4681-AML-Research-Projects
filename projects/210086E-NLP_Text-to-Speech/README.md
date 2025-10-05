# NLP:Text-to-Speech - 210086E

## Student Information

- **Index Number:** 210086E
- **Research Area:** NLP:Text-to-Speech
- **GitHub Username:** @irudachirath
- **Email:** iruda.21@cse.mrt.ac.lk

## Project Structure
```
210086E-NLP:Text-to-Speech/
├── README.md                    # This file
├── docs/
│   ├── research_proposal.md     # Research proposal overview
│   ├── literature_review.md     # Literature review and references
│   ├── methodology.md           # Detailed methodology
│   ├── istft_vocoder_design.md  # Vocoder architecture documentation
│   ├── training_guide.md        # Training instructions and configurations
│   └── progress_reports/        # Weekly progress reports
├── src/                         # Source code
│   ├── data/                    # Dataset loaders
│   └── models/                  # Model implementations
├── data/                        # Datasets (VCTK-Corpus)
├── experiments/                 # Jupyter notebooks for experiments
├── scripts/                     # Training and utility scripts
├── results/                     # Experimental results and outputs
├── logs/                        # Training logs
├── checkpoints/                 # Model checkpoints
└── requirements.txt             # Project dependencies
```

## Getting Started

1. **Review Documentation:** Check `docs/research_proposal.md` for project overview
2. **Literature Review:** See `docs/literature_review.md` for background research
3. **Set Up Environment:** Install dependencies from `requirements.txt`
4. **Start Development:** Implement code in the `src/` folder
5. **Run Experiments:** Use notebooks in `experiments/` for testing and analysis
6. **Track Progress:** Use GitHub Issues to report weekly progress

## Milestones

- [x] **Week 1-2:** VITS Baseline Setup & Analysis
- [x] **Week 3-4:** Phase 3 - Vocoder Optimization (iSTFT Implementation)
- [x] **Mid-Evaluation:** 4-5 page report submitted (IEEE format)
- [ ] **Week 5:** Vocoder Training & Evaluation
- [ ] **Week 6:** Integration & Benchmarking
- [ ] **Week 12:** Final Evaluation

## Current Status: Mid-Evaluation Submitted ✅

### Completed (Phase 3: Vocoder Optimization)
- ✅ HiFi-GAN bottleneck analysis
- ✅ iSTFT vocoder architecture design
- ✅ Single-band iSTFT vocoder implementation (~2.5M params)
- ✅ Multi-band extension (3-band parallel processing)
- ✅ VCTK dataset loader with train/val/test split
- ✅ Complete training pipeline with TensorBoard logging
- ✅ Testing framework and benchmarking tools
- ✅ Comprehensive documentation

### Next Steps
1. **Complete Training** - Continue for 6 more epochs to reach target MCD <6 dB
2. **Multi-band Implementation** - Implement and test multi-band iSTFT vocoder
3. **Comprehensive Benchmarking** - RTF measurements on CPU/GPU
4. **VITS Integration** - Replace HiFi-GAN in full VITS pipeline
5. **Comparative Evaluation** - Benchmark against baseline systems
6. **Subjective Testing** - Conduct listening tests for quality assessment

## Quick Start Training

```bash
# Navigate to project
cd projects/210086E-NLP_Text-to-Speech

# Activate environment
source .venv/bin/activate

# Start training
python scripts/train_vocoder.py

# Monitor progress
tensorboard --logdir logs/istft_vocoder
```

See `docs/training_guide.md` for detailed instructions.

## Progress Tracking

Create GitHub Issues with the following labels for tracking:
- `student-210086E` (automatically added)
- `literature-review`, `implementation`, `evaluation`, etc.
- Tag supervisors (@supervisor) for feedback

## Resources

- Check the main repository `docs/` folder for guidelines
- Use the `templates/` folder for document templates
- Refer to `docs/project_guidelines.md` for detailed instructions

## Academic Integrity

- All work must be original
- Properly cite all references
- Acknowledge any collaboration
- Follow university academic integrity policies

---

**Remember:** Regular commits and clear documentation are essential for project success!