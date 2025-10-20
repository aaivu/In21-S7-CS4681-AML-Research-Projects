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
- [x] **Week 5:** Vocoder Training & Evaluation
- [x] **Week 6:** Integration & Benchmarking
- [x] **Week 12:** Final Evaluation

## Current Status: Project Complete ✅

### Completed Work
- ✅ HiFi-GAN bottleneck analysis and profiling
- ✅ iSTFT vocoder architecture design and implementation
- ✅ Single-band iSTFT vocoder V2 (~2.5M params)
- ✅ Multi-band vocoder architecture design
- ✅ VCTK dataset loader with train/val/test split
- ✅ Complete training pipeline with TensorBoard logging
- ✅ **Model training completed** (100 epochs, ~250K steps)
- ✅ **Comprehensive evaluation** (MCD: 5.21 dB, RTF: 0.22)
- ✅ **Detailed evaluation report** with analysis and improvements
- ✅ Testing framework and benchmarking tools
- ✅ Full technical documentation

### Evaluation Results Summary
- **Model Size:** 2.5M parameters (~10MB)
- **Speed:** RTF 0.22 (real-time capable, 4-5× faster than playback)
- **Quality:** MCD 5.21 dB, SNR 18.9 dB
- **Inference Time:** ~7ms per utterance on GPU
- **Status:** Real-time capable, good quality with identified improvements

See `results/istft_vocoder_v2_evaluation_report.md` for detailed analysis.

### Next Steps
1. **Final Report** - Complete comprehensive project report (In Progress)
2. **Multi-band Implementation** - Implement and test multi-band iSTFT vocoder for quality improvement
3. **Advanced Enhancements** - Phase modeling improvements, high-frequency emphasis loss
4. **VITS Integration** - Replace HiFi-GAN in full VITS pipeline
5. **Production Optimization** - Post-processing refiner, deployment guide

## Quick Start

### Using Trained Models

```bash
# Navigate to project
cd projects/210086E-NLP_Text-to-Speech

# Activate environment
source .venv/bin/activate

# Use trained vocoder (Best MCD checkpoint)
# See docs/usage_instructions.md for inference examples

# View evaluation results
# Open results/istft_vocoder_v2_evaluation_report.md

# Check training logs
tensorboard --logdir logs/istft_vocoder_v2
```

### Training from Scratch

```bash
# Start training (if needed)
python scripts/train_vocoder.py

# Monitor progress
tensorboard --logdir logs/istft_vocoder_v2
```

**Available Checkpoints:**
- `checkpoints/istft_vocoder_v2/best_mcd.pt` - Best perceptual quality (Recommended)
- `checkpoints/istft_vocoder_v2/best_loss.pt` - Best reconstruction loss
- Multiple epoch and step checkpoints available

See `docs/training_guide.md` for detailed training instructions and `docs/usage_instructions.md` for inference guide.

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