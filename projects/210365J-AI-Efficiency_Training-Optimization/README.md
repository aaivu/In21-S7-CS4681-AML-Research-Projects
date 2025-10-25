# AI Efficiency: Training Optimization - 210365J

## Student Information

- **Index Number:** 210365J
- **Research Area:** AI Efficiency: Training Optimization
- **GitHub Username:** @MamtaNallaretnam
- **Email:** nallaretnam.21@cse.mrt.ac.lk
- **Supervisor:** Dr. Uthayasankar Thayasivam
- **Institution:** University of Moratuwa, Sri Lanka

## Project Overview

This research project focuses on developing a systematic methodology for optimizing training compute in large-scale AI models through dynamic resource allocation, adaptive precision management, and strategic recomputation policies. The project aims to achieve significant reductions in time-to-convergence and memory consumption while maintaining model quality across diverse AI architectures.

**Key Research Question:** How can we systematically optimize AI training compute through integrated dynamic resource allocation strategies to address the exponential growth in computational requirements?

## Project Structure
```
210365J-AI-Efficiency:Training-Optimization/
├── README.md                    
├── docs/
│   ├── research_proposal.md     
│   ├── literature_review.md     
│   ├── methodology.md           
│   └── progress_reports/        
├── src/                         
├── data/                        
├── experiments/                 
├── results/                     
└── requirements.txt             
```

## Research Components

### Core Optimization Techniques
- **Dynamic Batch Scheduling:** Variance-aware gradient accumulation with adaptive k-selection
- **Mixed Precision Training:** FP16 compute with FP32 master weights and dynamic loss scaling
- **Strategic Activation Checkpointing:** Optimal checkpoint placement at √L intervals
- **Adaptive Learning Rate:** Composite schedule with linear warmup and cosine annealing

### Evaluation Benchmarks
- **Computer Vision:** ResNet-50 on CIFAR-10 dataset
- **Natural Language Processing:** GPT-2 Small on WikiText-2 dataset

### Key Metrics
- Time-to-convergence reduction (target: 25-30%)
- Peak memory consumption reduction (target: 25%)
- Training throughput improvement
- Model quality preservation (accuracy/perplexity within ±0.5%)

## Getting Started

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/MamtaNallaretnam/210365J-AI-Efficiency-Training-Optimization.git
cd 210365J-AI-Efficiency-Training-Optimization

# Install dependencies
pip install -r requirements.txt
```

## Hardware & Software Requirements

### Hardware
- 4× NVIDIA RTX 3090 GPUs (24GB VRAM each)
- AMD Ryzen 9 5950X CPU (16 cores, 32 threads)
- 64GB DDR4-3600 RAM
- 2TB NVMe SSD storage

### Software
- Ubuntu 22.04 LTS
- CUDA 12.1, cuDNN 8.9.0
- Python 3.9+
- PyTorch 2.0.1
- DeepSpeed 0.10.0
- Transformers 4.30.0

## Expected Outcomes

- **Systematic Optimization Framework:** Modular components for training efficiency
- **Empirical Results:** 23-28% time-to-convergence reduction, 25% memory savings
- **Scalability Analysis:** 94.6% multi-GPU scaling efficiency demonstration
- **Practical Implementation:** PyTorch-compatible framework with minimal modifications
- **Research Contribution:** Clear path toward doubling training efficiency

## Resources

- [Project Guidelines](link-to-guidelines)
- [Templates Directory](link-to-templates) for document templates
- [Main Repository Docs](link-to-docs) for additional guidelines
- Reference papers in `docs/literature_review.md`

## Academic Integrity

- All work must be original and properly cited
- Acknowledge collaborations and external contributions
- Follow University of Moratuwa academic integrity policies
- Maintain transparent research practices with reproducible results
- Document all references in appropriate academic format

## Contact

- **Student:** Mamta Nallaretnam - nallaretnam.21@cse.mrt.ac.lk
- **Supervisor:** Dr. Uthayasanker Thayasivam - rtuthaya@cse.mrt.ac.lk
- **GitHub:** [@MamtaNallaretnam](https://github.com/MamtaNallaretnam)