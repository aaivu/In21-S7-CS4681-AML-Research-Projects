# Small LLMs:Mixture of Experts - 210017V

## Student Information

- **Index Number:** 210017V
- **Research Area:** Small LLMs:Mixture of Experts
- **GitHub Username:** @abineyan
- **Email:** ravichandran.21@cse.mrt.ac.lk

## Project Structure
```
210017V-Small-LLMs:Mixture-of-Experts/
├── README.md                    # This file
├── docs/
│   ├── research_proposal.md     # Your research proposal (Required)
│   ├── literature_review.md     # Literature review and references
│   ├── methodology.md           # Detailed methodology
│   └── progress_reports/        # Weekly progress reports
├── src/                         # Your source code
├── data/                        # Datasets and data files
├── experiments/                 # Experiment scripts and configs
├── results/                     # Experimental results
└── requirements.txt             # Project dependencies
```

## Getting Started

1. **Complete Research Proposal:** Fill out `docs/research_proposal.md`
2. **Literature Review:** Document your literature review in `docs/literature_review.md`
3. **Set Up Environment:** Add your dependencies to `requirements.txt`
4. **Start Coding:** Begin implementation in the `src/` folder
5. **Track Progress:** Use GitHub Issues to report weekly progress

## Milestones

- [ ] **Week 4:** Research Proposal Submission
- [ ] **Week 5:** Literature Review Completion  
- [ ] **Week 8:** Methodology Implementation
- [ ] **Week 12:** Final Evaluation

## Progress Tracking

Create GitHub Issues with the following labels for tracking:
- `student-210017V` (automatically added)
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

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/aaivu/In21-S7-CS4681-AML-Research-Projects.git
cd In21-S7-CS4681-AML-Research-Projects/projects/210017V-Small-LLMs_Mixture-of-Experts
```

### 2. Create and Activate a Virtual Environment
```bash
python3 -m venv venv
```
```bash
source venv/bin/activate  # On Linux / macOS
```
```bash
venv\Scripts\activate     # On Windows
```

### 3. Install Requirements

Install all dependencies listed in the requirements.txt file:
```bash
pip install -r requirements.txt
```

## Reproduce Experiments

### 1. Unzip Dataset
The dataset used in experiments is located at data/wikitext-2.zip.
Unzip it before running experiments:
```bash
unzip data/wikitext-2.zip
```

### 2. Running Experiments

To reproduce the training experiments, simply run:
```bash
python3 experiments/run_experiments.py
```
