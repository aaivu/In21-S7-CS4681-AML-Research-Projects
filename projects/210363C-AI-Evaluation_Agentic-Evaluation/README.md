# AI Evaluation:Agentic Evaluation - 210363C

## Student Information

- **Index Number:** 210363C
- **Research Area:** AI Evaluation:Agentic Evaluation
- **GitHub Username:** @CheliM7
- **Email:** chemini.21@cse.mrt.ac.lk

## Project Structure

```
210363C-AI-Evaluation:Agentic-Evaluation/
├── README.md                    # This file
├── docs/
│   ├── research_proposal.md     # Research proposal
│   ├── literature_review.md     # Literature review and references
│   ├── methodology.md           # Detailed methodology
│   └── progress_reports/        # Weekly progress reports
├── data/                        # Datasets and data files
├── experiments/                 # Experiment scripts and configs
│   └── logs/
├── results/                     # Experimental results
├── src/                         # Source code
│   ├── evaluator/
│   ├── model/
│   ├── sanity-checks/
│   └── utils/
├── visualizations/
│   └──  visualizations.py
├── .gitignore
├── .env
├── config.py
├── requirements.txt             # Project dependencies
└── run.py

```

## Prerequisites

- Python 3.8 or higher
- `pip` (Python package installer)

## Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/aaivu/In21-S7-CS4681-AML-Research-Projects.git
   cd projects
   cd 210363C-AI-Evaluation_Agentic-Evaluation
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv env
   ```

3. **Activate the Virtual Environment**

   - On Windows:

     ```bash
     .\env\Scripts\activate
     ```

   - On macOS/Linux:
     ```bash
     source env/bin/activate
     ```

4. **Create a `.env` File**
   In the root directory, create a file named `.env` and add the following values:

```env
GROQ_API_KEY=
MODEL_ID=
DATA_PATH=data/math_easy_int_120.jsonl
```

4.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Run the Project**
    (for initial testing, only the first five rows of the dataset will be processed. Modify `run.py` to handle the entire dataset as needed.)

        ```bash
        python run.py
        ```

6.  **Generate Visualizations**
    ```bash
    python src/visualizations/visualizations.py
    ```

## Getting Started

1. **Complete Research Proposal:** Fill out `docs/research_proposal.md`
2. **Literature Review:** Document your literature review in `docs/literature_review.md`
3. **Set Up Environment:** Add your dependencies to `requirements.txt`
4. **Start Coding:** Begin implementation in the `src/` folder
5. **Track Progress:** Use GitHub Issues to report weekly progress

## Milestones

- [✔] **Week 4:** Research Proposal Submission
- [✔] **Week 5:** Literature Review Completion
- [✔] **Week 8:** Methodology Implementation
- [ ] **Week 12:** Final Evaluation

## Progress Tracking

Create GitHub Issues with the following labels for tracking:

- `student-210363C` (automatically added)
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
