# Student Usage Instructions

Welcome to your research project repository! This guide will help you understand how to use and maintain your project workspace effectively throughout your research journey.

## 🚀 Getting Started

### 1. Repository Access
You should have received an invitation to access your research repository:
- Repository URL: `https://github.com/aaivu/200710F_NLP:Reasoning`
- Check your email for the GitHub invitation
- Accept the invitation and set up your local environment

### 2. Initial Setup

#### Clone Your Repository
```bash
# Clone your repository
git clone https://github.com/aaivu/200710F_NLP:Reasoning.git
cd 200710F_NLP:Reasoning

# Set up your development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### Configure Git (if not already done)
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@university.lk"
```

### 3. Understanding Your Repository Structure
Your repository is pre-organized with the following structure:
```
your_project/
├── README.md                 # Your main project documentation
├── requirements.txt          # Python dependencies
├── .gitignore               # Files to ignore in version control
├── literature_review/       # Literature review materials
├── methodology/            # Research methodology and planning
├── implementation/         # Your code and experiments
├── results/               # Experimental results and analysis
├── documentation/         # Project documentation and reports
└── paper/                # Your research paper
```

## 📚 Working with Your Project

### Project Phases and Milestones

Your project is organized around four major milestones:

1. **Literature Review** (Week 4) - 15% of grade
2. **Methodology and Implementation** (Week 8) - 20% of grade
3. **Mid-Term Evaluation** (Week 10) - 25% of grade
4. **Final Evaluation and Submission** (Week 16) - 40% of grade

### Managing Milestones

#### Tracking Progress
Each milestone has a corresponding GitHub issue that serves as:
- Progress tracking checklist
- Deliverable requirements
- Evaluation criteria
- Communication tool with supervisors

#### Working with Issues
1. **Find your milestone issues** in the Issues tab
2. **Read through requirements** carefully
3. **Use checkboxes** to track your progress
4. **Add comments** to document your work and ask questions
5. **Tag supervisors** (@supervisor) when you need feedback

### Daily Workflow

#### 1. Start Your Day
```bash
# Update your local repository
git pull origin main

# Check for any new issues or feedback
# Visit: https://github.com/aaivu/200710F_NLP:Reasoning/issues
```

#### 2. Work on Your Tasks
- Follow the project structure guidelines
- Document your work as you go
- Commit changes regularly with descriptive messages

#### 3. End Your Day
```bash
# Stage your changes
git add .

# Commit with a descriptive message
git commit -m "Add literature review notes for deep learning papers"

# Push to remote repository
git push origin main
```

## 📊 Progress Reporting

### Weekly Progress Reports
You must submit weekly progress reports using the provided template:

1. **Create a new issue** using the "Weekly Progress Report" template
2. **Fill out all sections** honestly and thoroughly
3. **Tag your supervisor** for review
4. **Submit by end of each week**

### Milestone Submissions
For each milestone:

1. **Complete all deliverables** in the appropriate folders
2. **Update the milestone issue** with your progress
3. **Create a pull request** (if required)
4. **Tag supervisors** for review
5. **Present your work** (for major milestones)

## 💻 Development Best Practices

### Code Organization
- **Follow the project structure** outlined in `project_structure.md`
- **Use meaningful file names** and organize logically
- **Document your code** with comments and docstrings
- **Write tests** for critical functions

### Version Control Best Practices
```bash
# Make frequent, small commits
git add specific_file.py
git commit -m "Implement baseline model architecture"

# Use descriptive commit messages
git commit -m "Fix data preprocessing bug in feature extraction"

# Create branches for major features (optional but recommended)
git checkout -b feature/new-model-architecture
# ... make changes ...
git checkout main
git merge feature/new-model-architecture
```

### Code Quality
- **Follow PEP 8** style guidelines for Python
- **Use consistent naming** conventions
- **Keep functions small** and focused
- **Remove unused code** and debug statements

## 📝 Documentation Guidelines

### README Updates
Keep your main README.md updated with:
- Current project status
- Recent achievements
- Instructions for running your code
- Links to important documents

### Code Documentation
```python
def preprocess_data(raw_data, config):
    """
    Preprocess raw data according to specified configuration.
    
    Args:
        raw_data (pd.DataFrame): Raw input data
        config (dict): Preprocessing configuration parameters
        
    Returns:
        pd.DataFrame: Preprocessed data ready for model training
        
    Raises:
        ValueError: If raw_data is empty or invalid
    """
    # Implementation here
    pass
```

### Research Documentation
- **Keep detailed notes** of your literature review
- **Document experiments** with parameters and results
- **Record insights and discoveries** as you work
- **Maintain a research journal** in your documentation folder

## 🤝 Communication with Supervisors

### Regular Communication
- **Weekly progress reports** are mandatory
- **Update milestone issues** regularly
- **Ask questions early** when you're stuck
- **Share intermediate results** for feedback

### Effective Communication
- **Be specific** in your questions and requests
- **Provide context** when reporting problems
- **Include relevant code/data** when seeking help
- **Respond promptly** to supervisor feedback

### Meeting Preparation
Before supervisor meetings:
- **Prepare specific questions** and discussion points
- **Have your work ready** to demonstrate
- **Update your progress** in relevant issues
- **Bring your laptop** with code and results

## 🔬 Research and Experimentation

### Literature Review Phase
1. **Search systematically** using academic databases
2. **Organize papers** in the `literature_review/papers/` folder
3. **Take detailed notes** for each paper
4. **Create summaries** and identify research gaps
5. **Update bibliography** regularly

### Implementation Phase
1. **Start with simple baselines** before complex models
2. **Test incrementally** as you develop
3. **Keep track of experiments** with clear naming
4. **Document parameters** and configurations
5. **Save intermediate results** for comparison

### Experimentation Best Practices
```python
# Example experiment organization
experiments/
├── baseline/
│   ├── simple_baseline.py
│   ├── config_baseline.yaml
│   └── results_baseline.json
├── main_experiments/
│   ├── experiment_01_cnn.py
│   ├── config_01.yaml
│   └── results_01.json
└── ablation/
    ├── ablation_study.py
    ├── config_ablation.yaml
    └── results_ablation.json
```

## 🛠️ Tools and Resources

### Development Tools
- **IDE/Editor:** VS Code, PyCharm, or your preferred editor
- **Version Control:** Git (command line or GUI)
- **Python Environment:** Use virtual environments
- **Jupyter Notebooks:** For exploration and prototyping

### Research Tools
- **Reference Management:** Mendeley, Zotero, or EndNote
- **Paper Search:** Google Scholar, IEEE Xplore, ACM Digital Library
- **Data Analysis:** Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, TensorFlow, PyTorch

### Collaboration Tools
- **GitHub Issues:** Project tracking and communication
- **GitHub Projects:** Kanban boards for organization
- **Email:** For formal communication with supervisors
- **Meetings:** Zoom, Google Meet, or in-person

## 📈 Monitoring Your Progress

### Self-Assessment Questions
Ask yourself regularly:
- Am I meeting my weekly goals?
- Is my work quality up to standards?
- Do I understand what I'm implementing?
- Am I documenting my work adequately?
- Do I need help with anything?

### Progress Indicators
- **Code commits:** Regular, meaningful commits
- **Issue updates:** Active engagement with milestones
- **Documentation:** Growing and improving documentation
- **Results:** Concrete experimental outcomes
- **Learning:** New skills and knowledge acquired

### Warning Signs
Watch out for:
- No commits for several days
- Falling behind on milestones
- Avoiding difficult problems
- Poor code quality or documentation
- Lack of communication with supervisors

## ⚠️ Common Pitfalls and Solutions

### Technical Issues
**Problem:** Code not working or producing errors
- **Solution:** Debug systematically, check documentation, ask for help early

**Problem:** Experimental results not as expected
- **Solution:** Verify implementation, check data, analyze systematically

**Problem:** Running out of computational resources
- **Solution:** Optimize code, use smaller datasets for testing, discuss with supervisors

### Time Management Issues
**Problem:** Falling behind schedule
- **Solution:** Prioritize core objectives, adjust scope if necessary, increase work hours

**Problem:** Spending too much time on minor details
- **Solution:** Focus on major objectives, document issues for future work

**Problem:** Procrastination or lack of motivation
- **Solution:** Break tasks into smaller pieces, celebrate small wins, seek support

### Communication Issues
**Problem:** Unclear requirements or expectations
- **Solution:** Ask specific questions, request examples, clarify in writing

**Problem:** Not receiving timely feedback
- **Solution:** Follow up politely, provide specific deadlines, use multiple communication channels

**Problem:** Difficulty explaining technical concepts
- **Solution:** Practice presentations, use visual aids, start with simple explanations

## 🏆 Success Strategies

### Academic Excellence
- **Stay organized** with clear folder structures and naming
- **Document everything** as you work, not afterward
- **Focus on quality** over quantity in your implementations
- **Learn from failures** and iterate on your approaches

### Research Skills
- **Read broadly** but focus on relevant work
- **Think critically** about existing approaches
- **Be creative** in your problem-solving
- **Stay current** with recent developments in your field

### Professional Development
- **Write clean code** that others can understand
- **Communicate clearly** in writing and presentations
- **Manage your time** effectively with realistic goals
- **Build relationships** with supervisors and peers

## 🆘 Getting Help

### When to Ask for Help
- You've been stuck for more than a few hours
- You're unsure about project requirements
- You need access to resources or tools
- You're behind schedule and need guidance

### How to Ask for Help Effectively
1. **Describe the problem clearly** with specific examples
2. **Explain what you've tried** already
3. **Provide relevant code/data** if applicable
4. **Specify what kind of help** you need
5. **Set realistic expectations** for response time

### Support Resources
- **Primary Supervisor:** Your main point of contact
- **Co-Supervisor:** Additional expertise and perspective
- **Peer Students:** Collaborate and learn together
- **Online Resources:** Documentation, tutorials, forums
- **University Services:** Library, IT support, counseling

## 📅 Important Deadlines

Mark these dates in your calendar:
- **Week 4:** Literature Review Due
- **Week 8:** Methodology and Implementation Due
- **Week 10:** Mid-Term Evaluation
- **Week 16:** Final Submission and Defense

### Submission Checklist
Before each milestone:
- [ ] All deliverables completed according to requirements
- [ ] Code is clean, documented, and tested
- [ ] Documentation is up-to-date and comprehensive
- [ ] Progress is reflected in GitHub issues
- [ ] Supervisors have been notified and materials shared

## 🎯 Final Tips for Success

1. **Start early** and work consistently rather than cramming
2. **Ask questions** when you're uncertain about anything
3. **Document your journey** - it will help with your final paper
4. **Celebrate small wins** to maintain motivation
5. **Stay organized** with clear file structures and naming
6. **Backup your work** regularly (Git helps with this)
7. **Take care of yourself** - manage stress and maintain work-life balance

Remember: This is a learning experience. Mistakes and challenges are normal and expected. The key is to learn from them and keep moving forward. Your supervisors are here to help you succeed!

## 📞 Contact Information

**Your Supervisors:**
- Primary Supervisor: {PRIMARY_SUPERVISOR_EMAIL}
- Co-Supervisor: {CO_SUPERVISOR_EMAIL}

**Technical Support:**
- Repository Issues: Use GitHub Issues for project-related questions
- IT Support: {IT_SUPPORT_EMAIL}

**Administrative:**
- Course Coordinator: {COORDINATOR_EMAIL}
- Department Office: {DEPARTMENT_EMAIL}