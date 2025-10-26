# Student Usage Instructions

## Overview
This document provides instructions for students participating in the Human-AI Collaboration AI Assistants research project, focusing on the HuggingGPT with Dynamic Majority implementation.

## Getting Started

### 1. Prerequisites
- Python 3.8 or higher installed
- OpenRouter API key (get from https://openrouter.ai/keys)
- OpenAI library version 1.0.0 or higher
- Access to the project repository
- Basic understanding of command-line operations and Jupyter notebooks

### 2. Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd 210708P-Human-AI-Collab_AI-Assistants

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install openai>=1.0.0
pip install jupyter
pip install -r requirements.txt
```

### 3. API Configuration
Set your OpenRouter API key as an environment variable:

**Windows:**
```bash
set OPENROUTER_API_KEY=your_api_key_here
```

**macOS/Linux:**
```bash
export OPENROUTER_API_KEY=your_api_key_here
```

Or pass it directly when initializing the planner (see usage examples below).

## HuggingGPT Task Planner

### Overview
The HuggingGPT Task Planner implements the Task Planning Stage from the HuggingGPT paper (NeurIPS 2023). It uses OpenRouter to access OpenAI models at lower cost.

### Supported Task Types
The system supports 24 different task types across multiple domains:

**NLP Tasks:**
- text-classification, token-classification, text2text-generation
- summarization, translation, question-answering
- conversational, text-generation, tabular-classification

**Computer Vision Tasks:**
- image-to-text, text-to-image, visual-question-answering
- image-segmentation, document-question-answering, image-classification
- image-to-image, object-detection, controlnet-sd

**Audio Tasks:**
- text-to-speech, audio-classification
- automatic-speech-recognition, audio-to-audio

**Video Tasks:**
- text-to-video, video-classification

### Basic Usage

#### 1. Initialize the Task Planner
```python
from HuggingGPTTaskPlanner import HuggingGPTTaskPlanner

# Initialize with OpenRouter API key
planner = HuggingGPTTaskPlanner(
    openrouter_api_key="your_api_key_here",
    model="openai/gpt-3.5-turbo"  # Default model from paper
)

# Alternative: Use GPT-4 for better results
planner = HuggingGPTTaskPlanner(
    openrouter_api_key="your_api_key_here",
    model="openai/gpt-4"  # More accurate, matches paper's best results
)
```

#### 2. Parse User Input into Tasks
```python
# Simple task
user_input = "Can you tell me how many objects are in e1.jpg?"
tasks = planner.parse_tasks(user_input)

# The output will be a list of task dictionaries:
# [{"task": "object-detection", "id": 0, "dep": [-1], "args": {"image": "e1.jpg"}}]
```

#### 3. Visualize Task Dependencies
```python
# Print task graph
print(planner.visualize_task_graph(tasks))

# Get execution order (tasks grouped by parallel execution batches)
execution_order = planner.get_execution_order(tasks)
for batch_idx, batch in enumerate(execution_order):
    print(f"Batch {batch_idx + 1}: Tasks {batch}")
```

### Example Use Cases

#### Example 1: Object Detection
```python
tasks = planner.parse_tasks("Can you tell me how many objects are in photo.jpg?")
# Output: [{"task": "object-detection", "id": 0, "dep": [-1], "args": {"image": "photo.jpg"}}]
```

#### Example 2: Multi-Task Request
```python
tasks = planner.parse_tasks("In animal.jpg, what's the animal and what's it doing?")
# Output: Multiple tasks including image-to-text, image-classification, 
# object-detection, and visual-question-answering
```

#### Example 3: Sequential Tasks with Dependencies
```python
user_input = """First generate a HED image of input.jpg, 
then based on the HED image and a text 'a girl reading a book', 
create a new image as a response."""

tasks = planner.parse_tasks(user_input)
# Task 1 generates HED image (dep: [-1])
# Task 2 uses output from Task 1 (dep: [0]) with "<resource>-0" reference
```

#### Example 4: Text-to-Image Generation
```python
tasks = planner.parse_tasks("Generate an image of a cat sitting on a chair")
# Output: [{"task": "text-to-image", "id": 0, "dep": [-1], 
#          "args": {"text": "a cat sitting on a chair"}}]
```

### Understanding Task Structure

Each task has the following structure:
```python
{
    "task": str,        # Task type (e.g., "object-detection")
    "id": int,          # Unique task identifier
    "dep": List[int],   # Dependencies ([-1] means no dependencies)
    "args": {           # Task arguments
        "text": str,    # Optional text input
        "image": str,   # Optional image path/URL
        "audio": str,   # Optional audio path/URL
        "video": str    # Optional video path/URL
    }
}
```

**Dependency System:**
- `dep: [-1]` - No dependencies, can execute immediately
- `dep: [0]` - Depends on task 0
- `dep: [0, 1]` - Depends on both task 0 and task 1
- `"<resource>-0"` - References output from task with id 0

## Running the Jupyter Notebook

### Launch Jupyter
```bash
jupyter notebook experiments/HuggingGpt_with_Dyn_maj.ipynb
```

### Notebook Structure
The notebook contains:
1. **Setup cells** - Import libraries and initialize planner
2. **Example demonstrations** - Test cases from the paper
3. **Custom experiments** - Areas to test your own inputs
4. **Analysis cells** - Visualize and analyze results

### Working with the Notebook

#### Cell 1: Imports and Setup
Run this first to import the HuggingGPTTaskPlanner class.

#### Cell 2: Initialize Planner
```python
planner = HuggingGPTTaskPlanner(
    openrouter_api_key="your_api_key",
    model="openai/gpt-3.5-turbo"
)
```

#### Cell 3-7: Run Test Cases
Execute the provided test cases to see how task planning works.

#### Cell 8+: Your Experiments
Add your own test cases and analyze the results.

## Project Structure
```
210708P-Human-AI-Collab_AI-Assistants/
├── experiments/
│   ├── HuggingGpt_with_Dyn_maj.ipynb    # Main notebook
│   └── task_planner.py                   # Core implementation
├── data/
│   ├── input/                            # Input images/files
│   └── output/                           # Generated results
├── docs/
│   ├── usage_instructions.md             # This file
│   └── references.md                     # Paper references
└── results/                              # Experiment outputs
```

## Troubleshooting

### Common Issues

**Issue: OpenAI library import error**
```
ImportError: OpenAI library not installed
```
- Solution: `pip install openai>=1.0.0`

**Issue: API key not found**
```
ValueError: OpenRouter API key required
```
- Solution: Set `OPENROUTER_API_KEY` environment variable or pass directly to constructor

**Issue: No tasks parsed from input**
```
⚠ No tasks parsed from response
```
- Solution: Check that your input clearly describes a task. Review demonstration examples.
- Try rephrasing your request to be more explicit about what you want

**Issue: JSON parsing errors**
- Solution: The parser uses robust extraction. If issues persist, check LLM response format
- Enable debug output to see raw LLM responses

**Issue: Circular dependency detected**
```
Warning: Circular dependency detected in task graph
```
- Solution: Review your task dependencies. No task should depend on itself or create a loop.

### Debug Mode
To see detailed parsing information:
```python
# The planner automatically prints debug info
tasks = planner.parse_tasks(user_input)
# Will show: "✓ Successfully parsed X tasks" or error details
```

## Model Selection Guide

### GPT-3.5-turbo (Default)
- **Cost:** ~$0.0015 per 1K tokens
- **Speed:** Fast
- **Use for:** Basic task planning, development, testing

### GPT-4
- **Cost:** ~$0.03 per 1K tokens  
- **Speed:** Slower
- **Use for:** Complex multi-task requests, production, final experiments

### GPT-4-turbo
- **Cost:** ~$0.01 per 1K tokens
- **Speed:** Medium
- **Use for:** Balance between accuracy and cost

## Best Practices

### 1. Input Formulation
- Be specific about file names (e.g., "image1.jpg" not "the image")
- Clearly state sequential steps if needed
- Reference the demonstration examples for formatting

### 2. Task Dependencies
- Understand that `dep: [-1]` means independent tasks
- Tasks with dependencies execute after their parent tasks complete
- Use `<resource>-X` to reference outputs from task X

### 3. Multi-Turn Conversations
- The planner maintains chat logs automatically
- It can reference previously mentioned resources
- Keep conversations focused on related tasks

### 4. Resource Management
- Place input files in `data/input/` directory
- Use consistent naming conventions
- Check that file paths in task args are correct

## Experimental Guidelines

### For Research Experiments

1. **Baseline Comparison**
   - Run same inputs with GPT-3.5 and GPT-4
   - Record task parsing accuracy
   - Note differences in task decomposition

2. **Complex Task Analysis**
   - Test multi-step workflows
   - Analyze dependency graph complexity
   - Measure execution order optimization

3. **Edge Cases**
   - Ambiguous requests
   - Requests outside supported task types
   - Requests with implicit dependencies

### Data Collection
```python
# Store results for analysis
results = {
    "input": user_input,
    "tasks": tasks,
    "model": planner.model,
    "execution_order": planner.get_execution_order(tasks)
}

# Save to JSON
import json
with open('results/experiment_01.json', 'w') as f:
    json.dump(results, f, indent=2)
```

## Submission Guidelines

### Required Components
1. Completed Jupyter notebook with your experiments
2. `results/` folder with experiment outputs
3. Summary report documenting:
   - Test cases used
   - Model(s) tested
   - Task parsing accuracy
   - Interesting findings
4. Any modifications to the base code (document in comments)

### Code Documentation
- Add comments explaining your experimental setup
- Document any parameter changes
- Include rationale for model selection

### Report Format
Your summary report should include:
- **Introduction:** What you tested and why
- **Methodology:** Models used, test cases, evaluation criteria
- **Results:** Task parsing accuracy, dependency graph analysis
- **Discussion:** Insights, limitations, future work
- **Conclusion:** Key findings

## Additional Resources

- **HuggingGPT Paper:** [arXiv:2303.17580](https://arxiv.org/abs/2303.17580)
- **OpenRouter Docs:** [openrouter.ai/docs](https://openrouter.ai/docs)
- **OpenAI API Reference:** [platform.openai.com/docs](https://platform.openai.com/docs)
- **Project Repository:** Contact supervisor for access

## Support

- Check demonstration examples in the code
- Review the original HuggingGPT paper (Section 3.1, Table 1)
- Contact project supervisor for technical assistance
- Refer to `docs/references.md` for research papers

## Tips for Success

1. **Start Simple:** Begin with single-task requests before complex workflows
2. **Study Demonstrations:** The 6 demonstration examples cover common patterns
3. **Iterate:** If parsing fails, rephrase your input based on working examples
4. **Visualize:** Always check the task graph to understand dependencies
5. **Document:** Keep notes on what works and what doesn't for your report
