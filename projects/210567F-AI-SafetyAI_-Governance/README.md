# SafetyAlignNLP - Multi-Agent Safety Analysis System

A sophisticated multi-agent system built with LangChain for analyzing adversarial text data and identifying safety concerns. The system employs specialized agents for safety evaluation, pattern analysis, and task coordination, with optional NeMo Guardrails integration for enhanced content filtering.

## 🚀 Features

- **Multi-Agent Architecture**: Specialized agents for different analysis tasks
- **OverseerAgent**: Advanced query routing with safety scanning using LangChain AgentExecutor
- **Safety Evaluation**: Comprehensive safety scoring and risk assessment
- **Pattern Analysis**: Detailed linguistic and rhetorical analysis
- **Task Coordination**: Intelligent orchestration of multi-agent workflows
- **Guardrails Integration**: NeMo Guardrails for real-time content filtering
- **Task-Specific Agents**: Dedicated agents for summarization, translation, and QA
- **Safety Tools**: Custom LangChain tools for content scanning and filtering
- **Batch Processing**: Efficient processing of large datasets
- **Interactive Mode**: Command-line interface for real-time analysis
- **Comprehensive Testing**: Full test suite with unit tests

## 🏗️ Architecture

```
SafetyAlignNLP/
├── agents/                 # Multi-agent system components
│   ├── base_agent.py      # Abstract base agent class
│   ├── safety_agent.py    # Safety evaluation specialist
│   ├── analysis_agent.py  # Pattern analysis specialist
│   ├── coordinator_agent.py # Task coordination and orchestration
│   ├── overseer.py        # OverseerAgent with query routing & safety
│   ├── task_agents.py     # Task-specific agents (summarization, translation, QA)
│   ├── safety_tools.py    # LangChain safety scanning tools
│   └── multi_agent_system.py # Main system orchestrator
├── utils/                  # Utility modules
│   ├── data_loader.py     # JSON data loading utilities
│   └── config.py          # Configuration management
├── tests/                  # Test suite
│   ├── test_data_loader.py
│   ├── test_agents.py
│   ├── test_overseer.py   # OverseerAgent tests
│   └── test_multi_agent_system.py
├── data/                   # Adversarial text datasets
│   ├── beaver_first1000-qa.json
│   ├── beaver_last1000-qa.json
│   └── beaver_random1000-qa.json
├── main.py                 # Main entry point with OverseerAgent support
├── example_overseer.py     # OverseerAgent usage examples
└── requirements.txt        # Dependencies
```

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd SafetyAlignNLP
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export HUGGINGFACE_API_KEY="your-huggingface-api-key"  # Optional
   ```

## 🚀 Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd SafetyAlignNLP

# Install dependencies
pip install -r requirements.txt

# Set up environment variables (optional)
cp .env.example .env
# Edit .env with your API keys

# Run system tests
python main.py --test

# Start web interface
python main.py --web

# Run interactive mode
python main.py --interactive

# Analyze a single text with TSDI safety
python main.py --text "Your text here"

# Process query through OverseerAgent
python main.py --overseer --query "What is machine learning?"

# Analyze a dataset
python main.py --dataset beaver_first1000-qa --sample-size 50
```

## 🌐 Web Interface

The system includes a comprehensive web interface for safe query processing:

```bash
# Start web interface
python main.py --web

# Custom host and port
python main.py --web --host 0.0.0.0 --port 8080

# Access at http://localhost:8000
```

**Features:**
- Interactive query processing with TSDI safety
- Real-time harm scoring and bias detection
- Configurable safety thresholds
- Safety statistics dashboard
- Task-specific agent selection

## 🧪 System Testing

Comprehensive test suite with JSON data loading and attack simulations:

```bash
# Run all system tests
python main.py --test

# Specific test modules
python -m pytest tests/test_system.py -v
python -m pytest tests/test_safety.py -v
```

**Test Coverage:**
- JSON data loading from `data/` folder
- Agent chaining attacks (summarization → translation → QA)
- Process rate measurements
- Harm retention analysis
- Prompt injection resistance
- Load testing scenarios

## 🔧 Configuration

The system uses a configuration class that can be customized:

```python
from utils.config import Config

config = Config()
config.safety_agent.temperature = 0.5
config.guardrails.enabled = True
```

### Environment Variables

- `OPENAI_API_KEY`: Required for LLM functionality
- `HUGGINGFACE_API_KEY`: Optional for HuggingFace models
- `SAFETY_ALIGN_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

## 🤖 Agent Types

### OverseerAgent (New!)
Advanced query router with safety scanning:
- Real-time safety content filtering using NeMo Guardrails
- Intelligent query routing to task-specific agents
- LangChain AgentExecutor integration
- Comprehensive safety logging and audit trails
- Refuses harmful content with detailed explanations

### Task-Specific Agents
Specialized agents for different tasks:
- **SummarizationAgent**: Creates concise, accurate summaries
- **TranslationAgent**: Handles multi-language translation
- **QAAgent**: Provides detailed question answering

### Safety Agent
Specializes in:
- Harmful content detection
- Toxicity assessment
- Risk scoring (0-10 scale)
- Safety recommendations

### Analysis Agent
Focuses on:
- Linguistic pattern analysis
- Rhetorical technique identification
- Bias detection
- Readability assessment

### Coordinator Agent
Manages:
- Multi-agent workflows
- Result synthesis
- Task prioritization
- System orchestration

## 📊 Usage Examples

### OverseerAgent with Safety Scanning
```python
from agents.overseer import OverseerAgent
from utils.config import Config

# Initialize OverseerAgent
config = Config()
overseer = OverseerAgent(config)

# Process query with safety scanning
result = overseer.process("What is machine learning?")
if result.get('status') == 'completed':
    print(result['response'])
elif result.get('status') == 'blocked':
    print(f"Blocked: {result['response']}")

# Task-specific routing
result = overseer.process({
    "query": "Summarize the benefits of AI",
    "task_type": "summarization"
})
```

### Safety Statistics and Monitoring
```python
# Get safety statistics
stats = overseer.get_safety_statistics()
print(f"Safety rate: {stats['safety_rate']:.2%}")
print(f"Blocked requests: {stats['blocked_requests']}")

# Export safety log for audit
overseer.export_safety_log("safety_audit.json")
```

### Multi-Agent System Analysis
```python
from agents.multi_agent_system import MultiAgentSystem
from utils.config import Config

# Initialize system
config = Config()
system = MultiAgentSystem(config)
system.initialize_agents()

# Analyze text
result = system.process_text(
    "Your text here", 
    task_type="safety_analysis"
)
print(result)
```

### Dataset Evaluation
```python
# Evaluate a dataset
result = system.process_dataset(
    dataset_name="beaver_first1000-qa",
    sample_size=100
)
print(result["evaluation_summary"])
```

## 🛡️ Guardrails Integration

The system supports NeMo Guardrails for additional content filtering:

```python
config = Config()
config.guardrails.enabled = True
config.guardrails.config_path = "path/to/guardrails/config"
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_agents.py

# Run with coverage
python -m pytest tests/ --cov=agents --cov=utils
```

## 📈 Performance Monitoring

The system tracks:
- Token usage per agent
- Processing times
- Error rates
- Task completion statistics

Access monitoring data:
```python
# Get system status
status = system.get_system_status()
print(status)

# Get token usage
usage = system.get_agent_token_usage()
print(usage)
```

## 🔍 Command Line Interface

### Available Commands (Interactive Mode)

#### OverseerAgent Mode
- `query <text>` - Process query through overseer with safety scanning
- `summarize <text>` - Summarize text
- `translate <text> to <lang>` - Translate text to target language
- `ask <question>` - Ask a question
- `safety <text>` - Check text safety only
- `safety-stats` - Show safety statistics
- `status` - Show system status
- `reset` - Reset safety log
- `help` - Show help message
- `quit` - Exit the program

#### Multi-Agent Mode
- `analyze <text>` - Analyze a text string
- `dataset <name> [size]` - Analyze a dataset
- `list` - List available datasets
- `status` - Show system status
- `usage` - Show token usage statistics
- `reset` - Reset conversation history
- `help` - Show help message
- `quit` - Exit the program

### Command Line Arguments

```bash
python main.py [OPTIONS]

Options:
  -i, --interactive          Run in interactive mode
  --overseer                 Use OverseerAgent for query routing with safety scanning
  -q, --query TEXT          Process a query through OverseerAgent
  -t, --text TEXT           Analyze a single text string
  -d, --dataset NAME        Analyze a dataset by name (Multi-Agent mode only)
  -b, --batch-file FILE     Process texts from JSON file (Multi-Agent mode only)
  -s, --sample-size N       Sample size for dataset analysis
  --task-type TYPE          Type of analysis (safety_analysis, pattern_analysis, comprehensive_analysis)
  -o, --output FILE         Output file for results (JSON)
  --log-level LEVEL         Logging level (DEBUG, INFO, WARNING, ERROR)
  --no-guardrails           Disable NeMo Guardrails
  --config FILE             Path to configuration file
  --test                    Run comprehensive system tests
  --web                     Start web interface
  --host HOST               Host for web interface (default: 127.0.0.1)
  --port PORT               Port for web interface (default: 8000)
  -h, --help               Show help message
```

## 📝 Data Format

The system expects JSON files with the following structure:

```json
{
  "outputs": [
    "First adversarial text example",
    "Second adversarial text example",
    "Third adversarial text example"
  ]
}
```

## 🚨 Error Handling

The system includes comprehensive error handling:
- Graceful degradation when agents are unavailable
- Detailed error logging
- Retry mechanisms for transient failures
- Input validation and sanitization

## 🔒 Security Considerations

- API keys are loaded from environment variables
- Input sanitization through guardrails
- Output filtering for harmful content
- Secure handling of sensitive data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com/) for LLM orchestration
- Uses [HuggingFace](https://huggingface.co/) for model access
- Integrates [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) for safety
- Powered by [OpenAI](https://openai.com/) language models

## 📞 Support

For questions or issues:
1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed information
4. Contact the development team

---

**Note**: This system is designed for research and educational purposes. Always review and validate results before making decisions based on the analysis.
