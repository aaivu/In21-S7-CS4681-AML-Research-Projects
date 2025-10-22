# SafetyAI - AI Safety Analysis System

A multi-agent AI safety analysis system powered by **Google Gemini 2.5 Pro Flash**. Analyzes text content for safety concerns, harmful patterns, and provides comprehensive risk assessment.

## ✨ Features

- 🤖 **Multi-Agent System**: Specialized AI agents for safety and analysis
- 🛡️ **Safety Evaluation**: Real-time safety scoring and risk assessment  
- 📊 **Pattern Analysis**: Detailed content analysis and insights
- 🌐 **Web Interface**: User-friendly web dashboard
- ⚡ **Google Gemini**: Powered by latest Gemini 2.5 Pro Flash model

## 📸 Screenshots

### Web Interface
![Application Screenshot 1](images/Application%20Screenshot%201.png)

### Safety Analysis Dashboard
![Application Screenshot 2](images/Application%20Screenshot%202.png)

### Multi-Agent System Interface
![Application Screenshot 3](images/Application%20Screenshot%203.png)

## 📁 Project Structure

```
SafetyAI/
├── agents/           # AI agent implementations
├── utils/            # Configuration and utilities
├── tests/            # Test suite
├── templates/        # Web interface templates
├── data/             # Sample datasets
├── docs/             # Documentation
├── main.py           # Main application
├── app.py            # Web application
└── requirements.txt  # Dependencies
```

## 🚀 Quick Start

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your Gemini API key
```

### 2. Get Your API Key
1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Create a new API key
3. Add it to your `.env` file:
   ```
   GEMINI_2_5_PRO_FLASH_API_KEY=your_api_key_here
   ```

### 3. Run the Application

**Web Interface (Recommended)**
```bash
python main.py --web
# Open http://localhost:8000
```

**Interactive Mode**
```bash
python main.py --interactive
```

**Analyze Text**
```bash
python main.py --text "Your text to analyze"
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Test system functionality
python main.py --test
```

## 📊 Usage Examples

**Safety Analysis**
```python
from agents.safety_agent import SafetyAgent
from utils.config import Config

config = Config()
agent = SafetyAgent(api_key=config.gemini_api_key)

result = agent.process("Your text here")
print(f"Safety Score: {result['safety_score']}")
```

**Multi-Agent Analysis**
```python
from agents.multi_agent_system import MultiAgentSystem
from utils.config import Config

config = Config()
system = MultiAgentSystem(config)
system.initialize_agents()

result = system.process_text("Your text here")
print(f"Safety Analysis: {result['safety_analysis']}")
```

## 🛠️ Configuration

Edit `.env` file:
```bash
GEMINI_2_5_PRO_FLASH_API_KEY=your_api_key_here
HUGGINGFACE_API_KEY=your_hf_key_here 
SAFETY_ALIGN_LOG_LEVEL=INFO
```

## 🤖 Available Agents

- **SafetyAgent**: Harmful content detection and risk scoring
- **AnalysisAgent**: Linguistic pattern analysis and insights
- **CoordinatorAgent**: Multi-agent workflow orchestration
- **OverseerAgent**: Query routing with safety scanning

## 🌐 Web Interface Features

- Real-time safety analysis
- Interactive text processing
- Safety scoring dashboard
- Batch text analysis
- Multi-agent coordination



