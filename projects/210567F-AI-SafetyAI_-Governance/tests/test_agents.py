"""Tests for the agent classes."""

import unittest
from unittest.mock import Mock, patch, MagicMock
from agents.base_agent import BaseAgent
from agents.safety_agent import SafetyAgent
from agents.analysis_agent import AnalysisAgent
from agents.coordinator_agent import CoordinatorAgent


class TestBaseAgent(unittest.TestCase):
    """Test cases for BaseAgent class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a concrete implementation of BaseAgent for testing
        class TestAgent(BaseAgent):
            def process(self, input_data):
                return {"test": "result"}
        
        self.test_agent = TestAgent(
            name="test_agent",
            model_name="gpt-3.5-turbo",
            temperature=0.5,
            system_prompt="Test prompt"
        )
    
    @patch('agents.base_agent.ChatOpenAI')
    def test_initialization(self, mock_chat_openai):
        """Test BaseAgent initialization."""
        self.assertEqual(self.test_agent.name, "test_agent")
        self.assertEqual(self.test_agent.model_name, "gpt-3.5-turbo")
        self.assertEqual(self.test_agent.temperature, 0.5)
        self.assertEqual(self.test_agent.system_prompt, "Test prompt")
        self.assertEqual(len(self.test_agent.conversation_history), 0)
    
    def test_prepare_messages(self):
        """Test message preparation."""
        messages = self.test_agent._prepare_messages("Test input")
        self.assertEqual(len(messages), 2)  # System prompt + user input
        self.assertEqual(messages[0].content, "Test prompt")
        self.assertEqual(messages[1].content, "Test input")
    
    def test_update_conversation_history(self):
        """Test conversation history update."""
        self.test_agent._update_conversation_history("User input", "Agent response")
        self.assertEqual(len(self.test_agent.conversation_history), 2)
    
    def test_get_token_usage(self):
        """Test token usage retrieval."""
        usage = self.test_agent.get_token_usage()
        self.assertIn("total_tokens", usage)
        self.assertIn("prompt_tokens", usage)
        self.assertIn("completion_tokens", usage)
    
    def test_reset_conversation(self):
        """Test conversation reset."""
        self.test_agent._update_conversation_history("Test", "Response")
        self.assertEqual(len(self.test_agent.conversation_history), 2)
        
        self.test_agent.reset_conversation()
        self.assertEqual(len(self.test_agent.conversation_history), 0)
    
    def test_get_info(self):
        """Test agent info retrieval."""
        info = self.test_agent.get_info()
        self.assertEqual(info["name"], "test_agent")
        self.assertEqual(info["model_name"], "gpt-3.5-turbo")
        self.assertEqual(info["temperature"], 0.5)


class TestSafetyAgent(unittest.TestCase):
    """Test cases for SafetyAgent class."""
    
    @patch('agents.base_agent.ChatOpenAI')
    def setUp(self, mock_chat_openai):
        """Set up test fixtures."""
        self.safety_agent = SafetyAgent()
    
    def test_initialization(self):
        """Test SafetyAgent initialization."""
        self.assertEqual(self.safety_agent.name, "SafetyAgent")
        self.assertIn("safety evaluation", self.safety_agent.system_prompt.lower())
    
    @patch.object(SafetyAgent, 'generate_response')
    def test_analyze_single_text(self, mock_generate):
        """Test single text analysis."""
        mock_generate.return_value = "Safety Score: 7\nHigh risk content detected"
        
        result = self.safety_agent._analyze_single_text("Test harmful text")
        
        self.assertEqual(result["agent"], "SafetyAgent")
        self.assertEqual(result["status"], "completed")
        self.assertIn("analysis", result)
        mock_generate.assert_called_once()
    
    def test_extract_safety_score(self):
        """Test safety score extraction."""
        response_with_score = "Safety Score: 8.5\nOther content"
        score = self.safety_agent._extract_safety_score(response_with_score)
        self.assertEqual(score, 8.5)
        
        response_without_score = "No score in this response"
        score = self.safety_agent._extract_safety_score(response_without_score)
        self.assertIsNone(score)
    
    def test_process_string_input(self):
        """Test processing string input."""
        with patch.object(self.safety_agent, '_analyze_single_text') as mock_analyze:
            mock_analyze.return_value = {"test": "result"}
            
            result = self.safety_agent.process("Test text")
            mock_analyze.assert_called_once_with("Test text")
    
    def test_process_dict_input(self):
        """Test processing dictionary input."""
        with patch.object(self.safety_agent, '_analyze_single_text') as mock_analyze:
            mock_analyze.return_value = {"test": "result"}
            
            result = self.safety_agent.process({"text": "Test text"})
            mock_analyze.assert_called_once_with("Test text")
    
    def test_process_list_input(self):
        """Test processing list input."""
        with patch.object(self.safety_agent, '_process_batch') as mock_batch:
            mock_batch.return_value = {"batch": "result"}
            
            result = self.safety_agent.process(["Text 1", "Text 2"])
            mock_batch.assert_called_once_with(["Text 1", "Text 2"])


class TestAnalysisAgent(unittest.TestCase):
    """Test cases for AnalysisAgent class."""
    
    @patch('agents.base_agent.ChatOpenAI')
    def setUp(self, mock_chat_openai):
        """Set up test fixtures."""
        self.analysis_agent = AnalysisAgent()
    
    def test_initialization(self):
        """Test AnalysisAgent initialization."""
        self.assertEqual(self.analysis_agent.name, "AnalysisAgent")
        self.assertIn("analysis agent", self.analysis_agent.system_prompt.lower())
    
    def test_calculate_basic_metrics(self):
        """Test basic metrics calculation."""
        text = "This is a test sentence. This is another sentence."
        metrics = self.analysis_agent._calculate_basic_metrics(text)
        
        self.assertIn("character_count", metrics)
        self.assertIn("word_count", metrics)
        self.assertIn("sentence_count", metrics)
        self.assertEqual(metrics["character_count"], len(text))
        self.assertEqual(metrics["word_count"], 9)  # 9 words
        self.assertEqual(metrics["sentence_count"], 2)  # 2 sentences
    
    def test_count_syllables(self):
        """Test syllable counting."""
        self.assertEqual(self.analysis_agent._count_syllables("hello"), 2)
        self.assertEqual(self.analysis_agent._count_syllables("cat"), 1)
        self.assertEqual(self.analysis_agent._count_syllables("beautiful"), 3)
    
    def test_extract_effectiveness_rating(self):
        """Test effectiveness rating extraction."""
        response_with_rating = "Effectiveness Rating: 9\nOther content"
        rating = self.analysis_agent._extract_effectiveness_rating(response_with_rating)
        self.assertEqual(rating, 9.0)
        
        response_without_rating = "No rating in this response"
        rating = self.analysis_agent._extract_effectiveness_rating(response_without_rating)
        self.assertIsNone(rating)


class TestCoordinatorAgent(unittest.TestCase):
    """Test cases for CoordinatorAgent class."""
    
    @patch('agents.base_agent.ChatOpenAI')
    def setUp(self, mock_chat_openai):
        """Set up test fixtures."""
        self.coordinator = CoordinatorAgent()
        
        # Create mock agents
        self.mock_safety_agent = Mock()
        self.mock_analysis_agent = Mock()
        
        # Register mock agents
        self.coordinator.register_agent('safety_agent', self.mock_safety_agent)
        self.coordinator.register_agent('analysis_agent', self.mock_analysis_agent)
    
    def test_initialization(self):
        """Test CoordinatorAgent initialization."""
        self.assertEqual(self.coordinator.name, "CoordinatorAgent")
        self.assertIn("coordinator", self.coordinator.system_prompt.lower())
    
    def test_register_agent(self):
        """Test agent registration."""
        self.assertIn('safety_agent', self.coordinator.available_agents)
        self.assertIn('analysis_agent', self.coordinator.available_agents)
    
    def test_get_available_agents(self):
        """Test getting available agents."""
        agents = self.coordinator.get_available_agents()
        self.assertIn('safety_agent', agents)
        self.assertIn('analysis_agent', agents)
    
    @patch.object(CoordinatorAgent, 'generate_response')
    def test_coordinate_safety_analysis(self, mock_generate):
        """Test safety analysis coordination."""
        # Setup mock responses
        self.mock_safety_agent.process.return_value = {"safety": "result"}
        mock_generate.return_value = "Coordination summary"
        
        task_data = {"task_type": "safety_analysis", "text": "Test text"}
        result = self.coordinator._handle_structured_task(task_data)
        
        self.assertEqual(result["task_type"], "safety_analysis")
        self.assertIn("safety_results", result)
        self.assertIn("coordination_summary", result)
        self.mock_safety_agent.process.assert_called_once_with("Test text")
    
    @patch.object(CoordinatorAgent, 'generate_response')
    def test_coordinate_comprehensive_analysis(self, mock_generate):
        """Test comprehensive analysis coordination."""
        # Setup mock responses
        self.mock_safety_agent.process.return_value = {"safety": "result"}
        self.mock_analysis_agent.process.return_value = {"analysis": "result"}
        mock_generate.return_value = "Synthesis summary"
        
        result = self.coordinator._coordinate_comprehensive_analysis("Test text")
        
        self.assertEqual(result["task_type"], "comprehensive_analysis")
        self.assertIn("individual_results", result)
        self.assertIn("synthesis", result)
        self.assertEqual(len(result["agents_involved"]), 2)
    
    def test_get_task_history(self):
        """Test task history retrieval."""
        # Simulate a completed task
        self.coordinator._log_task({
            "task_type": "test_task",
            "timestamp": "2023-01-01T00:00:00",
            "agents_involved": ["test_agent"]
        })
        
        history = self.coordinator.get_task_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["task_type"], "test_task")
    
    def test_get_system_status(self):
        """Test system status retrieval."""
        status = self.coordinator.get_system_status()
        
        self.assertEqual(status["coordinator"], "CoordinatorAgent")
        self.assertIn("available_agents", status)
        self.assertIn("total_tasks_completed", status)
        self.assertIn("agent_status", status)


if __name__ == '__main__':
    unittest.main()
