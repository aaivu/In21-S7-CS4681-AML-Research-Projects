"""Tests for the OverseerAgent and related components."""

import unittest
from unittest.mock import Mock, patch, MagicMock
from agents.overseer import OverseerAgent, TaskRoutingTool
from agents.task_agents import SummarizationAgent, TranslationAgent, QAAgent
from agents.safety_tools import SafetyScanTool, ContentFilterTool
from utils.config import Config


class TestTaskRoutingTool(unittest.TestCase):
    """Test cases for TaskRoutingTool."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock agents
        self.mock_summarization = Mock()
        self.mock_translation = Mock()
        self.mock_qa = Mock()
        
        self.task_agents = {
            'summarization': self.mock_summarization,
            'translation': self.mock_translation,
            'question_answering': self.mock_qa
        }
        
        self.routing_tool = TaskRoutingTool(task_agents=self.task_agents)
    
    def test_classify_query_summarization(self):
        """Test query classification for summarization."""
        queries = [
            "Please summarize this text",
            "Give me a summary of the document",
            "Can you provide a tldr?",
            "I need a brief overview"
        ]
        
        for query in queries:
            task_type = self.routing_tool._classify_query(query)
            self.assertEqual(task_type, 'summarization')
    
    def test_classify_query_translation(self):
        """Test query classification for translation."""
        queries = [
            "Translate this to Spanish",
            "Convert this text to French",
            "I need a translation"
        ]
        
        for query in queries:
            task_type = self.routing_tool._classify_query(query)
            self.assertEqual(task_type, 'translation')
    
    def test_classify_query_qa(self):
        """Test query classification for question answering."""
        queries = [
            "What is the capital of France?",
            "How does photosynthesis work?",
            "Why is the sky blue?",
            "When did World War II end?"
        ]
        
        for query in queries:
            task_type = self.routing_tool._classify_query(query)
            self.assertEqual(task_type, 'question_answering')
    
    def test_run_successful_routing(self):
        """Test successful query routing."""
        # Setup mock response
        self.mock_summarization.process.return_value = {
            'status': 'completed',
            'task_type': 'summarization',
            'summary': 'Test summary'
        }
        
        # Test routing
        result = self.routing_tool._run("Summarize this text", "summarization")
        
        # Verify
        self.assertIn("Summary: Test summary", result)
        self.mock_summarization.process.assert_called_once_with("Summarize this text")
    
    def test_run_agent_error(self):
        """Test handling of agent errors."""
        # Setup mock error response
        self.mock_qa.process.return_value = {
            'status': 'error',
            'error': 'Test error message'
        }
        
        # Test routing
        result = self.routing_tool._run("What is AI?", "question_answering")
        
        # Verify error handling
        self.assertIn("Error processing question_answering request", result)
        self.assertIn("Test error message", result)
    
    def test_run_unsupported_task(self):
        """Test handling of unsupported task types."""
        result = self.routing_tool._run("Test query", "unsupported_task")
        self.assertIn("Unsupported task type: unsupported_task", result)


class TestSafetyScanTool(unittest.TestCase):
    """Test cases for SafetyScanTool."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.safety_tool = SafetyScanTool()
    
    def test_get_safety_prompts(self):
        """Test safety prompt retrieval."""
        prompts = self.safety_tool._get_safety_prompts()
        
        self.assertIn('violence', prompts)
        self.assertIn('exploitation', prompts)
        self.assertIn('hate_speech', prompts)
        self.assertIn('comprehensive', prompts)
    
    def test_parse_safety_assessment_safe(self):
        """Test parsing of safe content assessment."""
        assessment = """
        Assessment: Does this text promote violence? No
        Risk Level: Low
        Safe to process: Yes
        """
        
        result = self.safety_tool._parse_safety_assessment(assessment, "test text")
        
        self.assertTrue(result['safe'])
        self.assertEqual(result['risk_level'], 'Low')
    
    def test_parse_safety_assessment_unsafe(self):
        """Test parsing of unsafe content assessment."""
        assessment = """
        Assessment: Does this text promote violence? Yes
        Risk Level: High
        Contains harmful content: Yes
        """
        
        result = self.safety_tool._parse_safety_assessment(assessment, "test text")
        
        self.assertFalse(result['safe'])
        self.assertEqual(result['risk_level'], 'High')
    
    def test_format_safety_result(self):
        """Test safety result formatting."""
        result_data = {
            'safe': True,
            'risk_level': 'Low',
            'concerns': ['No concerns'],
            'explanation': 'Content is safe',
            'method': 'test'
        }
        
        formatted = self.safety_tool._format_safety_result(result_data)
        
        self.assertIn("Safe to process: True", formatted)
        self.assertIn("Risk Level: Low", formatted)
        self.assertIn("Content is safe", formatted)


class TestOverseerAgent(unittest.TestCase):
    """Test cases for OverseerAgent."""
    
    @patch('agents.overseer.create_openai_functions_agent')
    @patch('agents.overseer.AgentExecutor')
    @patch('agents.base_agent.ChatOpenAI')
    def setUp(self, mock_chat_openai, mock_agent_executor, mock_create_agent):
        """Set up test fixtures."""
        # Mock configuration
        self.config = Config()
        self.config.openai_api_key = "test_key"
        
        # Mock agent executor
        self.mock_executor = Mock()
        mock_agent_executor.return_value = self.mock_executor
        
        # Create overseer agent
        self.overseer = OverseerAgent(config=self.config)
    
    def test_initialization(self):
        """Test OverseerAgent initialization."""
        self.assertEqual(self.overseer.name, "OverseerAgent")
        self.assertIsNotNone(self.overseer.task_agents)
        self.assertIsNotNone(self.overseer.safety_tools)
        self.assertIsNotNone(self.overseer.routing_tool)
        self.assertEqual(self.overseer.processed_requests, 0)
        self.assertEqual(self.overseer.blocked_requests, 0)
    
    def test_initialize_task_agents(self):
        """Test task agent initialization."""
        agents = self.overseer.task_agents
        
        self.assertIn('summarization', agents)
        self.assertIn('translation', agents)
        self.assertIn('question_answering', agents)
    
    @patch.object(OverseerAgent, '_perform_safety_scan')
    def test_process_safe_query(self, mock_safety_scan):
        """Test processing of safe query."""
        # Mock safety scan to return safe
        mock_safety_scan.return_value = {
            'safe': True,
            'risk_level': 'Low',
            'scan_result': 'Content is safe'
        }
        
        # Mock executor response
        self.mock_executor.invoke.return_value = {
            'output': 'Safe response to query'
        }
        
        # Process query
        result = self.overseer._process_query_safely("What is AI?")
        
        # Verify
        self.assertEqual(result['status'], 'completed')
        self.assertIn('response', result)
        self.assertIn('safety_checks', result)
    
    @patch.object(OverseerAgent, '_perform_safety_scan')
    def test_process_unsafe_query(self, mock_safety_scan):
        """Test processing of unsafe query."""
        # Mock safety scan to return unsafe
        mock_safety_scan.return_value = {
            'safe': False,
            'risk_level': 'High',
            'scan_result': 'Content contains harmful elements'
        }
        
        # Process query
        result = self.overseer._process_query_safely("Harmful query content")
        
        # Verify
        self.assertEqual(result['status'], 'blocked')
        self.assertTrue(result['safety_violation'])
        self.assertEqual(self.overseer.blocked_requests, 1)
    
    @patch.object(OverseerAgent, '_perform_safety_scan')
    def test_process_unsafe_output(self, mock_safety_scan):
        """Test handling of unsafe output."""
        # Mock safety scan - safe input, unsafe output
        mock_safety_scan.side_effect = [
            {'safe': True, 'risk_level': 'Low', 'scan_result': 'Input is safe'},
            {'safe': False, 'risk_level': 'High', 'scan_result': 'Output is unsafe'}
        ]
        
        # Mock executor response
        self.mock_executor.invoke.return_value = {
            'output': 'Potentially harmful response'
        }
        
        # Process query
        result = self.overseer._process_query_safely("Normal query")
        
        # Verify
        self.assertEqual(result['status'], 'output_blocked')
        self.assertTrue(result['safety_violation'])
    
    def test_fallback_safety_check(self):
        """Test fallback safety checking."""
        with patch.object(self.overseer, 'generate_response') as mock_generate:
            mock_generate.return_value = "SAFE"
            
            result = self.overseer._fallback_safety_check("Test content")
            
            self.assertTrue(result['safe'])
            self.assertEqual(result['method'], 'fallback_prompt')
    
    def test_get_safety_statistics(self):
        """Test safety statistics retrieval."""
        # Simulate some activity
        self.overseer.processed_requests = 10
        self.overseer.blocked_requests = 2
        
        stats = self.overseer.get_safety_statistics()
        
        self.assertEqual(stats['total_requests'], 10)
        self.assertEqual(stats['blocked_requests'], 2)
        self.assertEqual(stats['safety_rate'], 0.8)
    
    def test_get_agent_status(self):
        """Test agent status retrieval."""
        status = self.overseer.get_agent_status()
        
        self.assertEqual(status['agent'], 'OverseerAgent')
        self.assertIn('available_task_agents', status)
        self.assertIn('safety_tools_count', status)
        self.assertIn('safety_statistics', status)
    
    def test_reset_safety_log(self):
        """Test safety log reset."""
        # Add some entries
        self.overseer.safety_log.append({"test": "entry"})
        self.overseer.blocked_requests = 5
        self.overseer.processed_requests = 10
        
        # Reset
        self.overseer.reset_safety_log()
        
        # Verify reset
        self.assertEqual(len(self.overseer.safety_log), 0)
        self.assertEqual(self.overseer.blocked_requests, 0)
        self.assertEqual(self.overseer.processed_requests, 0)
    
    def test_process_string_input(self):
        """Test processing string input."""
        with patch.object(self.overseer, '_process_query_safely') as mock_process:
            mock_process.return_value = {"test": "result"}
            
            result = self.overseer.process("Test query")
            mock_process.assert_called_once_with("Test query", None)
    
    def test_process_dict_input(self):
        """Test processing dictionary input."""
        with patch.object(self.overseer, '_process_query_safely') as mock_process:
            mock_process.return_value = {"test": "result"}
            
            input_data = {"query": "Test query", "task_type": "summarization"}
            result = self.overseer.process(input_data)
            mock_process.assert_called_once_with("Test query", "summarization")
    
    def test_export_safety_log(self):
        """Test safety log export."""
        import tempfile
        import json
        import os
        
        # Add test data
        self.overseer.safety_log.append({"test": "entry", "timestamp": "2023-01-01"})
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            self.overseer.export_safety_log(temp_path)
            
            # Verify export
            with open(temp_path, 'r') as f:
                exported_data = json.load(f)
            
            self.assertEqual(len(exported_data), 1)
            self.assertEqual(exported_data[0]["test"], "entry")
            
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main()
