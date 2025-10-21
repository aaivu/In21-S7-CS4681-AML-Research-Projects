"""Tests for the multi-agent system."""

import unittest
from unittest.mock import Mock, patch, MagicMock
from agents.multi_agent_system import MultiAgentSystem
from utils.config import Config


class TestMultiAgentSystem(unittest.TestCase):
    """Test cases for MultiAgentSystem class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.config.openai_api_key = "test_key"
        self.system = MultiAgentSystem(self.config)
    
    def test_initialization(self):
        """Test MultiAgentSystem initialization."""
        self.assertIsNotNone(self.system.config)
        self.assertIsNotNone(self.system.data_loader)
        self.assertFalse(self.system.is_initialized)
        self.assertEqual(self.system.system_stats["tasks_completed"], 0)
    
    @patch('agents.multi_agent_system.SafetyAgent')
    @patch('agents.multi_agent_system.AnalysisAgent')
    @patch('agents.multi_agent_system.CoordinatorAgent')
    def test_initialize_agents(self, mock_coordinator, mock_analysis, mock_safety):
        """Test agent initialization."""
        # Setup mocks
        mock_safety_instance = Mock()
        mock_analysis_instance = Mock()
        mock_coordinator_instance = Mock()
        
        mock_safety.return_value = mock_safety_instance
        mock_analysis.return_value = mock_analysis_instance
        mock_coordinator.return_value = mock_coordinator_instance
        
        # Initialize agents
        result = self.system.initialize_agents()
        
        # Verify initialization
        self.assertTrue(result)
        self.assertTrue(self.system.is_initialized)
        self.assertIsNotNone(self.system.safety_agent)
        self.assertIsNotNone(self.system.analysis_agent)
        self.assertIsNotNone(self.system.coordinator_agent)
        
        # Verify agent registration
        mock_coordinator_instance.register_agent.assert_any_call('safety_agent', mock_safety_instance)
        mock_coordinator_instance.register_agent.assert_any_call('analysis_agent', mock_analysis_instance)
    
    def test_process_text_not_initialized(self):
        """Test processing text when system is not initialized."""
        with self.assertRaises(RuntimeError):
            self.system.process_text("Test text")
    
    @patch('agents.multi_agent_system.MultiAgentSystem.initialize_agents')
    def test_process_text_success(self, mock_init):
        """Test successful text processing."""
        # Setup system as initialized
        mock_init.return_value = True
        self.system.is_initialized = True
        
        # Setup mock coordinator
        mock_coordinator = Mock()
        mock_coordinator.process.return_value = {"result": "success", "synthesis": "test synthesis"}
        self.system.coordinator_agent = mock_coordinator
        
        # Process text
        result = self.system.process_text("Test text", "comprehensive_analysis")
        
        # Verify results
        self.assertEqual(result["result"], "success")
        self.assertEqual(self.system.system_stats["tasks_completed"], 1)
        self.assertEqual(self.system.system_stats["total_texts_processed"], 1)
        
        # Verify coordinator was called correctly
        expected_task_data = {
            "task_type": "comprehensive_analysis",
            "text": "Test text"
        }
        mock_coordinator.process.assert_called_once_with(expected_task_data)
    
    @patch('agents.multi_agent_system.MultiAgentSystem.initialize_agents')
    def test_process_text_error(self, mock_init):
        """Test text processing with error."""
        # Setup system as initialized
        mock_init.return_value = True
        self.system.is_initialized = True
        
        # Setup mock coordinator to raise exception
        mock_coordinator = Mock()
        mock_coordinator.process.side_effect = Exception("Test error")
        self.system.coordinator_agent = mock_coordinator
        
        # Process text
        result = self.system.process_text("Test text")
        
        # Verify error handling
        self.assertIn("error", result)
        self.assertEqual(result["status"], "error")
        self.assertEqual(self.system.system_stats["errors_encountered"], 1)
    
    @patch('agents.multi_agent_system.MultiAgentSystem.initialize_agents')
    def test_process_dataset_not_found(self, mock_init):
        """Test processing non-existent dataset."""
        # Setup system as initialized
        mock_init.return_value = True
        self.system.is_initialized = True
        
        # Mock data loader to return empty list
        mock_data_loader = Mock()
        mock_data_loader.list_available_datasets.return_value = []
        self.system.data_loader = mock_data_loader
        
        # Process dataset
        result = self.system.process_dataset("nonexistent_dataset")
        
        # Verify error
        self.assertIn("error", result)
        self.assertIn("not found", result["error"])
    
    @patch('agents.multi_agent_system.MultiAgentSystem.initialize_agents')
    def test_batch_process_texts(self, mock_init):
        """Test batch text processing."""
        # Setup system as initialized
        mock_init.return_value = True
        self.system.is_initialized = True
        
        # Setup mock coordinator
        mock_coordinator = Mock()
        mock_coordinator._coordinate_batch_analysis.return_value = {"batch": "result"}
        self.system.coordinator_agent = mock_coordinator
        
        # Process batch
        texts = ["Text 1", "Text 2", "Text 3"]
        result = self.system.batch_process_texts(texts)
        
        # Verify results
        self.assertEqual(result["batch"], "result")
        self.assertEqual(self.system.system_stats["tasks_completed"], 1)
        self.assertEqual(self.system.system_stats["total_texts_processed"], 3)
        
        # Verify coordinator was called
        mock_coordinator._coordinate_batch_analysis.assert_called_once_with(texts)
    
    def test_get_system_status_not_initialized(self):
        """Test getting system status when not initialized."""
        status = self.system.get_system_status()
        self.assertEqual(status["status"], "not_initialized")
    
    @patch('agents.multi_agent_system.MultiAgentSystem.initialize_agents')
    def test_get_system_status_initialized(self, mock_init):
        """Test getting system status when initialized."""
        # Setup system as initialized
        mock_init.return_value = True
        self.system.is_initialized = True
        
        # Mock data loader
        mock_data_loader = Mock()
        mock_data_loader.list_available_datasets.return_value = ["dataset1", "dataset2"]
        self.system.data_loader = mock_data_loader
        
        # Mock coordinator
        mock_coordinator = Mock()
        mock_coordinator.get_system_status.return_value = {"coordinator": "status"}
        self.system.coordinator_agent = mock_coordinator
        
        # Get status
        status = self.system.get_system_status()
        
        # Verify status
        self.assertTrue(status["system_initialized"])
        self.assertEqual(len(status["available_datasets"]), 2)
        self.assertIn("system_stats", status)
        self.assertIn("coordinator_status", status)
    
    @patch('agents.multi_agent_system.MultiAgentSystem.initialize_agents')
    def test_get_agent_token_usage(self, mock_init):
        """Test getting agent token usage."""
        # Setup system as initialized
        mock_init.return_value = True
        self.system.is_initialized = True
        
        # Setup mock agents
        mock_safety = Mock()
        mock_safety.get_token_usage.return_value = {"total_tokens": 100}
        
        mock_analysis = Mock()
        mock_analysis.get_token_usage.return_value = {"total_tokens": 200}
        
        mock_coordinator = Mock()
        mock_coordinator.get_token_usage.return_value = {"total_tokens": 50}
        
        self.system.safety_agent = mock_safety
        self.system.analysis_agent = mock_analysis
        self.system.coordinator_agent = mock_coordinator
        
        # Get usage
        usage = self.system.get_agent_token_usage()
        
        # Verify usage
        self.assertEqual(usage["safety_agent"]["total_tokens"], 100)
        self.assertEqual(usage["analysis_agent"]["total_tokens"], 200)
        self.assertEqual(usage["coordinator_agent"]["total_tokens"], 50)
    
    @patch('agents.multi_agent_system.MultiAgentSystem.initialize_agents')
    def test_reset_all_conversations(self, mock_init):
        """Test resetting all agent conversations."""
        # Setup system as initialized
        mock_init.return_value = True
        self.system.is_initialized = True
        
        # Setup mock agents
        mock_safety = Mock()
        mock_analysis = Mock()
        mock_coordinator = Mock()
        
        self.system.safety_agent = mock_safety
        self.system.analysis_agent = mock_analysis
        self.system.coordinator_agent = mock_coordinator
        
        # Reset conversations
        self.system.reset_all_conversations()
        
        # Verify reset was called on all agents
        mock_safety.reset_conversation.assert_called_once()
        mock_analysis.reset_conversation.assert_called_once()
        mock_coordinator.reset_conversation.assert_called_once()
    
    def test_shutdown(self):
        """Test system shutdown."""
        # Setup some state
        self.system.is_initialized = True
        self.system.system_stats["tasks_completed"] = 5
        
        # Shutdown
        self.system.shutdown()
        
        # Verify cleanup
        self.assertFalse(self.system.is_initialized)
        self.assertIsNone(self.system.safety_agent)
        self.assertIsNone(self.system.analysis_agent)
        self.assertIsNone(self.system.coordinator_agent)
        self.assertIsNone(self.system.guardrails)
    
    @patch('agents.multi_agent_system.NEMO_AVAILABLE', True)
    @patch('agents.multi_agent_system.RailsConfig')
    @patch('agents.multi_agent_system.LLMRails')
    def test_initialize_guardrails(self, mock_llm_rails, mock_rails_config):
        """Test guardrails initialization."""
        # Setup mocks
        mock_config = Mock()
        mock_rails_config.from_content.return_value = mock_config
        mock_rails_instance = Mock()
        mock_llm_rails.return_value = mock_rails_instance
        
        # Enable guardrails
        self.system.config.guardrails.enabled = True
        
        # Initialize guardrails
        self.system._initialize_guardrails()
        
        # Verify initialization
        mock_rails_config.from_content.assert_called_once()
        mock_llm_rails.assert_called_once_with(mock_config)
        self.assertEqual(self.system.guardrails, mock_rails_instance)
    
    def test_apply_input_guardrails_no_guardrails(self):
        """Test input guardrails when not available."""
        result = self.system._apply_input_guardrails("Test text")
        self.assertEqual(result, "Test text")
    
    def test_apply_output_guardrails_no_guardrails(self):
        """Test output guardrails when not available."""
        result = self.system._apply_output_guardrails("Test text")
        self.assertEqual(result, "Test text")


if __name__ == '__main__':
    unittest.main()
