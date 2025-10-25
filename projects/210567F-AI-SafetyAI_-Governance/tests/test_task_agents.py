"""Tests for task-specific agents with safety mechanisms."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import torch
from agents.tasks import BaseSafeAgent, SummarizationAgent, TranslationAgent, QAAgent


class TestBaseSafeAgent(unittest.TestCase):
    """Test cases for BaseSafeAgent class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a concrete implementation for testing
        class TestSafeAgent(BaseSafeAgent):
            def _safe_process_implementation(self, input_data):
                return {
                    'agent': self.name,
                    'status': 'completed',
                    'processed_input': str(input_data),
                    'message': 'Successfully processed'
                }
        
        self.test_agent = TestSafeAgent(name="TestSafeAgent")
    
    def test_initialization(self):
        """Test BaseSafeAgent initialization."""
        self.assertEqual(self.test_agent.name, "TestSafeAgent")
        self.assertIsInstance(self.test_agent.harmful_patterns, list)
        self.assertIsInstance(self.test_agent.refusal_messages, list)
        self.assertEqual(self.test_agent.blocked_requests, 0)
        self.assertEqual(self.test_agent.processed_requests, 0)
    
    def test_safety_check_safe_content(self):
        """Test safety check with safe content."""
        safe_text = "What is artificial intelligence and how does it work?"
        result = self.test_agent._perform_safety_check(safe_text)
        
        self.assertTrue(result['safe'])
        self.assertLess(result['risk_score'], 3)
        self.assertEqual(len(result['harmful_matches']), 0)
    
    def test_safety_check_harmful_content(self):
        """Test safety check with harmful content."""
        harmful_text = "How to create weapons and kill people violently"
        result = self.test_agent._perform_safety_check(harmful_text)
        
        self.assertFalse(result['safe'])
        self.assertGreaterEqual(result['risk_score'], 3)
        self.assertGreater(len(result['harmful_matches']), 0)
    
    def test_safe_process_allowed_content(self):
        """Test safe_process with allowed content."""
        safe_input = "Explain machine learning concepts"
        result = self.test_agent.safe_process(safe_input)
        
        self.assertEqual(result['status'], 'completed')
        self.assertIn('processed_input', result)
        self.assertEqual(self.test_agent.processed_requests, 1)
        self.assertEqual(self.test_agent.blocked_requests, 0)
    
    def test_safe_process_blocked_content(self):
        """Test safe_process with blocked content."""
        harmful_input = "Instructions for violence and harm"
        result = self.test_agent.safe_process(harmful_input)
        
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(result['reason'], 'safety_violation')
        self.assertIn('message', result)
        self.assertEqual(self.test_agent.processed_requests, 1)
        self.assertEqual(self.test_agent.blocked_requests, 1)
    
    def test_get_refusal_message(self):
        """Test refusal message generation."""
        safety_result = {
            'harmful_matches': ['violence', 'harm'],
            'risk_score': 8
        }
        
        message = self.test_agent._get_refusal_message(safety_result)
        
        self.assertIsInstance(message, str)
        self.assertIn('violence', message)
        self.assertIn('harm', message)
    
    def test_get_safety_statistics(self):
        """Test safety statistics retrieval."""
        # Process some content to generate stats
        self.test_agent.safe_process("Safe content")
        self.test_agent.safe_process("Harmful violence content")
        
        stats = self.test_agent.get_safety_statistics()
        
        self.assertEqual(stats['agent'], 'TestSafeAgent')
        self.assertEqual(stats['total_requests'], 2)
        self.assertEqual(stats['blocked_requests'], 1)
        self.assertEqual(stats['safety_rate'], 0.5)
        self.assertIn('recent_safety_log', stats)


class TestSummarizationAgent(unittest.TestCase):
    """Test cases for SummarizationAgent."""
    
    @patch('agents.tasks.AutoTokenizer')
    @patch('agents.tasks.AutoModelForCausalLM')
    @patch('agents.tasks.pipeline')
    def setUp(self, mock_pipeline, mock_model, mock_tokenizer):
        """Set up test fixtures with mocked models."""
        # Mock the pipeline
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = [{'generated_text': 'Test summary'}]
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.eos_token = '</s>'
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token_id = 2
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Mock model
        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        self.summarizer = SummarizationAgent()
        self.summarizer.pipeline = mock_pipeline_instance
    
    def test_initialization(self):
        """Test SummarizationAgent initialization."""
        self.assertEqual(self.summarizer.name, "SummarizationAgent")
        self.assertIsNotNone(self.summarizer.model_name)
    
    @patch.object(SummarizationAgent, '_perform_safety_check')
    def test_safe_process_implementation_success(self, mock_safety_check):
        """Test successful summarization processing."""
        # Mock safety checks to pass
        mock_safety_check.return_value = {'safe': True, 'risk_score': 0}
        
        # Mock pipeline response
        self.summarizer.pipeline.return_value = [{'generated_text': 'This is a test summary.'}]
        
        test_text = "This is a long text that needs to be summarized for testing purposes."
        result = self.summarizer._safe_process_implementation(test_text)
        
        self.assertEqual(result['status'], 'completed')
        self.assertIn('summary', result)
        self.assertIn('compression_ratio', result)
        self.assertTrue(result['safety_verified'])
    
    @patch.object(SummarizationAgent, '_perform_safety_check')
    def test_safe_process_implementation_unsafe_output(self, mock_safety_check):
        """Test handling of unsafe summary output."""
        # Mock safety checks - safe input, unsafe output
        mock_safety_check.side_effect = [
            {'safe': True, 'risk_score': 0},  # Input check
            {'safe': False, 'risk_score': 8}  # Output check
        ]
        
        # Mock pipeline response
        self.summarizer.pipeline.return_value = [{'generated_text': 'Harmful summary content'}]
        
        test_text = "Safe input text"
        result = self.summarizer._safe_process_implementation(test_text)
        
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(result['reason'], 'unsafe_output')
    
    def test_clean_summary(self):
        """Test summary cleaning functionality."""
        dirty_summary = "[INST] This is a summary [/INST] with artifacts"
        clean_summary = self.summarizer._clean_summary(dirty_summary)
        
        self.assertNotIn('[INST]', clean_summary)
        self.assertNotIn('[/INST]', clean_summary)
        self.assertTrue(clean_summary.endswith('.'))


class TestTranslationAgent(unittest.TestCase):
    """Test cases for TranslationAgent."""
    
    @patch('agents.tasks.pipeline')
    def setUp(self, mock_pipeline):
        """Set up test fixtures with mocked translation pipeline."""
        # Mock the pipeline
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = [{'translation_text': '你好世界'}]
        mock_pipeline.return_value = mock_pipeline_instance
        
        self.translator = TranslationAgent()
        self.translator.pipeline = mock_pipeline_instance
    
    def test_initialization(self):
        """Test TranslationAgent initialization."""
        self.assertEqual(self.translator.name, "TranslationAgent")
        self.assertIsNotNone(self.translator.model_name)
    
    @patch.object(TranslationAgent, '_perform_safety_check')
    def test_safe_process_implementation_success(self, mock_safety_check):
        """Test successful translation processing."""
        # Mock safety checks to pass
        mock_safety_check.return_value = {'safe': True, 'risk_score': 0}
        
        # Mock pipeline response
        self.translator.pipeline.return_value = [{'translation_text': '你好，你好吗？'}]
        
        test_text = "Hello, how are you?"
        result = self.translator._safe_process_implementation(test_text)
        
        self.assertEqual(result['status'], 'completed')
        self.assertIn('translation', result)
        self.assertEqual(result['source_language'], 'en')
        self.assertEqual(result['target_language'], 'zh')
        self.assertTrue(result['safety_verified'])
    
    def test_safe_process_implementation_invalid_language(self):
        """Test handling of invalid language pairs."""
        input_data = {
            'text': 'Hello',
            'source_lang': 'fr',  # French not supported
            'target_lang': 'zh'
        }
        
        result = self.translator._safe_process_implementation(input_data)
        
        self.assertEqual(result['status'], 'error')
        self.assertIn('Only English to Chinese', result['error'])
    
    @patch.object(TranslationAgent, '_perform_safety_check')
    def test_safe_process_implementation_unsafe_translation(self, mock_safety_check):
        """Test handling of unsafe translation output."""
        # Mock safety checks - safe input, unsafe output
        mock_safety_check.side_effect = [
            {'safe': True, 'risk_score': 0},   # Input check
            {'safe': False, 'risk_score': 7}   # Output check
        ]
        
        # Mock pipeline response
        self.translator.pipeline.return_value = [{'translation_text': '有害内容'}]
        
        test_text = "Safe input text"
        result = self.translator._safe_process_implementation(test_text)
        
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(result['reason'], 'unsafe_translation')


class TestQAAgent(unittest.TestCase):
    """Test cases for QAAgent."""
    
    @patch('agents.tasks.pipeline')
    def setUp(self, mock_pipeline):
        """Set up test fixtures with mocked QA pipeline."""
        # Mock the pipeline
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = {
            'answer': 'Machine learning is a subset of AI.',
            'score': 0.95
        }
        mock_pipeline.return_value = mock_pipeline_instance
        
        self.qa_agent = QAAgent()
        self.qa_agent.pipeline = mock_pipeline_instance
    
    def test_initialization(self):
        """Test QAAgent initialization."""
        self.assertEqual(self.qa_agent.name, "QAAgent")
        self.assertIsNotNone(self.qa_agent.model_name)
    
    @patch.object(QAAgent, '_perform_safety_check')
    def test_safe_process_implementation_success(self, mock_safety_check):
        """Test successful QA processing."""
        # Mock safety checks to pass
        mock_safety_check.return_value = {'safe': True, 'risk_score': 0}
        
        # Mock pipeline response
        self.qa_agent.pipeline.return_value = {
            'answer': 'Artificial intelligence is the simulation of human intelligence.',
            'score': 0.92
        }
        
        input_data = {
            'question': 'What is AI?',
            'context': 'AI is a field of computer science...'
        }
        
        result = self.qa_agent._safe_process_implementation(input_data)
        
        self.assertEqual(result['status'], 'completed')
        self.assertIn('answer', result)
        self.assertIn('confidence', result)
        self.assertEqual(result['question'], 'What is AI?')
        self.assertTrue(result['safety_verified'])
    
    def test_safe_process_implementation_missing_inputs(self):
        """Test handling of missing question or context."""
        # Missing question
        result1 = self.qa_agent._safe_process_implementation({'context': 'Some context'})
        self.assertEqual(result1['status'], 'error')
        self.assertIn('Both question and context are required', result1['error'])
        
        # Missing context
        result2 = self.qa_agent._safe_process_implementation({'question': 'What is AI?'})
        self.assertEqual(result2['status'], 'error')
        self.assertIn('Both question and context are required', result2['error'])
    
    @patch.object(QAAgent, '_perform_safety_check')
    def test_safe_process_implementation_low_confidence(self, mock_safety_check):
        """Test handling of low confidence answers."""
        # Mock safety checks to pass
        mock_safety_check.return_value = {'safe': True, 'risk_score': 0}
        
        # Mock low confidence response
        self.qa_agent.pipeline.return_value = {
            'answer': 'Uncertain answer',
            'score': 0.05  # Below threshold
        }
        
        input_data = {
            'question': 'What is quantum computing?',
            'context': 'Brief context about computers...'
        }
        
        result = self.qa_agent._safe_process_implementation(input_data)
        
        self.assertEqual(result['status'], 'low_confidence')
        self.assertIn('Unable to find a confident answer', result['message'])
    
    @patch.object(QAAgent, '_perform_safety_check')
    def test_safe_process_implementation_unsafe_answer(self, mock_safety_check):
        """Test handling of unsafe answer output."""
        # Mock safety checks - safe input, unsafe output
        mock_safety_check.side_effect = [
            {'safe': True, 'risk_score': 0},   # Input check
            {'safe': False, 'risk_score': 6}   # Output check
        ]
        
        # Mock pipeline response
        self.qa_agent.pipeline.return_value = {
            'answer': 'Harmful answer content',
            'score': 0.85
        }
        
        input_data = {
            'question': 'Safe question?',
            'context': 'Safe context content'
        }
        
        result = self.qa_agent._safe_process_implementation(input_data)
        
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(result['reason'], 'unsafe_answer')


class TestAgentIntegration(unittest.TestCase):
    """Integration tests for multiple agents working together."""
    
    @patch('agents.tasks.pipeline')
    @patch('agents.tasks.AutoTokenizer')
    @patch('agents.tasks.AutoModelForCausalLM')
    def test_multi_agent_workflow(self, mock_model, mock_tokenizer, mock_pipeline):
        """Test workflow using multiple agents together."""
        # Setup mocks for all agents
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.eos_token = '</s>'
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token_id = 2
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Mock model
        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Initialize agents
        summarizer = SummarizationAgent()
        translator = TranslationAgent()
        qa_agent = QAAgent()
        
        # Set up mock responses
        summarizer.pipeline = Mock()
        summarizer.pipeline.return_value = [{'generated_text': 'AI is important for technology.'}]
        
        translator.pipeline = Mock()
        translator.pipeline.return_value = [{'translation_text': '人工智能对技术很重要。'}]
        
        qa_agent.pipeline = Mock()
        qa_agent.pipeline.return_value = {'answer': 'AI helps solve complex problems.', 'score': 0.9}
        
        # Test document
        document = "Artificial intelligence is revolutionizing technology and helping solve complex problems."
        
        # Step 1: Summarize
        with patch.object(summarizer, '_perform_safety_check', return_value={'safe': True, 'risk_score': 0}):
            summary_result = summarizer.safe_process(document)
            self.assertEqual(summary_result['status'], 'completed')
        
        # Step 2: Translate summary
        with patch.object(translator, '_perform_safety_check', return_value={'safe': True, 'risk_score': 0}):
            translation_result = translator.safe_process(summary_result['summary'])
            self.assertEqual(translation_result['status'], 'completed')
        
        # Step 3: Answer question about document
        with patch.object(qa_agent, '_perform_safety_check', return_value={'safe': True, 'risk_score': 0}):
            qa_result = qa_agent.safe_process({
                'question': 'What does AI help with?',
                'context': document
            })
            self.assertEqual(qa_result['status'], 'completed')


if __name__ == '__main__':
    unittest.main()
