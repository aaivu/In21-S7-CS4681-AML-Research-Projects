"""Tests for TSDI safety module."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from utils.safety import (
    BiasVector, HarmScorer, TSDAISafetyLayer, SafetyWrapper,
    get_global_safety_layer, wrap_agent_with_safety
)


class TestBiasVector(unittest.TestCase):
    """Test cases for BiasVector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.bias_vector = BiasVector()
    
    def test_initialization(self):
        """Test BiasVector initialization."""
        self.assertIsInstance(self.bias_vector.gender_bias_terms, dict)
        self.assertIsInstance(self.bias_vector.racial_bias_terms, dict)
        self.assertIsInstance(self.bias_vector.religious_bias_terms, dict)
        self.assertIn('male', self.bias_vector.gender_bias_terms)
        self.assertIn('female', self.bias_vector.gender_bias_terms)
    
    @patch('utils.safety.SentenceTransformer')
    def test_initialize_embeddings(self, mock_sentence_transformer):
        """Test embedding model initialization."""
        # Mock the sentence transformer
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(5, 384)  # Mock embeddings
        mock_sentence_transformer.return_value = mock_model
        
        self.bias_vector.initialize_embeddings()
        
        self.assertIsNotNone(self.bias_vector.embedding_model)
        mock_sentence_transformer.assert_called_once()
    
    @patch('utils.safety.SentenceTransformer')
    def test_get_bias_score(self, mock_sentence_transformer):
        """Test bias score calculation."""
        # Mock the sentence transformer
        mock_model = Mock()
        mock_model.encode.return_value = [np.random.rand(384)]  # Mock text embedding
        mock_sentence_transformer.return_value = mock_model
        
        # Initialize with mock
        self.bias_vector.initialize_embeddings()
        self.bias_vector.bias_vectors['gender'] = np.random.rand(384)  # Mock bias vector
        
        score = self.bias_vector.get_bias_score("This is a test text", "gender")
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)
    
    def test_get_bias_score_no_model(self):
        """Test bias score calculation without initialized model."""
        score = self.bias_vector.get_bias_score("Test text", "gender")
        self.assertEqual(score, 0.0)


class TestHarmScorer(unittest.TestCase):
    """Test cases for HarmScorer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.harm_scorer = HarmScorer()
    
    def test_initialization(self):
        """Test HarmScorer initialization."""
        self.assertIsInstance(self.harm_scorer.harm_categories, dict)
        self.assertIsInstance(self.harm_scorer.violence_patterns, list)
        self.assertIsInstance(self.harm_scorer.harassment_patterns, list)
    
    @patch('utils.safety.pipeline')
    def test_initialize_models(self, mock_pipeline):
        """Test harm detection model initialization."""
        # Mock pipeline creation
        mock_pipeline.return_value = Mock()
        
        self.harm_scorer.initialize_models()
        
        # Should attempt to create pipelines
        self.assertTrue(mock_pipeline.called)
    
    def test_calculate_harm_score_safe_text(self):
        """Test harm score calculation for safe text."""
        safe_text = "This is a normal, safe text about technology and science."
        scores = self.harm_scorer.calculate_harm_score(safe_text)
        
        self.assertIsInstance(scores, dict)
        self.assertIn('overall', scores)
        self.assertIn('toxicity', scores)
        self.assertIn('violence', scores)
        self.assertLessEqual(scores['overall'], 0.5)  # Should be low for safe text
    
    def test_calculate_harm_score_harmful_text(self):
        """Test harm score calculation for harmful text."""
        harmful_text = "I want to kill and destroy everything with violence and weapons."
        scores = self.harm_scorer.calculate_harm_score(harmful_text)
        
        self.assertIsInstance(scores, dict)
        self.assertIn('overall', scores)
        self.assertIn('violence', scores)
        self.assertGreater(scores['violence'], 0.0)  # Should detect violence
        self.assertGreater(scores['overall'], 0.0)  # Overall should be elevated
    
    def test_get_pattern_score(self):
        """Test pattern-based scoring."""
        text_with_violence = "This text contains kill and murder words."
        score = self.harm_scorer._get_pattern_score(text_with_violence, self.harm_scorer.violence_patterns)
        
        self.assertGreater(score, 0.0)  # Should detect violent patterns
        self.assertLessEqual(score, 1.0)  # Should not exceed 1.0
    
    def test_get_simple_bias_score(self):
        """Test simple bias scoring."""
        biased_text = "All men are better than women at everything."
        score = self.harm_scorer._get_simple_bias_score(biased_text)
        
        self.assertGreater(score, 0.0)  # Should detect bias
        self.assertLessEqual(score, 1.0)  # Should not exceed 1.0


class TestTSDAISafetyLayer(unittest.TestCase):
    """Test cases for TSDAISafetyLayer class."""
    
    @patch('utils.safety.BiasVector')
    @patch('utils.safety.HarmScorer')
    def setUp(self, mock_harm_scorer, mock_bias_vector):
        """Set up test fixtures."""
        # Mock the components
        self.mock_bias_vector = Mock()
        self.mock_harm_scorer = Mock()
        
        mock_bias_vector.return_value = self.mock_bias_vector
        mock_harm_scorer.return_value = self.mock_harm_scorer
        
        self.safety_layer = TSDAISafetyLayer(harm_threshold=0.2)
    
    def test_initialization(self):
        """Test TSDAISafetyLayer initialization."""
        self.assertEqual(self.safety_layer.harm_threshold, 0.2)
        self.assertEqual(self.safety_layer.total_processed, 0)
        self.assertEqual(self.safety_layer.total_blocked, 0)
        self.assertEqual(self.safety_layer.bias_corrections, 0)
    
    def test_preprocess_input_safe(self):
        """Test input preprocessing with safe content."""
        # Mock harm scorer to return safe scores
        self.mock_harm_scorer.calculate_harm_score.return_value = {
            'overall': 0.1,  # Below threshold
            'toxicity': 0.05,
            'violence': 0.0
        }
        
        result = self.safety_layer.preprocess_input("This is safe text")
        
        self.assertEqual(result['status'], 'safe')
        self.assertIn('processed_text', result)
        self.assertEqual(self.safety_layer.total_processed, 1)
        self.assertEqual(self.safety_layer.total_blocked, 0)
    
    def test_preprocess_input_harmful(self):
        """Test input preprocessing with harmful content."""
        # Mock harm scorer to return high scores
        self.mock_harm_scorer.calculate_harm_score.return_value = {
            'overall': 0.8,  # Above threshold
            'toxicity': 0.7,
            'violence': 0.6
        }
        
        result = self.safety_layer.preprocess_input("This is harmful text")
        
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(result['reason'], 'harmful_input')
        self.assertEqual(self.safety_layer.total_processed, 1)
        self.assertEqual(self.safety_layer.total_blocked, 1)
    
    def test_postprocess_output_safe(self):
        """Test output postprocessing with safe content."""
        # Mock harm scorer to return safe scores
        self.mock_harm_scorer.calculate_harm_score.return_value = {
            'overall': 0.1,
            'toxicity': 0.05,
            'violence': 0.0
        }
        
        result = self.safety_layer.postprocess_output("This is safe output")
        
        self.assertEqual(result['status'], 'safe')
        self.assertIn('harm_scores', result)
    
    def test_postprocess_output_harmful(self):
        """Test output postprocessing with harmful content."""
        # Mock harm scorer to return high scores
        self.mock_harm_scorer.calculate_harm_score.return_value = {
            'overall': 0.9,
            'toxicity': 0.8,
            'violence': 0.7
        }
        
        result = self.safety_layer.postprocess_output("This is harmful output")
        
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(result['reason'], 'harmful_output')
    
    def test_subtract_bias(self):
        """Test bias subtraction functionality."""
        # Mock bias vector to return high bias score
        self.mock_bias_vector.embedding_model = Mock()  # Simulate initialized model
        self.mock_bias_vector.bias_vectors = {'gender': Mock()}
        self.mock_bias_vector.get_bias_score.return_value = 0.5  # High bias
        
        text = "He is a great programmer"
        processed_text, bias_info = self.safety_layer._subtract_bias(text)
        
        self.assertIsInstance(processed_text, str)
        self.assertIsInstance(bias_info, dict)
        self.assertIn('bias_detected', bias_info)
        self.assertIn('bias_scores', bias_info)
    
    def test_apply_bias_correction(self):
        """Test bias correction application."""
        text = "He is a good man and she is a nice woman"
        corrected = self.safety_layer._apply_bias_correction(text, 'gender', 0.5)
        
        # Should replace gendered terms with neutral ones
        self.assertNotEqual(text, corrected)
        self.assertIn('they', corrected.lower())
    
    def test_get_safety_statistics(self):
        """Test safety statistics retrieval."""
        # Process some content to generate stats
        self.safety_layer.total_processed = 10
        self.safety_layer.total_blocked = 2
        self.safety_layer.bias_corrections = 3
        
        stats = self.safety_layer.get_safety_statistics()
        
        self.assertEqual(stats['total_processed'], 10)
        self.assertEqual(stats['total_blocked'], 2)
        self.assertEqual(stats['bias_corrections'], 3)
        self.assertEqual(stats['safety_rate'], 0.8)  # (10-2)/10
        self.assertEqual(stats['harm_threshold'], 0.2)
    
    def test_update_threshold(self):
        """Test harm threshold update."""
        old_threshold = self.safety_layer.harm_threshold
        new_threshold = 0.5
        
        self.safety_layer.update_threshold(new_threshold)
        
        self.assertEqual(self.safety_layer.harm_threshold, new_threshold)
        self.assertNotEqual(self.safety_layer.harm_threshold, old_threshold)
    
    def test_reset_statistics(self):
        """Test statistics reset."""
        # Set some values
        self.safety_layer.total_processed = 10
        self.safety_layer.total_blocked = 2
        self.safety_layer.bias_corrections = 3
        self.safety_layer.safety_log = [{'test': 'event'}]
        
        self.safety_layer.reset_statistics()
        
        self.assertEqual(self.safety_layer.total_processed, 0)
        self.assertEqual(self.safety_layer.total_blocked, 0)
        self.assertEqual(self.safety_layer.bias_corrections, 0)
        self.assertEqual(len(self.safety_layer.safety_log), 0)


class TestSafetyWrapper(unittest.TestCase):
    """Test cases for SafetyWrapper class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock agent
        self.mock_agent = Mock()
        self.mock_agent.name = "TestAgent"
        self.mock_agent.process.return_value = {
            'status': 'completed',
            'response': 'Test response'
        }
        
        # Create mock safety layer
        self.mock_safety_layer = Mock()
        self.mock_safety_layer.preprocess_input.return_value = {
            'status': 'safe',
            'processed_text': 'processed text'
        }
        self.mock_safety_layer.postprocess_output.return_value = {
            'status': 'safe',
            'harm_scores': {'overall': 0.1}
        }
        
        self.wrapper = SafetyWrapper(self.mock_agent, self.mock_safety_layer)
    
    def test_initialization(self):
        """Test SafetyWrapper initialization."""
        self.assertEqual(self.wrapper.agent, self.mock_agent)
        self.assertEqual(self.wrapper.safety_layer, self.mock_safety_layer)
    
    def test_attribute_delegation(self):
        """Test attribute delegation to wrapped agent."""
        # Should delegate attribute access to the wrapped agent
        self.assertEqual(self.wrapper.name, "TestAgent")
    
    def test_safe_execute_success(self):
        """Test successful safe execution."""
        result = self.wrapper._safe_execute('process', 'test input')
        
        # Should call preprocess and postprocess
        self.mock_safety_layer.preprocess_input.assert_called_once()
        self.mock_safety_layer.postprocess_output.assert_called_once()
        
        # Should call agent method
        self.mock_agent.process.assert_called_once()
        
        # Should return successful result
        self.assertIn('safety_verified', result)
    
    def test_safe_execute_blocked_input(self):
        """Test safe execution with blocked input."""
        # Mock safety layer to block input
        self.mock_safety_layer.preprocess_input.return_value = {
            'status': 'blocked',
            'reason': 'harmful_input',
            'message': 'Input blocked',
            'timestamp': '2023-01-01T00:00:00'
        }
        
        result = self.wrapper._safe_execute('process', 'harmful input')
        
        # Should return blocked result
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(result['reason'], 'harmful_input')
        
        # Should not call agent method
        self.mock_agent.process.assert_not_called()
    
    def test_safe_execute_blocked_output(self):
        """Test safe execution with blocked output."""
        # Mock safety layer to block output
        self.mock_safety_layer.postprocess_output.return_value = {
            'status': 'blocked',
            'reason': 'harmful_output',
            'message': 'Output blocked',
            'timestamp': '2023-01-01T00:00:00'
        }
        
        result = self.wrapper._safe_execute('process', 'test input')
        
        # Should return blocked result
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(result['reason'], 'harmful_output')
    
    def test_extract_text_input(self):
        """Test text input extraction."""
        # Test string input
        text = self.wrapper._extract_text_input('test string')
        self.assertEqual(text, 'test string')
        
        # Test dict input
        text = self.wrapper._extract_text_input({'text': 'test text'})
        self.assertEqual(text, 'test text')
        
        # Test no text input
        text = self.wrapper._extract_text_input(123)
        self.assertIsNone(text)
    
    def test_extract_text_output(self):
        """Test text output extraction."""
        # Test string output
        text = self.wrapper._extract_text_output('test output')
        self.assertEqual(text, 'test output')
        
        # Test dict output
        text = self.wrapper._extract_text_output({'response': 'test response'})
        self.assertEqual(text, 'test response')
        
        # Test no text output
        text = self.wrapper._extract_text_output(123)
        self.assertIsNone(text)


class TestGlobalSafetyFunctions(unittest.TestCase):
    """Test cases for global safety functions."""
    
    @patch('utils.safety._global_safety_layer', None)
    def test_get_global_safety_layer(self):
        """Test global safety layer creation."""
        layer = get_global_safety_layer()
        
        self.assertIsInstance(layer, TSDAISafetyLayer)
        
        # Should return same instance on subsequent calls
        layer2 = get_global_safety_layer()
        self.assertIs(layer, layer2)
    
    def test_wrap_agent_with_safety(self):
        """Test agent wrapping with safety."""
        mock_agent = Mock()
        wrapped = wrap_agent_with_safety(mock_agent)
        
        self.assertIsInstance(wrapped, SafetyWrapper)
        self.assertEqual(wrapped.agent, mock_agent)


if __name__ == '__main__':
    unittest.main()
