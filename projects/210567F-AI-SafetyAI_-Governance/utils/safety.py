"""
Token-level Safety-Debiased Inference (TSDI) module for bias reduction and harm scoring.
This module provides comprehensive safety mechanisms including bias subtraction and harm detection.
"""

import logging
import re
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime

# Handle optional dependencies gracefully
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    F = None

try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        pipeline, BertTokenizer, BertModel
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    pipeline = None
    BertTokenizer = None
    BertModel = None

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

logger = logging.getLogger(__name__)


class BiasVector:
    """Represents bias vectors for different demographic groups and attributes."""
    
    def __init__(self):
        """Initialize bias vectors for common demographic attributes."""
        # Demographic bias terms for different groups
        self.gender_bias_terms = {
            'male': ['he', 'him', 'his', 'man', 'men', 'male', 'boy', 'boys', 'father', 'son', 'brother', 'husband'],
            'female': ['she', 'her', 'hers', 'woman', 'women', 'female', 'girl', 'girls', 'mother', 'daughter', 'sister', 'wife']
        }
        
        self.racial_bias_terms = {
            'white': ['white', 'caucasian', 'european'],
            'black': ['black', 'african', 'afro'],
            'asian': ['asian', 'chinese', 'japanese', 'korean', 'indian'],
            'hispanic': ['hispanic', 'latino', 'latina', 'mexican']
        }
        
        self.religious_bias_terms = {
            'christian': ['christian', 'catholic', 'protestant', 'church'],
            'muslim': ['muslim', 'islam', 'islamic', 'mosque'],
            'jewish': ['jewish', 'jew', 'judaism', 'synagogue'],
            'hindu': ['hindu', 'hinduism', 'temple'],
            'buddhist': ['buddhist', 'buddhism']
        }
        
        # Bias direction vectors (will be computed from embeddings)
        self.bias_vectors = {}
        self.embedding_model = None
        
    def initialize_embeddings(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize sentence embedding model for bias vector computation."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("SentenceTransformers not available, using fallback bias detection")
            return
            
        try:
            self.embedding_model = SentenceTransformer(model_name)
            self._compute_bias_vectors()
            logger.info("Bias vectors initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize bias embeddings: {e}")
            self.embedding_model = None
    
    def _compute_bias_vectors(self):
        """Compute bias direction vectors from demographic terms."""
        if not self.embedding_model:
            return
        
        try:
            # Compute gender bias vector
            male_embeddings = self.embedding_model.encode(self.gender_bias_terms['male'])
            female_embeddings = self.embedding_model.encode(self.gender_bias_terms['female'])
            
            male_centroid = np.mean(male_embeddings, axis=0)
            female_centroid = np.mean(female_embeddings, axis=0)
            self.bias_vectors['gender'] = male_centroid - female_centroid
            
            # Compute racial bias vectors
            for race, terms in self.racial_bias_terms.items():
                embeddings = self.embedding_model.encode(terms)
                self.bias_vectors[f'race_{race}'] = np.mean(embeddings, axis=0)
            
            # Compute religious bias vectors
            for religion, terms in self.religious_bias_terms.items():
                embeddings = self.embedding_model.encode(terms)
                self.bias_vectors[f'religion_{religion}'] = np.mean(embeddings, axis=0)
            
            logger.info(f"Computed {len(self.bias_vectors)} bias vectors")
            
        except Exception as e:
            logger.error(f"Error computing bias vectors: {e}")
    
    def get_bias_score(self, text: str, bias_type: str = 'gender') -> float:
        """Calculate bias score for given text and bias type."""
        if not self.embedding_model or bias_type not in self.bias_vectors:
            return 0.0
        
        try:
            text_embedding = self.embedding_model.encode([text])[0]
            bias_vector = self.bias_vectors[bias_type]
            
            # Calculate cosine similarity
            similarity = np.dot(text_embedding, bias_vector) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(bias_vector)
            )
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating bias score: {e}")
            return 0.0


class HarmScorer:
    """Harm scoring system for detecting potentially harmful content."""
    
    def __init__(self):
        """Initialize harm scoring models."""
        self.toxicity_model = None
        self.hate_speech_model = None
        self.bias_model = None
        
        # Harm categories and their weights
        self.harm_categories = {
            'toxicity': 0.3,
            'hate_speech': 0.25,
            'violence': 0.2,
            'harassment': 0.15,
            'bias': 0.1
        }
        
        # Harmful pattern matching
        self.violence_patterns = [
            r'\b(kill|murder|violence|harm|hurt|attack|destroy|weapon|bomb|shoot|stab)\b',
            r'\b(death|die|dead|corpse|blood|gore|torture|abuse)\b'
        ]
        
        self.harassment_patterns = [
            r'\b(hate|discriminat\w+|racist|sexist|homophobic|transphobic)\b',
            r'\b(stupid|idiot|moron|retard|freak|loser|worthless)\b'
        ]
        
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize pre-trained models for harm detection."""
        if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
            logger.warning("Transformers/PyTorch not available, using pattern-based harm detection only")
            return
            
        try:
            # Initialize toxicity detection model
            self.toxicity_model = pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Initialize hate speech detection model
            self.hate_speech_model = pipeline(
                "text-classification", 
                model="cardiffnlp/twitter-roberta-base-hate-latest",
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("Harm scoring models initialized successfully")
            
        except Exception as e:
            logger.warning(f"Some harm detection models failed to load: {e}")
            # Continue with pattern-based detection as fallback
    
    def calculate_harm_score(self, text: str) -> Dict[str, float]:
        """Calculate comprehensive harm score for given text."""
        scores = {}
        
        # Toxicity score
        scores['toxicity'] = self._get_toxicity_score(text)
        
        # Hate speech score
        scores['hate_speech'] = self._get_hate_speech_score(text)
        
        # Violence score (pattern-based)
        scores['violence'] = self._get_pattern_score(text, self.violence_patterns)
        
        # Harassment score (pattern-based)
        scores['harassment'] = self._get_pattern_score(text, self.harassment_patterns)
        
        # Bias score (simplified)
        scores['bias'] = self._get_simple_bias_score(text)
        
        # Calculate weighted overall score
        overall_score = sum(
            scores[category] * weight 
            for category, weight in self.harm_categories.items()
        )
        
        scores['overall'] = overall_score
        
        return scores
    
    def _get_toxicity_score(self, text: str) -> float:
        """Get toxicity score using pre-trained model."""
        if not self.toxicity_model:
            return self._get_pattern_score(text, self.violence_patterns + self.harassment_patterns)
        
        try:
            result = self.toxicity_model(text)
            # Find toxic label score
            for item in result:
                if item['label'].lower() in ['toxic', 'toxicity', '1']:
                    return item['score']
            return 0.0
        except Exception as e:
            logger.error(f"Error in toxicity scoring: {e}")
            return 0.0
    
    def _get_hate_speech_score(self, text: str) -> float:
        """Get hate speech score using pre-trained model."""
        if not self.hate_speech_model:
            return self._get_pattern_score(text, self.harassment_patterns)
        
        try:
            result = self.hate_speech_model(text)
            # Find hate/offensive label score
            for item in result:
                if item['label'].lower() in ['hate', 'offensive', 'hateful']:
                    return item['score']
            return 0.0
        except Exception as e:
            logger.error(f"Error in hate speech scoring: {e}")
            return 0.0
    
    def _get_pattern_score(self, text: str, patterns: List[str]) -> float:
        """Get harm score based on pattern matching."""
        text_lower = text.lower()
        matches = 0
        total_patterns = len(patterns)
        
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                matches += 1
        
        return matches / total_patterns if total_patterns > 0 else 0.0
    
    def _get_simple_bias_score(self, text: str) -> float:
        """Get simple bias score based on demographic term frequency."""
        text_lower = text.lower()
        bias_indicators = [
            'men are', 'women are', 'blacks are', 'whites are', 'asians are',
            'muslims are', 'christians are', 'jews are', 'all men', 'all women'
        ]
        
        matches = sum(1 for indicator in bias_indicators if indicator in text_lower)
        return min(matches * 0.2, 1.0)  # Cap at 1.0


class TSDAISafetyLayer:
    """Token-level Safety-Debiased Inference (TSDI) safety layer."""
    
    def __init__(self, harm_threshold: float = 0.2):
        """Initialize TSDI safety layer."""
        self.harm_threshold = harm_threshold
        self.bias_vector = BiasVector()
        self.harm_scorer = HarmScorer()
        
        # Safety statistics
        self.total_processed = 0
        self.total_blocked = 0
        self.bias_corrections = 0
        
        # Safety log
        self.safety_log = []
        
        # Initialize components
        self._initialize_safety_components()
        
        logger.info(f"TSDI Safety Layer initialized with threshold {harm_threshold}")
    
    def _initialize_safety_components(self):
        """Initialize bias vectors and harm scoring models."""
        try:
            self.bias_vector.initialize_embeddings()
            # Harm scorer is already initialized in its __init__
            logger.info("TSDI safety components initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing TSDI components: {e}")
    
    def preprocess_input(self, text: str, debias: bool = True) -> Dict[str, Any]:
        """Preprocess input text with bias subtraction and safety checks."""
        self.total_processed += 1
        
        # Initial safety check
        harm_scores = self.harm_scorer.calculate_harm_score(text)
        
        # Check if input should be blocked
        if harm_scores['overall'] > self.harm_threshold:
            self.total_blocked += 1
            self._log_safety_event('input_blocked', text, harm_scores)
            
            return {
                'status': 'blocked',
                'reason': 'harmful_input',
                'text': text,
                'harm_scores': harm_scores,
                'message': self._get_block_message(harm_scores),
                'timestamp': datetime.now().isoformat()
            }
        
        # Perform bias subtraction if requested
        processed_text = text
        bias_info = {}
        
        if debias and self.bias_vector.embedding_model:
            processed_text, bias_info = self._subtract_bias(text)
            if bias_info.get('bias_detected', False):
                self.bias_corrections += 1
        
        self._log_safety_event('input_processed', text, harm_scores, bias_info)
        
        return {
            'status': 'safe',
            'original_text': text,
            'processed_text': processed_text,
            'harm_scores': harm_scores,
            'bias_info': bias_info,
            'timestamp': datetime.now().isoformat()
        }
    
    def postprocess_output(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Post-process output text with harm scoring and safety checks."""
        # Calculate harm scores for output
        harm_scores = self.harm_scorer.calculate_harm_score(text)
        
        # Check if output should be blocked
        if harm_scores['overall'] > self.harm_threshold:
            self.total_blocked += 1
            self._log_safety_event('output_blocked', text, harm_scores, context)
            
            return {
                'status': 'blocked',
                'reason': 'harmful_output',
                'text': text,
                'harm_scores': harm_scores,
                'message': self._get_block_message(harm_scores),
                'timestamp': datetime.now().isoformat()
            }
        
        # Additional bias check for output
        bias_scores = {}
        if self.bias_vector.embedding_model:
            for bias_type in ['gender', 'race_white', 'race_black', 'religion_christian']:
                if bias_type in self.bias_vector.bias_vectors:
                    bias_scores[bias_type] = self.bias_vector.get_bias_score(text, bias_type)
        
        self._log_safety_event('output_processed', text, harm_scores, {'bias_scores': bias_scores})
        
        return {
            'status': 'safe',
            'text': text,
            'harm_scores': harm_scores,
            'bias_scores': bias_scores,
            'timestamp': datetime.now().isoformat()
        }
    
    def _subtract_bias(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """Subtract bias from text using TSDI approach."""
        bias_info = {
            'bias_detected': False,
            'corrections_made': [],
            'bias_scores': {}
        }
        
        # Calculate bias scores for different types
        for bias_type in ['gender']:  # Start with gender bias
            if bias_type in self.bias_vector.bias_vectors:
                bias_score = self.bias_vector.get_bias_score(text, bias_type)
                bias_info['bias_scores'][bias_type] = bias_score
                
                # If significant bias detected, apply correction
                if abs(bias_score) > 0.3:  # Threshold for bias correction
                    bias_info['bias_detected'] = True
                    text = self._apply_bias_correction(text, bias_type, bias_score)
                    bias_info['corrections_made'].append(bias_type)
        
        return text, bias_info
    
    def _apply_bias_correction(self, text: str, bias_type: str, bias_score: float) -> str:
        """Apply bias correction to text."""
        if bias_type == 'gender':
            # Simple gender-neutral replacements
            corrections = {
                r'\bhe\b': 'they',
                r'\bhim\b': 'them', 
                r'\bhis\b': 'their',
                r'\bshe\b': 'they',
                r'\bher\b': 'their',
                r'\bhers\b': 'theirs',
                r'\bman\b': 'person',
                r'\bwoman\b': 'person',
                r'\bguys?\b': 'people',
                r'\bgirls?\b': 'people'
            }
            
            corrected_text = text
            for pattern, replacement in corrections.items():
                corrected_text = re.sub(pattern, replacement, corrected_text, flags=re.IGNORECASE)
            
            return corrected_text
        
        return text  # No correction applied
    
    def _get_block_message(self, harm_scores: Dict[str, float]) -> str:
        """Generate appropriate block message based on harm scores."""
        max_category = max(harm_scores.items(), key=lambda x: x[1] if x[0] != 'overall' else 0)
        category, score = max_category
        
        messages = {
            'toxicity': f"Content blocked due to toxic language (score: {score:.3f})",
            'hate_speech': f"Content blocked due to hate speech (score: {score:.3f})",
            'violence': f"Content blocked due to violent content (score: {score:.3f})",
            'harassment': f"Content blocked due to harassment (score: {score:.3f})",
            'bias': f"Content blocked due to biased language (score: {score:.3f})"
        }
        
        return messages.get(category, f"Content blocked due to safety concerns (overall score: {harm_scores['overall']:.3f})")
    
    def _log_safety_event(self, event_type: str, text: str, harm_scores: Dict[str, float], 
                         additional_info: Optional[Dict[str, Any]] = None):
        """Log safety events for audit and monitoring."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'text_preview': text[:100] + '...' if len(text) > 100 else text,
            'text_length': len(text),
            'harm_scores': harm_scores,
            'overall_score': harm_scores.get('overall', 0.0),
            'blocked': event_type.endswith('_blocked'),
            'additional_info': additional_info or {}
        }
        
        self.safety_log.append(event)
        
        # Keep only last 1000 events
        if len(self.safety_log) > 1000:
            self.safety_log = self.safety_log[-1000:]
    
    def get_safety_statistics(self) -> Dict[str, Any]:
        """Get comprehensive safety statistics."""
        safety_rate = (self.total_processed - self.total_blocked) / max(self.total_processed, 1)
        
        return {
            'total_processed': self.total_processed,
            'total_blocked': self.total_blocked,
            'safety_rate': safety_rate,
            'bias_corrections': self.bias_corrections,
            'harm_threshold': self.harm_threshold,
            'recent_events': self.safety_log[-10:],  # Last 10 events
            'total_events_logged': len(self.safety_log)
        }
    
    def reset_statistics(self):
        """Reset safety statistics."""
        self.total_processed = 0
        self.total_blocked = 0
        self.bias_corrections = 0
        self.safety_log.clear()
        logger.info("Safety statistics reset")
    
    def update_threshold(self, new_threshold: float):
        """Update harm threshold."""
        old_threshold = self.harm_threshold
        self.harm_threshold = new_threshold
        logger.info(f"Harm threshold updated from {old_threshold} to {new_threshold}")
    
    def export_safety_log(self, filepath: str):
        """Export safety log to file."""
        import json
        try:
            with open(filepath, 'w') as f:
                json.dump(self.safety_log, f, indent=2, ensure_ascii=False)
            logger.info(f"Safety log exported to {filepath}")
        except Exception as e:
            logger.error(f"Error exporting safety log: {e}")
            raise


class SafetyWrapper:
    """Wrapper class to integrate TSDI safety layer with any agent."""
    
    def __init__(self, agent, safety_layer: TSDAISafetyLayer):
        """Initialize safety wrapper for an agent."""
        self.agent = agent
        self.safety_layer = safety_layer
        self.wrapped_methods = ['process', 'safe_process', '_safe_process_implementation']
        
    def __getattr__(self, name):
        """Delegate attribute access to wrapped agent."""
        return getattr(self.agent, name)
    
    def process(self, *args, **kwargs):
        """Wrap agent process method with safety layer."""
        return self._safe_execute('process', *args, **kwargs)
    
    def safe_process(self, *args, **kwargs):
        """Wrap agent safe_process method with safety layer."""
        return self._safe_execute('safe_process', *args, **kwargs)
    
    def _safe_execute(self, method_name: str, *args, **kwargs):
        """Execute agent method with TSDI safety wrapper."""
        # Extract text input for preprocessing
        input_text = self._extract_text_input(*args, **kwargs)
        
        if input_text:
            # Preprocess input
            preprocess_result = self.safety_layer.preprocess_input(input_text)
            
            if preprocess_result['status'] == 'blocked':
                return {
                    'agent': getattr(self.agent, 'name', 'Unknown'),
                    'status': 'blocked',
                    'reason': preprocess_result['reason'],
                    'message': preprocess_result['message'],
                    'safety_details': preprocess_result,
                    'timestamp': preprocess_result['timestamp']
                }
            
            # Update input with processed text if bias correction was applied
            if preprocess_result.get('processed_text') != input_text:
                args, kwargs = self._update_text_input(
                    preprocess_result['processed_text'], *args, **kwargs
                )
        
        # Execute original agent method
        try:
            if hasattr(self.agent, method_name):
                result = getattr(self.agent, method_name)(*args, **kwargs)
            else:
                raise AttributeError(f"Agent does not have method '{method_name}'")
            
            # Extract output text for postprocessing
            output_text = self._extract_text_output(result)
            
            if output_text:
                # Postprocess output
                postprocess_result = self.safety_layer.postprocess_output(
                    output_text, context={'agent_result': result}
                )
                
                if postprocess_result['status'] == 'blocked':
                    return {
                        'agent': getattr(self.agent, 'name', 'Unknown'),
                        'status': 'blocked',
                        'reason': postprocess_result['reason'],
                        'message': postprocess_result['message'],
                        'safety_details': postprocess_result,
                        'timestamp': postprocess_result['timestamp']
                    }
                
                # Add safety information to result
                if isinstance(result, dict):
                    result['safety_verified'] = True
                    result['harm_scores'] = postprocess_result['harm_scores']
                    result['bias_scores'] = postprocess_result.get('bias_scores', {})
            
            return result
            
        except Exception as e:
            logger.error(f"Error in wrapped agent execution: {e}")
            return {
                'agent': getattr(self.agent, 'name', 'Unknown'),
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _extract_text_input(self, *args, **kwargs) -> Optional[str]:
        """Extract text input from agent method arguments."""
        # Check args
        for arg in args:
            if isinstance(arg, str):
                return arg
            elif isinstance(arg, dict):
                for key in ['text', 'query', 'question', 'input']:
                    if key in arg and isinstance(arg[key], str):
                        return arg[key]
        
        # Check kwargs
        for key in ['text', 'query', 'question', 'input']:
            if key in kwargs and isinstance(kwargs[key], str):
                return kwargs[key]
        
        return None
    
    def _extract_text_output(self, result) -> Optional[str]:
        """Extract text output from agent result."""
        if isinstance(result, str):
            return result
        elif isinstance(result, dict):
            for key in ['response', 'summary', 'translation', 'answer', 'output', 'text']:
                if key in result and isinstance(result[key], str):
                    return result[key]
        
        return None
    
    def _update_text_input(self, new_text: str, *args, **kwargs):
        """Update text input in arguments with processed text."""
        # Update args
        new_args = []
        for arg in args:
            if isinstance(arg, str):
                new_args.append(new_text)
            elif isinstance(arg, dict):
                new_arg = arg.copy()
                for key in ['text', 'query', 'question', 'input']:
                    if key in new_arg and isinstance(new_arg[key], str):
                        new_arg[key] = new_text
                        break
                new_args.append(new_arg)
            else:
                new_args.append(arg)
        
        # Update kwargs
        new_kwargs = kwargs.copy()
        for key in ['text', 'query', 'question', 'input']:
            if key in new_kwargs and isinstance(new_kwargs[key], str):
                new_kwargs[key] = new_text
                break
        
        return tuple(new_args), new_kwargs


# Global safety layer instance
_global_safety_layer = None

def get_global_safety_layer() -> TSDAISafetyLayer:
    """Get or create global TSDI safety layer instance."""
    global _global_safety_layer
    if _global_safety_layer is None:
        _global_safety_layer = TSDAISafetyLayer()
    return _global_safety_layer

def wrap_agent_with_safety(agent) -> SafetyWrapper:
    """Wrap any agent with TSDI safety layer."""
    return SafetyWrapper(agent, get_global_safety_layer())

def set_global_harm_threshold(threshold: float):
    """Set global harm threshold for all safety operations."""
    safety_layer = get_global_safety_layer()
    safety_layer.update_threshold(threshold)

def get_global_safety_statistics() -> Dict[str, Any]:
    """Get global safety statistics."""
    safety_layer = get_global_safety_layer()
    return safety_layer.get_safety_statistics()
