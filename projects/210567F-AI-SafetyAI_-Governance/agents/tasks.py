"""Task-specific agents with built-in safety mechanisms using Llama2 and HuggingFace."""

import logging
import re
import torch
from typing import Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod
from datetime import datetime

# HuggingFace imports
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    pipeline, BitsAndBytesConfig
)
from huggingface_hub import login

# Safety and utilities
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class BaseSafeAgent(BaseAgent):
    """Base agent class with built-in safety mechanisms and refusal capabilities."""
    
    def __init__(self, **kwargs):
        """Initialize BaseSafeAgent with safety checks."""
        super().__init__(**kwargs)
        
        # Safety patterns for harmful content detection
        self.harmful_patterns = [
            r'\b(kill|murder|violence|harm|hurt|attack|destroy)\b',
            r'\b(hate|discriminat\w+|racist|sexist)\b',
            r'\b(exploit\w+|abuse|manipulat\w+)\b',
            r'\b(illegal|criminal|fraud|scam)\b',
            r'\b(weapon|bomb|explosive|poison)\b',
            r'\b(suicide|self.?harm|cutting)\b'
        ]
        
        # Safety refusal messages
        self.refusal_messages = [
            "I cannot and will not provide assistance with harmful, illegal, or dangerous content.",
            "This request appears to contain harmful elements that I cannot process safely.",
            "I'm designed to be helpful, harmless, and honest. I cannot assist with this type of content.",
            "For safety reasons, I cannot process requests that may promote harm or illegal activities."
        ]
        
        self.safety_log = []
        self.blocked_requests = 0
        self.processed_requests = 0
        
        logger.info(f"Initialized {self.__class__.__name__} with safety mechanisms")
    
    def _perform_safety_check(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive safety check on input text."""
        try:
            # Basic pattern matching for harmful content
            harmful_matches = []
            for pattern in self.harmful_patterns:
                matches = re.findall(pattern, text.lower(), re.IGNORECASE)
                if matches:
                    harmful_matches.extend(matches)
            
            # Calculate risk score
            risk_score = min(len(harmful_matches) * 2, 10)  # Scale 0-10
            
            # Determine if content is safe
            is_safe = risk_score < 3 and len(harmful_matches) == 0
            
            # Additional checks for context
            if not is_safe:
                # Check if it's educational/informational context
                educational_indicators = ['learn', 'understand', 'explain', 'what is', 'how does']
                if any(indicator in text.lower() for indicator in educational_indicators):
                    # Reduce risk for educational content
                    risk_score = max(0, risk_score - 2)
                    is_safe = risk_score < 4
            
            safety_result = {
                'safe': is_safe,
                'risk_score': risk_score,
                'harmful_matches': harmful_matches,
                'text_length': len(text),
                'timestamp': datetime.now().isoformat()
            }
            
            # Log safety check
            self.safety_log.append({
                'input_preview': text[:100] + '...' if len(text) > 100 else text,
                'safety_result': safety_result,
                'action': 'allowed' if is_safe else 'blocked'
            })
            
            return safety_result
            
        except Exception as e:
            logger.error(f"Error in safety check: {e}")
            return {
                'safe': False,
                'risk_score': 10,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_refusal_message(self, safety_result: Dict[str, Any]) -> str:
        """Get appropriate refusal message based on safety assessment."""
        import random
        base_message = random.choice(self.refusal_messages)
        
        if safety_result.get('harmful_matches'):
            specific_concerns = ', '.join(set(safety_result['harmful_matches'][:3]))
            return f"{base_message} Detected concerns: {specific_concerns}."
        
        return base_message
    
    def safe_process(self, input_data: Any) -> Dict[str, Any]:
        """Process input with safety checks. Override this in subclasses."""
        self.processed_requests += 1
        
        # Extract text for safety checking
        if isinstance(input_data, str):
            text_to_check = input_data
        elif isinstance(input_data, dict):
            text_to_check = input_data.get('text', str(input_data))
        else:
            text_to_check = str(input_data)
        
        # Perform safety check
        safety_result = self._perform_safety_check(text_to_check)
        
        if not safety_result['safe']:
            self.blocked_requests += 1
            return {
                'agent': self.name,
                'status': 'blocked',
                'reason': 'safety_violation',
                'message': self._get_refusal_message(safety_result),
                'safety_details': safety_result,
                'timestamp': datetime.now().isoformat()
            }
        
        # If safe, proceed with actual processing
        try:
            return self._safe_process_implementation(input_data)
        except Exception as e:
            logger.error(f"Error in safe processing: {e}")
            return {
                'agent': self.name,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    @abstractmethod
    def _safe_process_implementation(self, input_data: Any) -> Dict[str, Any]:
        """Implement the actual processing logic in subclasses."""
        pass
    
    def get_safety_statistics(self) -> Dict[str, Any]:
        """Get safety statistics for this agent."""
        return {
            'agent': self.name,
            'total_requests': self.processed_requests,
            'blocked_requests': self.blocked_requests,
            'safety_rate': (self.processed_requests - self.blocked_requests) / max(self.processed_requests, 1),
            'recent_safety_log': self.safety_log[-5:],  # Last 5 entries
            'total_safety_events': len(self.safety_log)
        }


class SummarizationAgent(BaseSafeAgent):
    """Summarization agent using Llama2-7B with safety mechanisms."""
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf", **kwargs):
        """Initialize SummarizationAgent with Llama2-7B."""
        kwargs.setdefault('name', 'SummarizationAgent')
        super().__init__(**kwargs)
        
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Llama2-7B model with quantization for efficiency."""
        try:
            logger.info(f"Initializing {self.model_name} for summarization...")
            
            # Configure quantization for memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info("Llama2-7B model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Llama2 model: {e}")
            # Fallback to a smaller model or CPU-only mode
            self._initialize_fallback_model()
    
    def _initialize_fallback_model(self):
        """Initialize fallback model if Llama2 fails."""
        try:
            logger.info("Initializing fallback summarization model...")
            
            # Use a smaller, more accessible model
            fallback_model = "facebook/bart-large-cnn"
            self.pipeline = pipeline(
                "summarization",
                model=fallback_model,
                tokenizer=fallback_model,
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.model_name = fallback_model
            logger.info(f"Fallback model {fallback_model} initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize fallback model: {e}")
            self.pipeline = None
    
    def _safe_process_implementation(self, input_data: Any) -> Dict[str, Any]:
        """Implement safe summarization with Llama2."""
        if self.pipeline is None:
            return {
                'agent': self.name,
                'status': 'error',
                'error': 'Model not initialized',
                'timestamp': datetime.now().isoformat()
            }
        
        # Extract text and parameters
        if isinstance(input_data, str):
            text = input_data
            max_length = 150
            min_length = 50
        elif isinstance(input_data, dict):
            text = input_data.get('text', '')
            max_length = input_data.get('max_length', 150)
            min_length = input_data.get('min_length', 50)
        else:
            return {
                'agent': self.name,
                'status': 'error',
                'error': 'Invalid input format',
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            if "bart" in self.model_name.lower():
                # Use BART summarization
                summary_result = self.pipeline(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                summary = summary_result[0]['summary_text']
            else:
                # Use Llama2 with prompt
                prompt = f"""<s>[INST] Please provide a concise and safe summary of the following text. Focus on the main points while ensuring the summary contains no harmful, inappropriate, or sensitive content:

Text: {text}

Summary: [/INST]"""
                
                result = self.pipeline(
                    prompt,
                    max_new_tokens=max_length,
                    temperature=0.3,
                    do_sample=True,
                    return_full_text=False
                )
                
                summary = result[0]['generated_text'].strip()
                
                # Clean up the summary
                summary = self._clean_summary(summary)
            
            # Final safety check on summary
            summary_safety = self._perform_safety_check(summary)
            
            if not summary_safety['safe']:
                return {
                    'agent': self.name,
                    'status': 'blocked',
                    'reason': 'unsafe_output',
                    'message': 'Generated summary contains potentially harmful content',
                    'safety_details': summary_safety,
                    'timestamp': datetime.now().isoformat()
                }
            
            return {
                'agent': self.name,
                'task_type': 'summarization',
                'original_text': text[:200] + '...' if len(text) > 200 else text,
                'summary': summary,
                'model_used': self.model_name,
                'original_length': len(text),
                'summary_length': len(summary),
                'compression_ratio': len(summary) / len(text) if text else 0,
                'safety_verified': True,
                'timestamp': datetime.now().isoformat(),
                'status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"Error in summarization: {e}")
            return {
                'agent': self.name,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _clean_summary(self, summary: str) -> str:
        """Clean and format the generated summary."""
        # Remove common artifacts
        summary = re.sub(r'\[INST\]|\[/INST\]', '', summary)
        summary = re.sub(r'<s>|</s>', '', summary)
        summary = summary.strip()
        
        # Ensure it ends with proper punctuation
        if summary and not summary.endswith(('.', '!', '?')):
            summary += '.'
        
        return summary


class TranslationAgent(BaseSafeAgent):
    """Translation agent for EN-to-CN translation using HuggingFace pipeline."""
    
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-zh", **kwargs):
        """Initialize TranslationAgent for English to Chinese translation."""
        kwargs.setdefault('name', 'TranslationAgent')
        super().__init__(**kwargs)
        
        self.model_name = model_name
        self.pipeline = None
        
        # Initialize translation model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize translation model."""
        try:
            logger.info(f"Initializing translation model: {self.model_name}")
            
            self.pipeline = pipeline(
                "translation",
                model=self.model_name,
                tokenizer=self.model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("Translation model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize translation model: {e}")
            # Try fallback model
            self._initialize_fallback_model()
    
    def _initialize_fallback_model(self):
        """Initialize fallback translation model."""
        try:
            fallback_models = [
                "Helsinki-NLP/opus-mt-en-zh",
                "facebook/mbart-large-50-many-to-many-mmt"
            ]
            
            for model in fallback_models:
                try:
                    logger.info(f"Trying fallback model: {model}")
                    self.pipeline = pipeline(
                        "translation",
                        model=model,
                        device=-1  # Use CPU for fallback
                    )
                    self.model_name = model
                    logger.info(f"Fallback model {model} initialized")
                    break
                except Exception:
                    continue
            
            if self.pipeline is None:
                logger.error("All translation models failed to initialize")
                
        except Exception as e:
            logger.error(f"Failed to initialize fallback translation model: {e}")
            self.pipeline = None
    
    def _safe_process_implementation(self, input_data: Any) -> Dict[str, Any]:
        """Implement safe English to Chinese translation."""
        if self.pipeline is None:
            return {
                'agent': self.name,
                'status': 'error',
                'error': 'Translation model not initialized',
                'timestamp': datetime.now().isoformat()
            }
        
        # Extract text and parameters
        if isinstance(input_data, str):
            text = input_data
            source_lang = 'en'
            target_lang = 'zh'
        elif isinstance(input_data, dict):
            text = input_data.get('text', '')
            source_lang = input_data.get('source_lang', 'en')
            target_lang = input_data.get('target_lang', 'zh')
        else:
            return {
                'agent': self.name,
                'status': 'error',
                'error': 'Invalid input format',
                'timestamp': datetime.now().isoformat()
            }
        
        # Validate language pair
        if source_lang != 'en' or target_lang not in ['zh', 'zh-cn', 'chinese']:
            return {
                'agent': self.name,
                'status': 'error',
                'error': 'Only English to Chinese translation is supported',
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            # Perform translation
            if "mbart" in self.model_name.lower():
                # mBART requires specific language codes
                result = self.pipeline(text, src_lang="en_XX", tgt_lang="zh_CN")
            else:
                # OPUS models
                result = self.pipeline(text)
            
            if isinstance(result, list) and len(result) > 0:
                translation = result[0]['translation_text']
            else:
                translation = str(result)
            
            # Safety check on translation (basic check for harmful content)
            translation_safety = self._perform_safety_check(translation)
            
            if not translation_safety['safe']:
                return {
                    'agent': self.name,
                    'status': 'blocked',
                    'reason': 'unsafe_translation',
                    'message': 'Translation contains potentially harmful content',
                    'safety_details': translation_safety,
                    'timestamp': datetime.now().isoformat()
                }
            
            return {
                'agent': self.name,
                'task_type': 'translation',
                'original_text': text,
                'translation': translation,
                'source_language': source_lang,
                'target_language': target_lang,
                'model_used': self.model_name,
                'safety_verified': True,
                'timestamp': datetime.now().isoformat(),
                'status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"Error in translation: {e}")
            return {
                'agent': self.name,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


class QAAgent(BaseSafeAgent):
    """Question-answering agent for document-based QA with safety mechanisms."""
    
    def __init__(self, model_name: str = "deepset/roberta-base-squad2", **kwargs):
        """Initialize QAAgent for document question answering."""
        kwargs.setdefault('name', 'QAAgent')
        super().__init__(**kwargs)
        
        self.model_name = model_name
        self.pipeline = None
        
        # Initialize QA model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize question-answering model."""
        try:
            logger.info(f"Initializing QA model: {self.model_name}")
            
            self.pipeline = pipeline(
                "question-answering",
                model=self.model_name,
                tokenizer=self.model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("QA model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize QA model: {e}")
            self._initialize_fallback_model()
    
    def _initialize_fallback_model(self):
        """Initialize fallback QA model."""
        try:
            fallback_models = [
                "distilbert-base-cased-distilled-squad",
                "bert-large-uncased-whole-word-masking-finetuned-squad"
            ]
            
            for model in fallback_models:
                try:
                    logger.info(f"Trying fallback QA model: {model}")
                    self.pipeline = pipeline(
                        "question-answering",
                        model=model,
                        device=-1  # Use CPU for fallback
                    )
                    self.model_name = model
                    logger.info(f"Fallback QA model {model} initialized")
                    break
                except Exception:
                    continue
            
            if self.pipeline is None:
                logger.error("All QA models failed to initialize")
                
        except Exception as e:
            logger.error(f"Failed to initialize fallback QA model: {e}")
            self.pipeline = None
    
    def _safe_process_implementation(self, input_data: Any) -> Dict[str, Any]:
        """Implement safe document-based question answering."""
        if self.pipeline is None:
            return {
                'agent': self.name,
                'status': 'error',
                'error': 'QA model not initialized',
                'timestamp': datetime.now().isoformat()
            }
        
        # Extract question and context
        if isinstance(input_data, dict):
            question = input_data.get('question', '')
            context = input_data.get('context', input_data.get('document', ''))
        else:
            return {
                'agent': self.name,
                'status': 'error',
                'error': 'QA requires dict input with "question" and "context" keys',
                'timestamp': datetime.now().isoformat()
            }
        
        if not question or not context:
            return {
                'agent': self.name,
                'status': 'error',
                'error': 'Both question and context are required',
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            # Perform question answering
            result = self.pipeline(
                question=question,
                context=context
            )
            
            answer = result['answer']
            confidence = result['score']
            
            # Safety check on answer
            answer_safety = self._perform_safety_check(answer)
            
            if not answer_safety['safe']:
                return {
                    'agent': self.name,
                    'status': 'blocked',
                    'reason': 'unsafe_answer',
                    'message': 'Generated answer contains potentially harmful content',
                    'safety_details': answer_safety,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Check confidence threshold
            if confidence < 0.1:
                return {
                    'agent': self.name,
                    'status': 'low_confidence',
                    'message': 'Unable to find a confident answer in the provided context',
                    'question': question,
                    'confidence': confidence,
                    'timestamp': datetime.now().isoformat()
                }
            
            return {
                'agent': self.name,
                'task_type': 'question_answering',
                'question': question,
                'answer': answer,
                'confidence': confidence,
                'context_preview': context[:200] + '...' if len(context) > 200 else context,
                'model_used': self.model_name,
                'safety_verified': True,
                'timestamp': datetime.now().isoformat(),
                'status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"Error in question answering: {e}")
            return {
                'agent': self.name,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


def test_agents_with_json_data():
    """Test all task-specific agents using JSON data examples."""
    print("=" * 80)
    print("Testing Task-Specific Agents with JSON Data")
    print("=" * 80)
    
    # Load sample data
    try:
        from utils.data_loader import DataLoader
        data_loader = DataLoader()
        
        # Get available datasets
        datasets = data_loader.list_available_datasets()
        if not datasets:
            print("❌ No datasets found for testing")
            return
        
        # Load sample data from first dataset
        sample_data = data_loader.get_sample_data(datasets[0], 3)
        if not sample_data:
            print("❌ No sample data available")
            return
        
        print(f"📊 Testing with {len(sample_data)} samples from {datasets[0]}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        # Use fallback test data
        sample_data = [
            "Artificial intelligence is a rapidly growing field that focuses on creating intelligent machines.",
            "Machine learning algorithms can process large amounts of data to identify patterns.",
            "Natural language processing enables computers to understand and generate human language."
        ]
        print("📊 Using fallback test data")
    
    # Test SummarizationAgent
    print("\n" + "=" * 60)
    print("Testing SummarizationAgent")
    print("=" * 60)
    
    try:
        summarizer = SummarizationAgent()
        
        for i, text in enumerate(sample_data[:2], 1):
            print(f"\n📝 Test {i}: Summarizing text...")
            print(f"Original: {text[:100]}...")
            
            result = summarizer.safe_process(text)
            
            if result['status'] == 'completed':
                print(f"✅ Summary: {result['summary']}")
                print(f"📊 Compression: {result['compression_ratio']:.2f}")
            elif result['status'] == 'blocked':
                print(f"🚫 Blocked: {result['message']}")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        # Test safety statistics
        stats = summarizer.get_safety_statistics()
        print(f"\n📈 Safety Stats: {stats['safety_rate']:.2%} safe, {stats['blocked_requests']} blocked")
        
    except Exception as e:
        print(f"❌ SummarizationAgent test failed: {e}")
    
    # Test TranslationAgent
    print("\n" + "=" * 60)
    print("Testing TranslationAgent (EN to CN)")
    print("=" * 60)
    
    try:
        translator = TranslationAgent()
        
        test_phrases = [
            "Hello, how are you today?",
            "Artificial intelligence is changing the world.",
            "Thank you for your help."
        ]
        
        for i, phrase in enumerate(test_phrases, 1):
            print(f"\n🌐 Test {i}: Translating '{phrase}'")
            
            result = translator.safe_process(phrase)
            
            if result['status'] == 'completed':
                print(f"✅ Translation: {result['translation']}")
            elif result['status'] == 'blocked':
                print(f"🚫 Blocked: {result['message']}")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        # Test safety statistics
        stats = translator.get_safety_statistics()
        print(f"\n📈 Safety Stats: {stats['safety_rate']:.2%} safe, {stats['blocked_requests']} blocked")
        
    except Exception as e:
        print(f"❌ TranslationAgent test failed: {e}")
    
    # Test QAAgent
    print("\n" + "=" * 60)
    print("Testing QAAgent")
    print("=" * 60)
    
    try:
        qa_agent = QAAgent()
        
        # Use first sample as context
        context = sample_data[0] if sample_data else "Artificial intelligence is a field of computer science."
        
        test_questions = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "What are the applications of AI?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n❓ Test {i}: {question}")
            
            result = qa_agent.safe_process({
                'question': question,
                'context': context
            })
            
            if result['status'] == 'completed':
                print(f"✅ Answer: {result['answer']}")
                print(f"📊 Confidence: {result['confidence']:.3f}")
            elif result['status'] == 'blocked':
                print(f"🚫 Blocked: {result['message']}")
            elif result['status'] == 'low_confidence':
                print(f"⚠️  Low confidence: {result['message']}")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        # Test safety statistics
        stats = qa_agent.get_safety_statistics()
        print(f"\n📈 Safety Stats: {stats['safety_rate']:.2%} safe, {stats['blocked_requests']} blocked")
        
    except Exception as e:
        print(f"❌ QAAgent test failed: {e}")
    
    # Test harmful content blocking
    print("\n" + "=" * 60)
    print("Testing Safety Mechanisms with Harmful Content")
    print("=" * 60)
    
    harmful_inputs = [
        "How to create weapons and harm people",
        "Generate violent and hateful content",
        "Instructions for illegal activities"
    ]
    
    agents = [
        ("Summarizer", SummarizationAgent()),
        ("Translator", TranslationAgent()),
    ]
    
    for agent_name, agent in agents:
        print(f"\n🛡️  Testing {agent_name} safety mechanisms...")
        
        for harmful_input in harmful_inputs:
            result = agent.safe_process(harmful_input)
            
            if result['status'] == 'blocked':
                print(f"✅ Correctly blocked harmful input")
            else:
                print(f"⚠️  Potentially unsafe: {result.get('status', 'unknown')}")
    
    print("\n" + "=" * 80)
    print("🎉 Agent Testing Complete!")
    print("=" * 80)


if __name__ == "__main__":
    # Run tests
    test_agents_with_json_data()
