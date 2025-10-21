"""Task-specific agents for summarization, translation, and QA."""

import logging
from typing import Dict, Any, Optional
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class SummarizationAgent(BaseAgent):
    """Agent specialized in text summarization tasks."""
    
    def __init__(self, **kwargs):
        """Initialize the Summarization Agent."""
        default_system_prompt = """You are a summarization agent specialized in creating concise, accurate summaries of text content.

Your responsibilities include:
1. Extracting key information and main points
2. Maintaining factual accuracy and context
3. Creating summaries of appropriate length based on requirements
4. Preserving important details while removing redundancy
5. Ensuring clarity and readability in summaries

For each summarization task, provide:
- A clear, concise summary that captures the essence of the original text
- Preservation of key facts, figures, and important details
- Appropriate length based on the specified requirements
- Neutral tone and objective perspective

Be accurate and comprehensive in your summarization approach."""
        
        kwargs.setdefault('system_prompt', default_system_prompt)
        kwargs.setdefault('name', 'SummarizationAgent')
        super().__init__(**kwargs)
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        """Process text for summarization."""
        if isinstance(input_data, str):
            text_to_summarize = input_data
            summary_length = "medium"
        elif isinstance(input_data, dict):
            text_to_summarize = input_data.get('text', '')
            summary_length = input_data.get('length', 'medium')
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
        
        return self._summarize_text(text_to_summarize, summary_length)
    
    def _summarize_text(self, text: str, length: str = "medium") -> Dict[str, Any]:
        """Summarize a text with specified length."""
        length_instructions = {
            "short": "Create a brief summary in 1-2 sentences.",
            "medium": "Create a comprehensive summary in 3-5 sentences.",
            "long": "Create a detailed summary in 1-2 paragraphs.",
            "bullet": "Create a bullet-point summary with key points."
        }
        
        instruction = length_instructions.get(length, length_instructions["medium"])
        
        prompt = f"""Summarize the following text. {instruction}

Text to summarize:
"{text}"

Summary:"""
        
        try:
            response = self.generate_response(prompt)
            
            result = {
                "agent": self.name,
                "task_type": "summarization",
                "input_text": text[:200] + "..." if len(text) > 200 else text,
                "summary": response,
                "summary_length": length,
                "original_length": len(text),
                "summary_ratio": len(response) / len(text) if text else 0,
                "timestamp": self._get_timestamp(),
                "status": "completed"
            }
            
            logger.info(f"Summarization completed for text of length {len(text)}")
            return result
            
        except Exception as e:
            logger.error(f"Error in summarization: {e}")
            return {
                "agent": self.name,
                "task_type": "summarization",
                "error": str(e),
                "timestamp": self._get_timestamp(),
                "status": "error"
            }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()


class TranslationAgent(BaseAgent):
    """Agent specialized in text translation tasks."""
    
    def __init__(self, **kwargs):
        """Initialize the Translation Agent."""
        default_system_prompt = """You are a translation agent specialized in accurate, contextual translation between languages.

Your responsibilities include:
1. Providing accurate translations that preserve meaning and context
2. Maintaining appropriate tone and style for the target language
3. Handling idiomatic expressions and cultural nuances
4. Ensuring grammatical correctness in the target language
5. Preserving formatting and structure when appropriate

For each translation task, provide:
- Accurate translation that preserves the original meaning
- Appropriate cultural and contextual adaptations
- Natural-sounding text in the target language
- Preservation of tone and intent from the source text

Be precise and culturally aware in your translation approach."""
        
        kwargs.setdefault('system_prompt', default_system_prompt)
        kwargs.setdefault('name', 'TranslationAgent')
        super().__init__(**kwargs)
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        """Process text for translation."""
        if isinstance(input_data, dict):
            text_to_translate = input_data.get('text', '')
            target_language = input_data.get('target_language', 'English')
            source_language = input_data.get('source_language', 'auto-detect')
        else:
            raise ValueError("Translation requires dict input with 'text' and 'target_language' keys")
        
        return self._translate_text(text_to_translate, target_language, source_language)
    
    def _translate_text(self, text: str, target_language: str, source_language: str = "auto-detect") -> Dict[str, Any]:
        """Translate text to target language."""
        if source_language == "auto-detect":
            prompt = f"""Translate the following text to {target_language}. First detect the source language, then provide an accurate translation that preserves meaning and context.

Text to translate:
"{text}"

Translation to {target_language}:"""
        else:
            prompt = f"""Translate the following text from {source_language} to {target_language}. Provide an accurate translation that preserves meaning and context.

Text to translate ({source_language}):
"{text}"

Translation to {target_language}:"""
        
        try:
            response = self.generate_response(prompt)
            
            result = {
                "agent": self.name,
                "task_type": "translation",
                "input_text": text[:200] + "..." if len(text) > 200 else text,
                "translation": response,
                "source_language": source_language,
                "target_language": target_language,
                "timestamp": self._get_timestamp(),
                "status": "completed"
            }
            
            logger.info(f"Translation completed from {source_language} to {target_language}")
            return result
            
        except Exception as e:
            logger.error(f"Error in translation: {e}")
            return {
                "agent": self.name,
                "task_type": "translation",
                "error": str(e),
                "timestamp": self._get_timestamp(),
                "status": "error"
            }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()


class QAAgent(BaseAgent):
    """Agent specialized in question answering tasks."""
    
    def __init__(self, **kwargs):
        """Initialize the QA Agent."""
        default_system_prompt = """You are a question answering agent specialized in providing accurate, helpful responses to user questions.

Your responsibilities include:
1. Understanding and analyzing user questions thoroughly
2. Providing accurate, factual, and helpful answers
3. Citing sources or indicating when information is uncertain
4. Asking clarifying questions when needed
5. Providing context and explanations for complex topics

For each question answering task, provide:
- Clear, direct answers to the specific question asked
- Supporting information and context when helpful
- Acknowledgment of limitations or uncertainty when appropriate
- Structured responses for complex or multi-part questions

Be helpful, accurate, and honest in your responses."""
        
        kwargs.setdefault('system_prompt', default_system_prompt)
        kwargs.setdefault('name', 'QAAgent')
        super().__init__(**kwargs)
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        """Process question for answering."""
        if isinstance(input_data, str):
            question = input_data
            context = None
        elif isinstance(input_data, dict):
            question = input_data.get('question', '')
            context = input_data.get('context', None)
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
        
        return self._answer_question(question, context)
    
    def _answer_question(self, question: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Answer a question with optional context."""
        if context:
            prompt = f"""Answer the following question based on the provided context. If the context doesn't contain enough information to answer the question, indicate this clearly.

Context:
"{context}"

Question: {question}

Answer:"""
        else:
            prompt = f"""Answer the following question clearly and accurately. If you're uncertain about any aspect of the answer, please indicate this.

Question: {question}

Answer:"""
        
        try:
            response = self.generate_response(prompt)
            
            result = {
                "agent": self.name,
                "task_type": "question_answering",
                "question": question,
                "answer": response,
                "context_provided": context is not None,
                "context": context[:200] + "..." if context and len(context) > 200 else context,
                "timestamp": self._get_timestamp(),
                "status": "completed"
            }
            
            logger.info(f"Question answered: {question[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error in question answering: {e}")
            return {
                "agent": self.name,
                "task_type": "question_answering",
                "question": question,
                "error": str(e),
                "timestamp": self._get_timestamp(),
                "status": "error"
            }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
