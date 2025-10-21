"""Base agent class for the multi-agent system."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""
    
    def __init__(
        self,
        name: str,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        system_prompt: str = "",
        api_key: Optional[str] = None
    ):
        """Initialize the base agent."""
        self.name = name
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        
        # Initialize the language model
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=api_key
        )
        
        self.conversation_history: List[BaseMessage] = []
        self.token_usage = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
        
        logger.info(f"Initialized {self.name} with model {model_name}")
    
    def _prepare_messages(self, user_input: str, include_history: bool = False) -> List[BaseMessage]:
        """Prepare messages for the language model."""
        messages = []
        
        # Add system prompt if provided
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        
        # Add conversation history if requested
        if include_history and self.conversation_history:
            messages.extend(self.conversation_history)
        
        # Add current user input
        messages.append(HumanMessage(content=user_input))
        
        return messages
    
    def _update_conversation_history(self, user_input: str, response: str):
        """Update the conversation history."""
        self.conversation_history.append(HumanMessage(content=user_input))
        self.conversation_history.append(SystemMessage(content=response))
        
        # Keep only last 10 messages to prevent context overflow
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def _track_token_usage(self, callback_data):
        """Track token usage for cost monitoring."""
        if hasattr(callback_data, 'total_tokens'):
            self.token_usage["total_tokens"] += callback_data.total_tokens
            self.token_usage["prompt_tokens"] += callback_data.prompt_tokens
            self.token_usage["completion_tokens"] += callback_data.completion_tokens
    
    def generate_response(
        self, 
        user_input: str, 
        include_history: bool = False,
        track_usage: bool = True
    ) -> str:
        """Generate a response using the language model."""
        try:
            messages = self._prepare_messages(user_input, include_history)
            
            if track_usage:
                with get_openai_callback() as cb:
                    response = self.llm(messages)
                    self._track_token_usage(cb)
            else:
                response = self.llm(messages)
            
            response_content = response.content
            
            # Update conversation history
            if include_history:
                self._update_conversation_history(user_input, response_content)
            
            logger.debug(f"{self.name} generated response of length {len(response_content)}")
            return response_content
            
        except Exception as e:
            logger.error(f"Error generating response in {self.name}: {e}")
            raise
    
    @abstractmethod
    def process(self, input_data: Any) -> Dict[str, Any]:
        """Process input data and return results. Must be implemented by subclasses."""
        pass
    
    def get_token_usage(self) -> Dict[str, int]:
        """Get current token usage statistics."""
        return self.token_usage.copy()
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history.clear()
        logger.info(f"Reset conversation history for {self.name}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information."""
        return {
            "name": self.name,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "conversation_length": len(self.conversation_history),
            "token_usage": self.token_usage
        }
