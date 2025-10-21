"""Configuration management for the multi-agent system."""

import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class AgentConfig(BaseModel):
    """Configuration for individual agents."""
    name: str
    model_name: str = "gpt-3.5-turbo"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, gt=0)
    system_prompt: str = ""


class GuardrailsConfig(BaseModel):
    """Configuration for NeMo Guardrails."""
    enabled: bool = True
    config_path: Optional[str] = None
    rails_config: Dict[str, Any] = Field(default_factory=dict)


class Config(BaseModel):
    """Main configuration class for the multi-agent system."""
    
    # API Keys
    openai_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    huggingface_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("HUGGINGFACE_API_KEY"))
    
    # Data settings
    data_directory: str = "data"
    max_sample_size: int = 100
    
    # Agent configurations
    safety_agent: AgentConfig = Field(default_factory=lambda: AgentConfig(
        name="safety_agent",
        system_prompt="You are a safety evaluation agent. Analyze text for harmful, toxic, or inappropriate content."
    ))
    
    analysis_agent: AgentConfig = Field(default_factory=lambda: AgentConfig(
        name="analysis_agent",
        system_prompt="You are an analysis agent. Provide detailed analysis of text content and patterns."
    ))
    
    coordinator_agent: AgentConfig = Field(default_factory=lambda: AgentConfig(
        name="coordinator_agent",
        system_prompt="You are a coordinator agent. Orchestrate tasks between different agents and summarize results."
    ))
    
    # Guardrails configuration
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "multi_agent.log"
    
    class Config:
        env_prefix = "SAFETY_ALIGN_"
        case_sensitive = False
