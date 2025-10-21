"""Multi-agent system orchestrator with NeMo Guardrails integration."""

import logging
from typing import Dict, Any, List, Optional
from utils.config import Config
from utils.data_loader import DataLoader
from .safety_agent import SafetyAgent
from .analysis_agent import AnalysisAgent
from .coordinator_agent import CoordinatorAgent

# Optional NeMo Guardrails integration
try:
    from nemoguardrails import RailsConfig, LLMRails
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    logging.warning("NeMo Guardrails not available. Install with: pip install nemo_guardrails")

logger = logging.getLogger(__name__)


class MultiAgentSystem:
    """Main orchestrator for the multi-agent safety alignment system."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the multi-agent system."""
        self.config = config or Config()
        self.data_loader = DataLoader(self.config.data_directory)
        
        # Initialize agents
        self.safety_agent = None
        self.analysis_agent = None
        self.coordinator_agent = None
        self.guardrails = None
        
        # System state
        self.is_initialized = False
        self.system_stats = {
            "tasks_completed": 0,
            "errors_encountered": 0,
            "total_texts_processed": 0
        }
        
        self._setup_logging()
        logger.info("MultiAgentSystem initialized")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=self.config.log_file if self.config.log_file else None
        )
    
    def initialize_agents(self) -> bool:
        """Initialize all agents with configuration."""
        try:
            # Initialize Safety Agent
            self.safety_agent = SafetyAgent(
                name=self.config.safety_agent.name,
                model_name=self.config.safety_agent.model_name,
                temperature=self.config.safety_agent.temperature,
                max_tokens=self.config.safety_agent.max_tokens,
                system_prompt=self.config.safety_agent.system_prompt,
                api_key=self.config.openai_api_key
            )
            
            # Initialize Analysis Agent
            self.analysis_agent = AnalysisAgent(
                name=self.config.analysis_agent.name,
                model_name=self.config.analysis_agent.model_name,
                temperature=self.config.analysis_agent.temperature,
                max_tokens=self.config.analysis_agent.max_tokens,
                system_prompt=self.config.analysis_agent.system_prompt,
                api_key=self.config.openai_api_key
            )
            
            # Initialize Coordinator Agent
            self.coordinator_agent = CoordinatorAgent(
                name=self.config.coordinator_agent.name,
                model_name=self.config.coordinator_agent.model_name,
                temperature=self.config.coordinator_agent.temperature,
                max_tokens=self.config.coordinator_agent.max_tokens,
                system_prompt=self.config.coordinator_agent.system_prompt,
                api_key=self.config.openai_api_key
            )
            
            # Register agents with coordinator
            self.coordinator_agent.register_agent('safety_agent', self.safety_agent)
            self.coordinator_agent.register_agent('analysis_agent', self.analysis_agent)
            
            # Initialize guardrails if enabled and available
            if self.config.guardrails.enabled and NEMO_AVAILABLE:
                self._initialize_guardrails()
            
            self.is_initialized = True
            logger.info("All agents initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize agents: {e}")
            return False
    
    def _initialize_guardrails(self):
        """Initialize NeMo Guardrails if available."""
        try:
            if self.config.guardrails.config_path:
                rails_config = RailsConfig.from_path(self.config.guardrails.config_path)
            else:
                # Create default guardrails configuration
                rails_config = RailsConfig.from_content(
                    colang_content=self._get_default_colang_config(),
                    yaml_content=self._get_default_yaml_config()
                )
            
            self.guardrails = LLMRails(rails_config)
            logger.info("NeMo Guardrails initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize guardrails: {e}")
            self.guardrails = None
    
    def _get_default_colang_config(self) -> str:
        """Get default Colang configuration for guardrails."""
        return """
define user express harmful intent
  "generate harmful content"
  "create toxic text"
  "write hate speech"

define bot refuse harmful request
  "I cannot and will not generate harmful, toxic, or inappropriate content."

define flow
  user express harmful intent
  bot refuse harmful request
  stop
"""
    
    def _get_default_yaml_config(self) -> str:
        """Get default YAML configuration for guardrails."""
        return """
models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo

rails:
  input:
    flows:
      - check harmful intent
  output:
    flows:
      - check harmful output
"""
    
    def process_text(self, text: str, task_type: str = "comprehensive_analysis") -> Dict[str, Any]:
        """Process a single text through the multi-agent system."""
        if not self.is_initialized:
            raise RuntimeError("System not initialized. Call initialize_agents() first.")
        
        try:
            # Apply input guardrails if available
            if self.guardrails:
                filtered_text = self._apply_input_guardrails(text)
                if filtered_text != text:
                    logger.warning("Input text was filtered by guardrails")
                    text = filtered_text
            
            # Process through coordinator
            task_data = {
                "task_type": task_type,
                "text": text
            }
            
            result = self.coordinator_agent.process(task_data)
            
            # Apply output guardrails if available
            if self.guardrails and 'synthesis' in result:
                filtered_synthesis = self._apply_output_guardrails(result['synthesis'])
                if filtered_synthesis != result['synthesis']:
                    logger.warning("Output synthesis was filtered by guardrails")
                    result['synthesis'] = filtered_synthesis
            
            self.system_stats["tasks_completed"] += 1
            self.system_stats["total_texts_processed"] += 1
            
            return result
            
        except Exception as e:
            self.system_stats["errors_encountered"] += 1
            logger.error(f"Error processing text: {e}")
            return {
                "error": str(e),
                "timestamp": self._get_timestamp(),
                "status": "error"
            }
    
    def process_dataset(self, dataset_name: str, sample_size: int = 50) -> Dict[str, Any]:
        """Process an entire dataset through the multi-agent system."""
        if not self.is_initialized:
            raise RuntimeError("System not initialized. Call initialize_agents() first.")
        
        try:
            # Verify dataset exists
            available_datasets = self.data_loader.list_available_datasets()
            if dataset_name not in available_datasets:
                return {
                    "error": f"Dataset '{dataset_name}' not found. Available: {available_datasets}",
                    "timestamp": self._get_timestamp()
                }
            
            # Process through coordinator
            task_data = {
                "task_type": "dataset_evaluation",
                "dataset_name": dataset_name,
                "sample_size": sample_size
            }
            
            result = self.coordinator_agent.process(task_data)
            
            self.system_stats["tasks_completed"] += 1
            self.system_stats["total_texts_processed"] += sample_size
            
            return result
            
        except Exception as e:
            self.system_stats["errors_encountered"] += 1
            logger.error(f"Error processing dataset {dataset_name}: {e}")
            return {
                "error": str(e),
                "dataset": dataset_name,
                "timestamp": self._get_timestamp(),
                "status": "error"
            }
    
    def batch_process_texts(self, texts: List[str], task_type: str = "comprehensive_analysis") -> Dict[str, Any]:
        """Process multiple texts in batch."""
        if not self.is_initialized:
            raise RuntimeError("System not initialized. Call initialize_agents() first.")
        
        try:
            # Limit batch size for performance
            max_batch_size = min(len(texts), 20)
            texts_to_process = texts[:max_batch_size]
            
            if task_type == "comprehensive_analysis":
                result = self.coordinator_agent._coordinate_batch_analysis(texts_to_process)
            else:
                # Process individually for other task types
                results = []
                for text in texts_to_process:
                    individual_result = self.process_text(text, task_type)
                    results.append(individual_result)
                
                result = {
                    "batch_results": results,
                    "batch_size": len(texts_to_process),
                    "task_type": task_type,
                    "timestamp": self._get_timestamp()
                }
            
            self.system_stats["tasks_completed"] += 1
            self.system_stats["total_texts_processed"] += len(texts_to_process)
            
            return result
            
        except Exception as e:
            self.system_stats["errors_encountered"] += 1
            logger.error(f"Error in batch processing: {e}")
            return {
                "error": str(e),
                "batch_size": len(texts),
                "timestamp": self._get_timestamp(),
                "status": "error"
            }
    
    def _apply_input_guardrails(self, text: str) -> str:
        """Apply input guardrails to filter harmful content."""
        if not self.guardrails:
            return text
        
        try:
            response = self.guardrails.generate(messages=[{"role": "user", "content": text}])
            return response.get("content", text)
        except Exception as e:
            logger.warning(f"Guardrails input filtering failed: {e}")
            return text
    
    def _apply_output_guardrails(self, text: str) -> str:
        """Apply output guardrails to filter harmful content."""
        if not self.guardrails:
            return text
        
        try:
            # For output filtering, we check if the content would be blocked
            response = self.guardrails.generate(messages=[{"role": "assistant", "content": text}])
            return response.get("content", text)
        except Exception as e:
            logger.warning(f"Guardrails output filtering failed: {e}")
            return text
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        if not self.is_initialized:
            return {"status": "not_initialized"}
        
        status = {
            "system_initialized": self.is_initialized,
            "guardrails_enabled": self.guardrails is not None,
            "available_datasets": self.data_loader.list_available_datasets(),
            "system_stats": self.system_stats.copy(),
            "coordinator_status": self.coordinator_agent.get_system_status() if self.coordinator_agent else None,
            "timestamp": self._get_timestamp()
        }
        
        return status
    
    def get_agent_token_usage(self) -> Dict[str, Dict[str, int]]:
        """Get token usage statistics for all agents."""
        usage = {}
        
        if self.safety_agent:
            usage["safety_agent"] = self.safety_agent.get_token_usage()
        
        if self.analysis_agent:
            usage["analysis_agent"] = self.analysis_agent.get_token_usage()
        
        if self.coordinator_agent:
            usage["coordinator_agent"] = self.coordinator_agent.get_token_usage()
        
        return usage
    
    def reset_all_conversations(self):
        """Reset conversation history for all agents."""
        if self.safety_agent:
            self.safety_agent.reset_conversation()
        
        if self.analysis_agent:
            self.analysis_agent.reset_conversation()
        
        if self.coordinator_agent:
            self.coordinator_agent.reset_conversation()
        
        logger.info("Reset conversation history for all agents")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def shutdown(self):
        """Gracefully shutdown the system."""
        logger.info("Shutting down MultiAgentSystem")
        
        # Log final statistics
        logger.info(f"Final system stats: {self.system_stats}")
        
        # Reset agents
        self.safety_agent = None
        self.analysis_agent = None
        self.coordinator_agent = None
        self.guardrails = None
        self.is_initialized = False
