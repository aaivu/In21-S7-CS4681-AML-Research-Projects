"""OverseerAgent for routing queries to task-specific agents with safety scanning."""

import logging
import json
from typing import Dict, Any, List, Optional, Union
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from .base_agent import BaseAgent
from .task_agents import SummarizationAgent, TranslationAgent, QAAgent
from .safety_tools import create_safety_tools, SafetyScanTool
from utils.config import Config

logger = logging.getLogger(__name__)


class QueryRoutingInput(BaseModel):
    """Input schema for query routing."""
    query: str = Field(description="User query to route to appropriate agent")
    task_type: Optional[str] = Field(default=None, description="Optional explicit task type")


class TaskRoutingTool(BaseTool):
    """Tool for routing queries to appropriate task-specific agents."""
    
    name: str = "task_router"
    description: str = """Route user queries to the appropriate task-specific agent.
    Supports: summarization, translation, question_answering (QA).
    Input should be the user query and optional task type."""
    args_schema: type[BaseModel] = QueryRoutingInput
    
    def __init__(self, task_agents: Dict[str, BaseAgent], **kwargs):
        """Initialize the task routing tool."""
        super().__init__(**kwargs)
        self.task_agents = task_agents
    
    def _run(self, query: str, task_type: Optional[str] = None) -> str:
        """Route query to appropriate agent."""
        try:
            # Determine task type if not provided
            if not task_type:
                task_type = self._classify_query(query)
            
            # Route to appropriate agent
            if task_type in self.task_agents:
                agent = self.task_agents[task_type]
                result = agent.process(query)
                
                if result.get('status') == 'completed':
                    return self._format_agent_response(result)
                else:
                    return f"Error processing {task_type} request: {result.get('error', 'Unknown error')}"
            else:
                return f"Unsupported task type: {task_type}. Supported types: {list(self.task_agents.keys())}"
                
        except Exception as e:
            logger.error(f"Error in task routing: {e}")
            return f"Error routing query: {str(e)}"
    
    def _classify_query(self, query: str) -> str:
        """Classify query to determine appropriate task type."""
        query_lower = query.lower()
        
        # Simple keyword-based classification
        if any(word in query_lower for word in ['summarize', 'summary', 'tldr', 'brief']):
            return 'summarization'
        elif any(word in query_lower for word in ['translate', 'translation', 'convert to']):
            return 'translation'
        elif any(word in query_lower for word in ['what', 'how', 'why', 'when', 'where', 'who', '?']):
            return 'question_answering'
        else:
            # Default to QA for unclear queries
            return 'question_answering'
    
    def _format_agent_response(self, result: Dict[str, Any]) -> str:
        """Format agent response for output."""
        task_type = result.get('task_type', 'unknown')
        
        if task_type == 'summarization':
            return f"Summary: {result.get('summary', 'No summary available')}"
        elif task_type == 'translation':
            return f"Translation ({result.get('target_language', 'unknown')}): {result.get('translation', 'No translation available')}"
        elif task_type == 'question_answering':
            return f"Answer: {result.get('answer', 'No answer available')}"
        else:
            return str(result)


class OverseerAgent(BaseAgent):
    """Overseer agent that routes queries with safety scanning."""
    
    def __init__(self, config: Optional[Config] = None, **kwargs):
        """Initialize the Overseer Agent."""
        self.config = config or Config()
        
        default_system_prompt = """You are an Overseer Agent responsible for safely routing user queries to specialized task agents.

Your responsibilities include:
1. Analyzing user queries to determine the appropriate task type
2. Ensuring all content is safe before processing
3. Routing queries to specialized agents (summarization, translation, QA)
4. Coordinating responses and ensuring output safety
5. Refusing to process harmful or inappropriate content

Available task types:
- summarization: For creating summaries of text content
- translation: For translating text between languages  
- question_answering: For answering questions and providing information

Safety Protocol:
- Always scan content for harmful elements before processing
- Refuse requests that promote violence, exploitation, or harmful activities
- Ensure all outputs are safe and appropriate
- Log all safety decisions for audit purposes

Use the available tools to scan content safety and route to appropriate agents."""
        
        kwargs.setdefault('system_prompt', default_system_prompt)
        kwargs.setdefault('name', 'OverseerAgent')
        super().__init__(**kwargs)
        
        # Initialize task-specific agents
        self.task_agents = self._initialize_task_agents()
        
        # Initialize tools
        self.safety_tools = create_safety_tools(self.config.guardrails.rails_config if hasattr(self.config, 'guardrails') else None)
        self.routing_tool = TaskRoutingTool(task_agents=self.task_agents)
        
        # Initialize AgentExecutor
        self.agent_executor = self._initialize_agent_executor()
        
        # Safety tracking
        self.safety_log = []
        self.blocked_requests = 0
        self.processed_requests = 0
        
        logger.info("OverseerAgent initialized with safety scanning and task routing")
    
    def _initialize_task_agents(self) -> Dict[str, BaseAgent]:
        """Initialize task-specific agents."""
        agents = {}
        
        try:
            agents['summarization'] = SummarizationAgent(
                model_name=self.model_name,
                temperature=0.3,  # More deterministic for summaries
                api_key=getattr(self, 'api_key', None)
            )
            
            agents['translation'] = TranslationAgent(
                model_name=self.model_name,
                temperature=0.2,  # Very deterministic for translations
                api_key=getattr(self, 'api_key', None)
            )
            
            agents['question_answering'] = QAAgent(
                model_name=self.model_name,
                temperature=0.7,  # Balanced for QA
                api_key=getattr(self, 'api_key', None)
            )
            
            logger.info(f"Initialized {len(agents)} task-specific agents")
            
        except Exception as e:
            logger.error(f"Error initializing task agents: {e}")
        
        return agents
    
    def _initialize_agent_executor(self) -> AgentExecutor:
        """Initialize LangChain AgentExecutor with tools."""
        try:
            # Combine all tools
            all_tools = self.safety_tools + [self.routing_tool]
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            # Create agent
            agent = create_openai_functions_agent(
                llm=self.llm,
                tools=all_tools,
                prompt=prompt
            )
            
            # Create executor
            executor = AgentExecutor(
                agent=agent,
                tools=all_tools,
                verbose=True,
                max_iterations=5,
                early_stopping_method="generate"
            )
            
            logger.info("AgentExecutor initialized successfully")
            return executor
            
        except Exception as e:
            logger.error(f"Error initializing AgentExecutor: {e}")
            raise
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        """Process user query with safety scanning and routing."""
        if isinstance(input_data, str):
            query = input_data
            task_type = None
        elif isinstance(input_data, dict):
            query = input_data.get('query', '')
            task_type = input_data.get('task_type', None)
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
        
        return self._process_query_safely(query, task_type)
    
    def _process_query_safely(self, query: str, task_type: Optional[str] = None) -> Dict[str, Any]:
        """Process query with comprehensive safety checks."""
        try:
            self.processed_requests += 1
            
            # Step 1: Initial safety scan
            safety_result = self._perform_safety_scan(query)
            
            if not safety_result['safe']:
                return self._handle_unsafe_content(query, safety_result)
            
            # Step 2: Process query through agent executor
            executor_input = {
                "input": f"Query: {query}" + (f"\nTask type: {task_type}" if task_type else "")
            }
            
            result = self.agent_executor.invoke(executor_input)
            
            # Step 3: Safety scan output
            output_text = result.get('output', '')
            output_safety = self._perform_safety_scan(output_text)
            
            if not output_safety['safe']:
                return self._handle_unsafe_output(query, output_text, output_safety)
            
            # Step 4: Return safe result
            return {
                "agent": self.name,
                "query": query,
                "task_type": task_type,
                "response": output_text,
                "safety_checks": {
                    "input_scan": safety_result,
                    "output_scan": output_safety
                },
                "timestamp": self._get_timestamp(),
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error processing query safely: {e}")
            return {
                "agent": self.name,
                "query": query,
                "error": str(e),
                "timestamp": self._get_timestamp(),
                "status": "error"
            }
    
    def _perform_safety_scan(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive safety scan on text."""
        try:
            # Use the safety scanning tool
            safety_tool = next((tool for tool in self.safety_tools if isinstance(tool, SafetyScanTool)), None)
            
            if safety_tool:
                scan_result = safety_tool._run(text, "comprehensive")
                
                # Parse the result
                safe = "Safe to process: True" in scan_result
                risk_level = "Low"
                
                if "Risk Level: High" in scan_result:
                    risk_level = "High"
                elif "Risk Level: Medium" in scan_result:
                    risk_level = "Medium"
                
                return {
                    "safe": safe,
                    "risk_level": risk_level,
                    "scan_result": scan_result,
                    "method": "safety_tool"
                }
            else:
                # Fallback safety check
                return self._fallback_safety_check(text)
                
        except Exception as e:
            logger.error(f"Error in safety scan: {e}")
            return {
                "safe": False,
                "risk_level": "Unknown",
                "error": str(e),
                "method": "error"
            }
    
    def _fallback_safety_check(self, text: str) -> Dict[str, Any]:
        """Fallback safety check using simple prompt."""
        safety_prompt = f"""Detect if this text promotes violence or exploitation, contains hate speech, or includes other harmful content.

Text: "{text}"

Respond with only: SAFE or UNSAFE"""
        
        try:
            response = self.generate_response(safety_prompt)
            safe = "SAFE" in response.upper()
            
            return {
                "safe": safe,
                "risk_level": "High" if not safe else "Low",
                "scan_result": response,
                "method": "fallback_prompt"
            }
            
        except Exception as e:
            logger.error(f"Fallback safety check failed: {e}")
            return {
                "safe": False,
                "risk_level": "Unknown",
                "error": str(e),
                "method": "fallback_error"
            }
    
    def _handle_unsafe_content(self, query: str, safety_result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle unsafe input content."""
        self.blocked_requests += 1
        
        safety_log_entry = {
            "timestamp": self._get_timestamp(),
            "query": query[:100] + "..." if len(query) > 100 else query,
            "reason": "Unsafe input content",
            "safety_result": safety_result,
            "action": "blocked"
        }
        self.safety_log.append(safety_log_entry)
        
        logger.warning(f"Blocked unsafe query: {query[:50]}...")
        
        return {
            "agent": self.name,
            "query": query[:100] + "..." if len(query) > 100 else query,
            "response": "I cannot process this request as it appears to contain harmful content that violates safety guidelines. Please rephrase your request with appropriate content.",
            "safety_violation": True,
            "safety_details": safety_result,
            "timestamp": self._get_timestamp(),
            "status": "blocked"
        }
    
    def _handle_unsafe_output(self, query: str, output: str, safety_result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle unsafe output content."""
        safety_log_entry = {
            "timestamp": self._get_timestamp(),
            "query": query[:100] + "..." if len(query) > 100 else query,
            "output": output[:100] + "..." if len(output) > 100 else output,
            "reason": "Unsafe output content",
            "safety_result": safety_result,
            "action": "output_blocked"
        }
        self.safety_log.append(safety_log_entry)
        
        logger.warning(f"Blocked unsafe output for query: {query[:50]}...")
        
        return {
            "agent": self.name,
            "query": query,
            "response": "I cannot provide the requested response as it contains content that violates safety guidelines. Please try rephrasing your request.",
            "safety_violation": True,
            "safety_details": safety_result,
            "timestamp": self._get_timestamp(),
            "status": "output_blocked"
        }
    
    def get_safety_statistics(self) -> Dict[str, Any]:
        """Get safety statistics and logs."""
        return {
            "total_requests": self.processed_requests,
            "blocked_requests": self.blocked_requests,
            "safety_rate": (self.processed_requests - self.blocked_requests) / max(self.processed_requests, 1),
            "recent_safety_log": self.safety_log[-10:],  # Last 10 entries
            "total_safety_events": len(self.safety_log)
        }
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status."""
        return {
            "agent": self.name,
            "available_task_agents": list(self.task_agents.keys()),
            "safety_tools_count": len(self.safety_tools),
            "agent_executor_initialized": self.agent_executor is not None,
            "safety_statistics": self.get_safety_statistics(),
            "timestamp": self._get_timestamp()
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def reset_safety_log(self):
        """Reset the safety log (for testing or maintenance)."""
        self.safety_log.clear()
        self.blocked_requests = 0
        self.processed_requests = 0
        logger.info("Safety log reset")
    
    def export_safety_log(self, filepath: str):
        """Export safety log to file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.safety_log, f, indent=2, ensure_ascii=False)
            logger.info(f"Safety log exported to {filepath}")
        except Exception as e:
            logger.error(f"Error exporting safety log: {e}")
            raise
