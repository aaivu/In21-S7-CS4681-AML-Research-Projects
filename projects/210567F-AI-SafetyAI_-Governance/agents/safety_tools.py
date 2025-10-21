"""Safety scanning tools using NeMo Guardrails and custom safety checks."""

import logging
from typing import Dict, Any, Optional, List
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

# Optional NeMo Guardrails integration
try:
    from nemoguardrails import RailsConfig, LLMRails
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    logging.warning("NeMo Guardrails not available. Using fallback safety checks.")

logger = logging.getLogger(__name__)


class SafetyScanInput(BaseModel):
    """Input schema for safety scanning tool."""
    text: str = Field(description="Text to scan for harmful content")
    scan_type: str = Field(default="comprehensive", description="Type of safety scan to perform")


class SafetyScanTool(BaseTool):
    """Custom LangChain tool for safety scanning using NeMo Guardrails."""
    
    name: str = "safety_scanner"
    description: str = """Use this tool to scan text for harmful content before processing. 
    Input should be the text to scan. Returns a safety assessment with risk level and details."""
    args_schema: type[BaseModel] = SafetyScanInput
    
    def __init__(self, guardrails_config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize the safety scanning tool."""
        super().__init__(**kwargs)
        self.guardrails = None
        self.safety_prompts = self._get_safety_prompts()
        
        if NEMO_AVAILABLE and guardrails_config:
            self._initialize_guardrails(guardrails_config)
    
    def _initialize_guardrails(self, config: Dict[str, Any]):
        """Initialize NeMo Guardrails if available."""
        try:
            if config.get('config_path'):
                rails_config = RailsConfig.from_path(config['config_path'])
            else:
                rails_config = RailsConfig.from_content(
                    colang_content=self._get_default_colang_config(),
                    yaml_content=self._get_default_yaml_config()
                )
            
            self.guardrails = LLMRails(rails_config)
            logger.info("NeMo Guardrails initialized for safety scanning")
            
        except Exception as e:
            logger.error(f"Failed to initialize guardrails: {e}")
            self.guardrails = None
    
    def _get_default_colang_config(self) -> str:
        """Get default Colang configuration for safety scanning."""
        return """
define user express harmful intent
  "generate harmful content"
  "create violent content"
  "write hate speech"
  "promote violence"
  "encourage exploitation"
  "create discriminatory content"

define user ask for illegal content
  "how to make weapons"
  "illegal activities"
  "harmful instructions"

define bot refuse harmful request
  "I cannot and will not generate harmful, violent, or inappropriate content."

define bot explain safety
  "This content appears to contain harmful elements that violate safety guidelines."

define flow safety_check
  user express harmful intent
  bot refuse harmful request
  stop

define flow illegal_content_check
  user ask for illegal content
  bot refuse harmful request
  stop
"""
    
    def _get_default_yaml_config(self) -> str:
        """Get default YAML configuration for safety scanning."""
        return """
models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo

rails:
  input:
    flows:
      - safety_check
      - illegal_content_check
  output:
    flows:
      - safety_check
"""
    
    def _get_safety_prompts(self) -> Dict[str, str]:
        """Get safety scanning prompts for different types of checks."""
        return {
            "violence": """Analyze the following text and determine if it promotes, encourages, or contains explicit violence or threats. 
            Look for content that could incite violence, contains graphic violent descriptions, or promotes harm to individuals or groups.
            
            Text: "{text}"
            
            Assessment: Does this text promote violence or contain violent threats? (Yes/No)
            Risk Level: (Low/Medium/High)
            Explanation: [Brief explanation of the assessment]""",
            
            "exploitation": """Analyze the following text and determine if it promotes exploitation, abuse, or harmful manipulation of individuals or groups.
            Look for content that could encourage exploitation, contains instructions for harmful activities, or promotes unethical behavior.
            
            Text: "{text}"
            
            Assessment: Does this text promote exploitation or harmful manipulation? (Yes/No)
            Risk Level: (Low/Medium/High)
            Explanation: [Brief explanation of the assessment]""",
            
            "hate_speech": """Analyze the following text and determine if it contains hate speech, discriminatory language, or promotes prejudice against individuals or groups.
            Look for content that targets people based on race, gender, religion, nationality, or other protected characteristics.
            
            Text: "{text}"
            
            Assessment: Does this text contain hate speech or discriminatory content? (Yes/No)
            Risk Level: (Low/Medium/High)
            Explanation: [Brief explanation of the assessment]""",
            
            "comprehensive": """Detect if this text promotes violence or exploitation, contains hate speech, or includes other harmful content.
            Analyze for: violence, threats, exploitation, discrimination, illegal activities, harmful instructions, or inappropriate content.
            
            Text: "{text}"
            
            Safety Assessment:
            - Contains harmful content: (Yes/No)
            - Primary concerns: [List main safety issues]
            - Risk level: (Low/Medium/High)
            - Safe to process: (Yes/No)
            - Explanation: [Detailed reasoning for the assessment]"""
        }
    
    def _run(self, text: str, scan_type: str = "comprehensive") -> str:
        """Run the safety scanning tool."""
        try:
            # First try NeMo Guardrails if available
            if self.guardrails:
                guardrails_result = self._scan_with_guardrails(text)
                if guardrails_result.get('blocked', False):
                    return self._format_safety_result({
                        "safe": False,
                        "risk_level": "High",
                        "concerns": ["Content blocked by guardrails"],
                        "explanation": guardrails_result.get('reason', 'Content violates safety guidelines'),
                        "method": "nemo_guardrails"
                    })
            
            # Fallback to prompt-based safety checking
            prompt_result = self._scan_with_prompts(text, scan_type)
            return self._format_safety_result(prompt_result)
            
        except Exception as e:
            logger.error(f"Error in safety scanning: {e}")
            return self._format_safety_result({
                "safe": False,
                "risk_level": "Unknown",
                "concerns": ["Safety scan failed"],
                "explanation": f"Error during safety scan: {str(e)}",
                "method": "error"
            })
    
    def _scan_with_guardrails(self, text: str) -> Dict[str, Any]:
        """Scan text using NeMo Guardrails."""
        try:
            response = self.guardrails.generate(
                messages=[{"role": "user", "content": text}]
            )
            
            # Check if content was blocked
            if hasattr(response, 'blocked') and response.blocked:
                return {
                    "blocked": True,
                    "reason": getattr(response, 'reason', 'Content violates safety guidelines')
                }
            
            return {"blocked": False}
            
        except Exception as e:
            logger.warning(f"Guardrails scanning failed: {e}")
            return {"blocked": False, "error": str(e)}
    
    def _scan_with_prompts(self, text: str, scan_type: str) -> Dict[str, Any]:
        """Scan text using prompt-based safety checking."""
        from langchain.chat_models import ChatOpenAI
        from langchain.schema import HumanMessage
        
        try:
            # Initialize LLM for safety checking
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)
            
            # Get appropriate prompt
            prompt_template = self.safety_prompts.get(scan_type, self.safety_prompts["comprehensive"])
            prompt = prompt_template.format(text=text)
            
            # Generate safety assessment
            response = llm([HumanMessage(content=prompt)])
            assessment = response.content
            
            # Parse the response
            return self._parse_safety_assessment(assessment, text)
            
        except Exception as e:
            logger.error(f"Prompt-based safety scanning failed: {e}")
            return {
                "safe": False,
                "risk_level": "Unknown",
                "concerns": ["Safety assessment failed"],
                "explanation": f"Could not complete safety assessment: {str(e)}",
                "method": "prompt_error"
            }
    
    def _parse_safety_assessment(self, assessment: str, original_text: str) -> Dict[str, Any]:
        """Parse the safety assessment response."""
        assessment_lower = assessment.lower()
        
        # Determine if content is safe
        contains_harmful = any(phrase in assessment_lower for phrase in [
            "yes", "contains harmful", "promotes violence", "promotes exploitation",
            "hate speech", "discriminatory", "unsafe", "harmful content: yes"
        ])
        
        # Extract risk level
        risk_level = "Low"
        if "high" in assessment_lower:
            risk_level = "High"
        elif "medium" in assessment_lower:
            risk_level = "Medium"
        
        # Extract concerns
        concerns = []
        if "violence" in assessment_lower:
            concerns.append("Violence/Threats")
        if "exploitation" in assessment_lower:
            concerns.append("Exploitation")
        if "hate speech" in assessment_lower or "discriminatory" in assessment_lower:
            concerns.append("Hate Speech/Discrimination")
        if "illegal" in assessment_lower:
            concerns.append("Illegal Activities")
        
        return {
            "safe": not contains_harmful,
            "risk_level": risk_level,
            "concerns": concerns if concerns else ["No specific concerns identified"],
            "explanation": assessment,
            "method": "prompt_based",
            "original_text_length": len(original_text)
        }
    
    def _format_safety_result(self, result: Dict[str, Any]) -> str:
        """Format the safety scanning result."""
        return f"""Safety Scan Result:
Safe to process: {result.get('safe', False)}
Risk Level: {result.get('risk_level', 'Unknown')}
Primary Concerns: {', '.join(result.get('concerns', []))}
Method: {result.get('method', 'unknown')}

Explanation: {result.get('explanation', 'No explanation provided')}"""


class ContentFilterTool(BaseTool):
    """Tool for filtering and cleaning content before processing."""
    
    name: str = "content_filter"
    description: str = """Use this tool to filter and clean content before processing.
    Removes potentially harmful elements while preserving safe content."""
    args_schema: type[BaseModel] = SafetyScanInput
    
    def _run(self, text: str, scan_type: str = "comprehensive") -> str:
        """Filter content and return cleaned version."""
        try:
            # Basic content filtering
            filtered_text = self._apply_basic_filters(text)
            
            # Check if significant filtering occurred
            if len(filtered_text) < len(text) * 0.5:
                return "Content heavily filtered due to safety concerns. Original content may contain harmful elements."
            
            return filtered_text
            
        except Exception as e:
            logger.error(f"Error in content filtering: {e}")
            return "Content could not be safely filtered."
    
    def _apply_basic_filters(self, text: str) -> str:
        """Apply basic content filters."""
        # This is a simplified implementation
        # In production, you'd use more sophisticated filtering
        
        harmful_patterns = [
            r'\b(kill|murder|violence)\b',
            r'\b(hate|discriminat\w+)\b',
            r'\b(exploit\w+|abuse)\b'
        ]
        
        import re
        filtered_text = text
        
        for pattern in harmful_patterns:
            filtered_text = re.sub(pattern, '[FILTERED]', filtered_text, flags=re.IGNORECASE)
        
        return filtered_text


def create_safety_tools(guardrails_config: Optional[Dict[str, Any]] = None) -> List[BaseTool]:
    """Create and return safety scanning tools."""
    tools = [
        SafetyScanTool(guardrails_config=guardrails_config),
        ContentFilterTool()
    ]
    
    logger.info(f"Created {len(tools)} safety tools")
    return tools
