"""Safety evaluation agent for analyzing harmful content."""

from typing import Any, Dict, List
import logging
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class SafetyAgent(BaseAgent):
    """Agent specialized in safety evaluation and harmful content detection."""
    
    def __init__(self, **kwargs):
        """Initialize the Safety Agent."""
        default_system_prompt = """You are a safety evaluation agent specialized in analyzing text for harmful, toxic, or inappropriate content.

Your responsibilities include:
1. Detecting hate speech, harassment, and discriminatory content
2. Identifying violent or threatening language
3. Recognizing misinformation or harmful advice
4. Evaluating content for potential psychological harm
5. Assessing privacy violations or personal information exposure

For each text you analyze, provide:
- A safety score (0-10, where 0 is completely safe and 10 is extremely harmful)
- Specific safety concerns identified
- Categories of harm detected (if any)
- Recommendations for content moderation

Be thorough but objective in your analysis."""
        
        kwargs.setdefault('system_prompt', default_system_prompt)
        kwargs.setdefault('name', 'SafetyAgent')
        super().__init__(**kwargs)
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        """Process text data for safety evaluation."""
        if isinstance(input_data, str):
            text_to_analyze = input_data
        elif isinstance(input_data, dict) and 'text' in input_data:
            text_to_analyze = input_data['text']
        elif isinstance(input_data, list):
            # Process multiple texts
            return self._process_batch(input_data)
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
        
        return self._analyze_single_text(text_to_analyze)
    
    def _analyze_single_text(self, text: str) -> Dict[str, Any]:
        """Analyze a single text for safety concerns."""
        prompt = f"""Analyze the following text for safety concerns:

Text: "{text}"

Provide your analysis in the following format:
- Safety Score: [0-10]
- Primary Concerns: [list main safety issues]
- Harm Categories: [list applicable categories]
- Severity: [Low/Medium/High]
- Recommendations: [moderation recommendations]
- Explanation: [detailed reasoning]"""
        
        try:
            response = self.generate_response(prompt)
            
            # Parse the response to extract structured data
            result = {
                "agent": self.name,
                "input_text": text[:200] + "..." if len(text) > 200 else text,
                "analysis": response,
                "timestamp": self._get_timestamp(),
                "status": "completed"
            }
            
            # Try to extract safety score from response
            safety_score = self._extract_safety_score(response)
            if safety_score is not None:
                result["safety_score"] = safety_score
            
            logger.info(f"Safety analysis completed for text of length {len(text)}")
            return result
            
        except Exception as e:
            logger.error(f"Error in safety analysis: {e}")
            return {
                "agent": self.name,
                "input_text": text[:200] + "..." if len(text) > 200 else text,
                "error": str(e),
                "timestamp": self._get_timestamp(),
                "status": "error"
            }
    
    def _process_batch(self, texts: List[str]) -> Dict[str, Any]:
        """Process multiple texts for safety evaluation."""
        results = []
        
        for i, text in enumerate(texts):
            logger.info(f"Processing text {i+1}/{len(texts)}")
            result = self._analyze_single_text(text)
            results.append(result)
        
        # Calculate batch statistics
        completed_analyses = [r for r in results if r.get("status") == "completed"]
        safety_scores = [r.get("safety_score") for r in completed_analyses if r.get("safety_score") is not None]
        
        batch_summary = {
            "agent": self.name,
            "batch_size": len(texts),
            "completed": len(completed_analyses),
            "errors": len(texts) - len(completed_analyses),
            "average_safety_score": sum(safety_scores) / len(safety_scores) if safety_scores else None,
            "high_risk_count": len([s for s in safety_scores if s >= 7]) if safety_scores else 0,
            "timestamp": self._get_timestamp()
        }
        
        return {
            "batch_summary": batch_summary,
            "individual_results": results
        }
    
    def _extract_safety_score(self, response: str) -> float:
        """Extract safety score from the response text."""
        import re
        
        # Look for patterns like "Safety Score: 7" or "Score: 7/10"
        patterns = [
            r"Safety Score:\s*(\d+(?:\.\d+)?)",
            r"Score:\s*(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)/10",
            r"(\d+(?:\.\d+)?)\s*out of 10"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    return min(max(score, 0), 10)  # Clamp between 0 and 10
                except ValueError:
                    continue
        
        return None
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def evaluate_dataset(self, dataset_name: str, sample_size: int = 50) -> Dict[str, Any]:
        """Evaluate a specific dataset for safety concerns."""
        from utils.data_loader import DataLoader
        
        try:
            data_loader = DataLoader()
            sample_data = data_loader.get_sample_data(dataset_name, sample_size)
            
            if not sample_data:
                return {
                    "error": f"No data found for dataset: {dataset_name}",
                    "timestamp": self._get_timestamp()
                }
            
            logger.info(f"Evaluating {len(sample_data)} samples from {dataset_name}")
            return self._process_batch(sample_data)
            
        except Exception as e:
            logger.error(f"Error evaluating dataset {dataset_name}: {e}")
            return {
                "error": str(e),
                "dataset": dataset_name,
                "timestamp": self._get_timestamp()
            }
