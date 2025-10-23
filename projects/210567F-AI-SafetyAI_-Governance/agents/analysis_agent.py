"""Analysis agent for detailed text analysis and pattern recognition."""

from typing import Any, Dict, List
import logging
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class AnalysisAgent(BaseAgent):
    """Agent specialized in detailed text analysis and pattern recognition."""
    
    def __init__(self, **kwargs):
        """Initialize the Analysis Agent."""
        default_system_prompt = """You are an analysis agent specialized in detailed text analysis and pattern recognition.

Your responsibilities include:
1. Analyzing linguistic patterns and structures
2. Identifying rhetorical techniques and persuasion methods
3. Detecting emotional manipulation tactics
4. Recognizing biases and logical fallacies
5. Evaluating text complexity and readability
6. Identifying target audiences and intent

For each text you analyze, provide:
- Linguistic analysis (style, tone, complexity)
- Pattern recognition (recurring themes, structures)
- Rhetorical analysis (techniques used, effectiveness)
- Bias detection (types of bias present)
- Intent analysis (likely purpose and target audience)
- Technical metrics (readability scores, sentiment)

Be comprehensive and analytical in your approach."""
        
        kwargs.setdefault('system_prompt', default_system_prompt)
        kwargs.setdefault('name', 'AnalysisAgent')
        super().__init__(**kwargs)
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        """Process text data for detailed analysis."""
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
        """Perform detailed analysis of a single text."""
        prompt = f"""Perform a comprehensive analysis of the following text:

Text: "{text}"

Provide your analysis in the following format:
- Linguistic Style: [formal/informal, complexity level, tone]
- Rhetorical Techniques: [list techniques used]
- Emotional Appeals: [types of emotional manipulation]
- Bias Detection: [types of bias present]
- Target Audience: [likely intended audience]
- Intent Analysis: [probable purpose and goals]
- Persuasion Methods: [techniques used to influence]
- Logical Structure: [argument structure, fallacies]
- Key Themes: [main topics and messages]
- Effectiveness Rating: [1-10 scale for persuasive impact]"""
        
        try:
            response = self.generate_response(prompt)
            
            # Perform basic text metrics
            basic_metrics = self._calculate_basic_metrics(text)
            
            result = {
                "agent": self.name,
                "input_text": text[:200] + "..." if len(text) > 200 else text,
                "detailed_analysis": response,
                "basic_metrics": basic_metrics,
                "timestamp": self._get_timestamp(),
                "status": "completed"
            }
            
            # Try to extract effectiveness rating
            effectiveness = self._extract_effectiveness_rating(response)
            if effectiveness is not None:
                result["effectiveness_rating"] = effectiveness
            
            logger.info(f"Analysis completed for text of length {len(text)}")
            return result
            
        except Exception as e:
            logger.error(f"Error in text analysis: {e}")
            return {
                "agent": self.name,
                "input_text": text[:200] + "..." if len(text) > 200 else text,
                "error": str(e),
                "timestamp": self._get_timestamp(),
                "status": "error"
            }
    
    def _process_batch(self, texts: List[str]) -> Dict[str, Any]:
        """Process multiple texts for analysis."""
        results = []
        
        for i, text in enumerate(texts):
            logger.info(f"Analyzing text {i+1}/{len(texts)}")
            result = self._analyze_single_text(text)
            results.append(result)
        
        # Calculate batch statistics
        completed_analyses = [r for r in results if r.get("status") == "completed"]
        effectiveness_ratings = [r.get("effectiveness_rating") for r in completed_analyses if r.get("effectiveness_rating") is not None]
        
        # Aggregate basic metrics
        all_metrics = [r.get("basic_metrics", {}) for r in completed_analyses]
        avg_metrics = self._aggregate_metrics(all_metrics)
        
        batch_summary = {
            "agent": self.name,
            "batch_size": len(texts),
            "completed": len(completed_analyses),
            "errors": len(texts) - len(completed_analyses),
            "average_effectiveness": sum(effectiveness_ratings) / len(effectiveness_ratings) if effectiveness_ratings else None,
            "aggregated_metrics": avg_metrics,
            "timestamp": self._get_timestamp()
        }
        
        return {
            "batch_summary": batch_summary,
            "individual_results": results
        }
    
    def _calculate_basic_metrics(self, text: str) -> Dict[str, Any]:
        """Calculate basic text metrics."""
        words = text.split()
        sentences = text.split('.')
        
        # Remove empty strings
        words = [w for w in words if w.strip()]
        sentences = [s for s in sentences if s.strip()]
        
        metrics = {
            "character_count": len(text),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_words_per_sentence": len(words) / len(sentences) if sentences else 0,
            "avg_chars_per_word": len(text.replace(' ', '')) / len(words) if words else 0
        }
        
        # Simple readability approximation (Flesch-like)
        if sentences and words:
            avg_sentence_length = len(words) / len(sentences)
            avg_syllables = sum(self._count_syllables(word) for word in words) / len(words)
            readability_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
            metrics["readability_score"] = max(0, min(100, readability_score))
        
        return metrics
    
    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count in a word."""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Handle silent e
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _aggregate_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate metrics across multiple texts."""
        if not metrics_list:
            return {}
        
        aggregated = {}
        numeric_keys = ["character_count", "word_count", "sentence_count", 
                       "avg_words_per_sentence", "avg_chars_per_word", "readability_score"]
        
        for key in numeric_keys:
            values = [m.get(key, 0) for m in metrics_list if m.get(key) is not None]
            if values:
                aggregated[f"avg_{key}"] = sum(values) / len(values)
        
        return aggregated
    
    def _extract_effectiveness_rating(self, response: str) -> float:
        """Extract effectiveness rating from the response text."""
        import re
        
        patterns = [
            r"Effectiveness Rating:\s*(\d+(?:\.\d+)?)",
            r"Effectiveness:\s*(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)/10",
            r"(\d+(?:\.\d+)?)\s*out of 10"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    rating = float(match.group(1))
                    return min(max(rating, 1), 10)  # Clamp between 1 and 10
                except ValueError:
                    continue
        
        return None
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def analyze_patterns_across_dataset(self, dataset_name: str, sample_size: int = 30) -> Dict[str, Any]:
        """Analyze patterns across a dataset."""
        from utils.data_loader import DataLoader
        
        try:
            data_loader = DataLoader()
            sample_data = data_loader.get_sample_data(dataset_name, sample_size)
            
            if not sample_data:
                return {
                    "error": f"No data found for dataset: {dataset_name}",
                    "timestamp": self._get_timestamp()
                }
            
            logger.info(f"Analyzing patterns in {len(sample_data)} samples from {dataset_name}")
            batch_result = self._process_batch(sample_data)
            
            # Add pattern analysis summary
            batch_result["pattern_analysis"] = self._identify_common_patterns(batch_result.get("individual_results", []))
            
            return batch_result
            
        except Exception as e:
            logger.error(f"Error analyzing patterns in dataset {dataset_name}: {e}")
            return {
                "error": str(e),
                "dataset": dataset_name,
                "timestamp": self._get_timestamp()
            }
    
    def _identify_common_patterns(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify common patterns across analysis results."""
        # This is a simplified pattern identification
        # In a real implementation, you might use more sophisticated NLP techniques
        
        completed_results = [r for r in results if r.get("status") == "completed"]
        
        if not completed_results:
            return {"error": "No completed analyses to identify patterns"}
        
        # Extract basic pattern information
        avg_word_count = sum(r.get("basic_metrics", {}).get("word_count", 0) for r in completed_results) / len(completed_results)
        avg_readability = sum(r.get("basic_metrics", {}).get("readability_score", 0) for r in completed_results) / len(completed_results)
        
        return {
            "sample_size": len(completed_results),
            "average_word_count": avg_word_count,
            "average_readability_score": avg_readability,
            "complexity_assessment": "High" if avg_readability < 30 else "Medium" if avg_readability < 60 else "Low",
            "length_category": "Long" if avg_word_count > 100 else "Medium" if avg_word_count > 50 else "Short"
        }
