import os
import time
import requests
import numpy as np
from typing import List, Dict, Optional
import logging
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)

class PerspectiveAPIClient:
    """Client for Google's Perspective API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Perspective API client.
        
        Args:
            api_key: Perspective API key. If None, will try to get from environment.
        """
        self.api_key = api_key or os.getenv('PERSPECTIVE_API_KEY')
        if not self.api_key:
            logger.warning("No Perspective API key provided. Will use mock scores.")
            self.use_mock = True
        else:
            self.use_mock = False
            
        self.base_url = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
        
        # Rate limiting
        self.requests_per_second = 1  # Perspective API rate limit
        self.last_request_time = 0
    
    def _wait_for_rate_limit(self):
        """Ensure we don't exceed rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < 1.0 / self.requests_per_second:
            time.sleep(1.0 / self.requests_per_second - time_since_last)
        self.last_request_time = time.time()
    
    def get_toxicity_score(self, text: str, attributes: List[str] = None) -> Dict[str, float]:
        """
        Get toxicity scores from Perspective API.
        
        Args:
            text: Text to analyze
            attributes: List of attributes to analyze (default: ['TOXICITY'])
            
        Returns:
            Dictionary with attribute scores
        """
        if attributes is None:
            attributes = ['TOXICITY']
        
        if self.use_mock:
            return self._get_mock_scores(text, attributes)
        
        self._wait_for_rate_limit()
        
        # Prepare request
        data = {
            'comment': {'text': text},
            'requestedAttributes': {attr: {} for attr in attributes},
            'languages': ['en']
        }
        
        try:
            response = requests.post(
                f"{self.base_url}?key={self.api_key}",
                data=json.dumps(data),
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                scores = {}
                
                for attr in attributes:
                    if attr in result.get('attributeScores', {}):
                        scores[attr.lower()] = result['attributeScores'][attr]['summaryScore']['value']
                    else:
                        scores[attr.lower()] = 0.0
                        
                return scores
            else:
                logger.error(f"Perspective API error: {response.status_code} - {response.text}")
                return {attr.lower(): 0.0 for attr in attributes}
                
        except Exception as e:
            logger.error(f"Error calling Perspective API: {e}")
            return {attr.lower(): 0.0 for attr in attributes}
    
    def _get_mock_scores(self, text: str, attributes: List[str]) -> Dict[str, float]:
        """
        Generate mock Perspective API scores for testing.
        This simulates the biased behavior mentioned in your research.
        """
        # Simulate biased scoring patterns
        np.random.seed(hash(text) % 2**32)  # Deterministic based on text
        
        base_score = np.random.beta(2, 5)  # Generally low toxicity
        
        # Simulate bias against certain patterns (mentioned in your research)
        bias_keywords = ['sick', 'bad', 'crazy', 'wild', 'insane']
        identity_keywords = ['black', 'white', 'gay', 'christian', 'muslim', 'jewish']
        
        # Over-flag identity-related content
        if any(keyword in text.lower() for keyword in identity_keywords):
            base_score += np.random.uniform(0.1, 0.3)
        
        # Misinterpret slang/sarcasm
        if any(keyword in text.lower() for keyword in bias_keywords):
            base_score += np.random.uniform(0.05, 0.2)
        
        # Context insensitivity - miss sarcasm indicators
        if '"' in text or '!' in text or 'yeah right' in text.lower():
            base_score += np.random.uniform(0.1, 0.25)  # Should recognize sarcasm but doesn't
        
        base_score = min(base_score, 1.0)
        
        return {attr.lower(): base_score for attr in attributes}
    
    def batch_analyze(self, texts: List[str], batch_size: int = 10) -> List[Dict[str, float]]:
        """
        Analyze multiple texts with batching and error handling.
        
        Args:
            texts: List of texts to analyze
            batch_size: Number of texts to process at once
            
        Returns:
            List of score dictionaries
        """
        results = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing with Perspective API"):
            batch = texts[i:i + batch_size]
            
            for text in batch:
                try:
                    scores = self.get_toxicity_score(text)
                    results.append(scores)
                except Exception as e:
                    logger.error(f"Error processing text: {e}")
                    results.append({'toxicity': 0.0})
        
        return results