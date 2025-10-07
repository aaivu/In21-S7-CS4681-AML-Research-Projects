#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM-based paraphrasing for financial text data augmentation using GPT models.
"""

import os
import requests
import json
import time
from typing import List, Dict, Union, Optional
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class LLMParaphraser:
    """Class for LLM-based paraphrasing using OpenAI's API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        retry_delay: int = 2,
    ):
        """
        Initialize the LLM paraphraser.

        Args:
            api_key: OpenAI API key (will use from environment if not provided)
            model: Model to use for paraphrasing
            max_retries: Maximum number of retries for API calls
            retry_delay: Delay in seconds between retries
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No API key provided. LLM paraphrasing will not work.")

        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.api_url = "https://api.openai.com/v1/chat/completions"

    def get_paraphrase_prompt(self, text: str, sentiment: str) -> str:
        """
        Create a prompt for paraphrasing with sentiment preservation.

        Args:
            text: Text to paraphrase
            sentiment: Sentiment of the text (positive, negative, neutral)

        Returns:
            str: Prompt for the LLM
        """
        return (
            f"Paraphrase the following financial text while preserving its {sentiment} sentiment. "
            f"Maintain the same financial meaning but use different wording. Keep the same level of formality. "
            f"Response should only contain the paraphrased text without any additional explanation or formatting.\n\n"
            f"Original: {text}\n\n"
            f"Paraphrased:"
        )

    def paraphrase_text(self, text: str, sentiment: str = "neutral") -> Optional[str]:
        """
        Paraphrase a text while preserving sentiment.

        Args:
            text: Text to paraphrase
            sentiment: Sentiment to preserve (positive, negative, neutral)

        Returns:
            Optional[str]: Paraphrased text or None if API call fails
        """
        if not self.api_key:
            logger.error("No API key available. Cannot paraphrase.")
            return None

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        prompt = self.get_paraphrase_prompt(text, sentiment)

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 150,
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url, headers=headers, data=json.dumps(payload)
                )

                if response.status_code == 200:
                    data = response.json()
                    paraphrased_text = data["choices"][0]["message"]["content"].strip()
                    return paraphrased_text
                else:
                    logger.warning(
                        f"API error: {response.status_code}, {response.text}"
                    )

            except Exception as e:
                logger.warning(f"Error calling API: {e}")

            # Wait before retrying
            time.sleep(self.retry_delay)

        logger.error(f"Failed to paraphrase after {self.max_retries} attempts")
        return None

    def batch_paraphrase(
        self, texts: List[str], sentiments: List[str]
    ) -> List[Optional[str]]:
        """
        Paraphrase a batch of texts.

        Args:
            texts: List of texts to paraphrase
            sentiments: List of sentiments for each text

        Returns:
            List[Optional[str]]: List of paraphrased texts
        """
        if len(texts) != len(sentiments):
            raise ValueError("Texts and sentiments lists must have the same length")

        results = []
        for text, sentiment in zip(texts, sentiments):
            paraphrased = self.paraphrase_text(text, sentiment)
            results.append(paraphrased)

        return results


if __name__ == "__main__":
    # Example usage
    paraphraser = LLMParaphraser()

    examples = [
        ("The company reported strong earnings in Q3.", "positive"),
        ("The stock price declined by 5% following the announcement.", "negative"),
        ("The board approved a new share buyback program.", "neutral"),
    ]

    for text, sentiment in examples:
        paraphrased = paraphraser.paraphrase_text(text, sentiment)
        print(f"Original ({sentiment}): {text}")
        print(f"Paraphrased: {paraphrased}")
        print()
