#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data augmentation for FinBERT model.
Implements synonym replacement, back-translation, and LLM-based paraphrasing.
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import nlpaug.augmenter.word as naw
from BackTranslation import BackTranslation
import argparse
import logging
from tqdm import tqdm
from typing import List, Tuple
import os
from dotenv import load_dotenv
from openai import OpenAI

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class FinBertDataAugmenter:
    """Data augmentation class for FinBERT model."""

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        similarity_threshold: float = 0.75,
    ):
        """
        Initialize the data augmenter with specified parameters.

        Args:
            model_name: The name of the model to use for embeddings
            similarity_threshold: Threshold for cosine similarity filtering
        """
        self.similarity_threshold = similarity_threshold
        self.model_name = model_name

        # Initialize tokenizer and model for embeddings
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning(
                "OPENAI_API_KEY not found in environment. LLM paraphrasing will not work."
            )
            self.openai_client = None
        else:
            self.openai_client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully")

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text using FinBERT."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use the CLS token embedding as the sentence representation
        return outputs.last_hidden_state[:, 0, :].numpy().squeeze()

    def check_similarity(self, original_text: str, augmented_text: str) -> float:
        """
        Check cosine similarity between original and augmented texts.

        Args:
            original_text: The original text
            augmented_text: The augmented text

        Returns:
            float: Cosine similarity score
        """
        orig_embedding = self.get_embedding(original_text)
        aug_embedding = self.get_embedding(augmented_text)

        # Calculate cosine similarity
        similarity = cosine_similarity(
            orig_embedding.reshape(1, -1), aug_embedding.reshape(1, -1)
        )[0][0]

        return similarity

    def apply_llm_paraphrasing(self, text: str, sentiment_label: str) -> str:
        """
        Apply LLM-based paraphrasing using GPT-5.

        Args:
            text: The text to augment
            sentiment_label: The sentiment label (positive, negative, neutral)

        Returns:
            str: Augmented text
        """
        if not self.openai_client:
            logger.warning("OpenAI client not initialized. Returning original text.")
            return text

        try:
            # Create a prompt that preserves sentiment while paraphrasing
            prompt = f"""You are a financial text paraphrasing expert. Your task is to paraphrase the following financial text while preserving its exact sentiment ({sentiment_label}).

Important instructions:
1. Keep the sentiment EXACTLY the same ({sentiment_label})
2. Maintain the financial context and terminology
3. Use different words and sentence structure while keeping the same meaning
4. Return ONLY the paraphrased text without any explanations or quotes
5. The paraphrase should sound natural and fluent

Original text: {text}

Paraphrased text:"""

            # Call OpenAI API
            response = self.openai_client.responses.create(
                model="gpt-5",
                input=[
                    {
                        "role": "system",
                        "content": "You are an expert at paraphrasing financial texts while preserving sentiment and meaning.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            paraphrased_text = response.output_text

            # Clean up any quotes that might have been added
            paraphrased_text = paraphrased_text.strip('"').strip("'")

            logger.debug(f"Original: {text}")
            logger.debug(f"Paraphrased: {paraphrased_text}")

            return paraphrased_text

        except Exception as e:
            logger.error(f"Error during LLM paraphrasing: {str(e)}")
            return text  # Return original text if paraphrasing fails

    def augment_text(self, text: str, label: str) -> List[Tuple[str, str]]:
        """
        Augment a single text with multiple techniques.

        Args:
            text: The text to augment
            label: The sentiment label

        Returns:
            List[Tuple[str, str]]: List of (augmented_text, label) pairs
        """
        augmented_texts = []

        # Apply LLM paraphrasing
        if True:
            llm_augmented = self.apply_llm_paraphrasing(text, label)
            if llm_augmented != text:
                similarity = self.check_similarity(text, llm_augmented)
                if similarity >= self.similarity_threshold and similarity != 1.0:
                    augmented_texts.append((llm_augmented, label))

        return augmented_texts

    def augment_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        label_column: str = "label",
        augment_percentage: float = 0.4,
    ) -> pd.DataFrame:
        """
        Augment a dataframe with multiple techniques.

        Args:
            df: The dataframe to augment
            text_column: The column containing text
            label_column: The column containing labels
            augment_percentage: Percentage of samples to augment (0.3-0.5)

        Returns:
            pd.DataFrame: Original + augmented data
        """
        # Sample a percentage of the data for augmentation
        n_samples = int(len(df) * augment_percentage)
        sample_indices = np.random.choice(df.index, size=n_samples, replace=False)
        sample_df = df.loc[sample_indices].copy()

        augmented_rows = []
        for idx, row in tqdm(
            sample_df.iterrows(), total=len(sample_df), desc="Augmenting data"
        ):
            text = row[text_column]
            label = row[label_column]

            augmented_pairs = self.augment_text(text, label)

            for aug_text, aug_label in augmented_pairs:
                new_row = row.copy()
                new_row[text_column] = aug_text
                new_row[label_column] = aug_label
                augmented_rows.append(new_row)

        # Create a new dataframe with augmented data
        if augmented_rows:
            augmented_df = pd.DataFrame(augmented_rows)
            # Combine original with augmented
            combined_df = pd.concat([df, augmented_df], ignore_index=True)
            logger.info(
                f"Original data size: {len(df)}, Augmented: {len(augmented_df)}, Combined: {len(combined_df)}"
            )
            return combined_df
        else:
            logger.warning("No augmented data generated.")
            return df

    def augment_and_save(
        self,
        input_path: str,
        output_path: str,
        text_column: str = "text",
        label_column: str = "label",
        augment_percentage: float = 0.4,
    ) -> None:
        """
        Load data, augment, and save to output path.

        Args:
            input_path: Path to input CSV file
            output_path: Path to save the augmented CSV
            text_column: Column containing text
            label_column: Column containing labels
            augment_percentage: Percentage of samples to augment (0.3-0.5)
        """
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path, sep="\t")

        logger.info(f"Original data size: {len(df)}")
        logger.info(f"Label distribution: {df[label_column].value_counts().to_dict()}")

        augmented_df = self.augment_dataframe(
            df,
            text_column=text_column,
            label_column=label_column,
            augment_percentage=augment_percentage,
        )

        logger.info(f"Augmented data size: {len(augmented_df)}")
        logger.info(
            f"New label distribution: {augmented_df[label_column].value_counts().to_dict()}"
        )

        # Save to output file
        augmented_df.to_csv(output_path, sep="\t", index=False)
        logger.info(f"Saved augmented data to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data augmentation for FinBERT")
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to input CSV file"
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to save augmented CSV file"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ProsusAI/finbert",
        help="Model name for embeddings",
    )
    parser.add_argument(
        "--augment_percentage",
        type=float,
        default=0.4,
        help="Percentage of data to augment (0.3-0.5)",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.75,
        help="Cosine similarity threshold for filtering",
    )

    args = parser.parse_args()

    augmenter = FinBertDataAugmenter(
        model_name=args.model_name, similarity_threshold=args.similarity_threshold
    )

    augmenter.augment_and_save(
        input_path=args.input_file,
        output_path=args.output_file,
        augment_percentage=args.augment_percentage,
    )
