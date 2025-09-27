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
import random
from tqdm import tqdm
from typing import List, Tuple

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Financial domain-specific synonym dictionary
FINANCIAL_SYNONYMS = {
    "profit": ["earnings", "gain", "income", "return", "margin"],
    "loss": ["deficit", "shortfall", "decline", "negative earnings"],
    "revenue": ["sales", "income", "turnover", "proceeds", "top line"],
    "cost": ["expense", "expenditure", "outlay", "charge", "payment"],
    "debt": ["liability", "obligation", "loan", "borrowing", "financing"],
    "asset": ["resource", "property", "holding", "possession", "investment"],
    "growth": ["expansion", "increase", "development", "appreciation", "rise"],
    "decline": ["decrease", "reduction", "fall", "drop", "contraction"],
    "dividend": ["payout", "distribution", "return", "disbursement"],
    "investment": ["expenditure", "funding", "financing", "backing", "stake"],
    "acquisition": ["takeover", "purchase", "buyout", "merger", "procurement"],
    "share": ["stock", "security", "equity", "holding", "interest"],
    "market": ["exchange", "bourse", "trade", "industry", "sector"],
    "strategy": ["plan", "approach", "policy", "procedure", "method"],
    "performance": ["result", "outcome", "return", "achievement", "execution"],
    "forecast": ["projection", "estimate", "prediction", "outlook", "guidance"],
    "increase": ["rise", "gain", "growth", "appreciation", "rally"],
    "decrease": ["fall", "drop", "reduction", "decline", "depreciation"],
    "investor": ["shareholder", "stockholder", "backer", "financier"],
    "subsidiary": ["affiliate", "division", "unit", "branch", "offshoot"],
}


class FinBertDataAugmenter:
    """Data augmentation class for FinBERT model."""

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        synonym_aug_p: float = 0.3,
        backtranslate_p: float = 0.3,
        llm_p: float = 0.4,
        similarity_threshold: float = 0.75,
    ):
        """
        Initialize the data augmenter with specified parameters.

        Args:
            model_name: The name of the model to use for embeddings
            synonym_aug_p: Probability of applying synonym augmentation
            backtranslate_p: Probability of applying back-translation
            llm_p: Probability of applying LLM-based paraphrasing
            similarity_threshold: Threshold for cosine similarity filtering
        """
        self.model_name = model_name
        self.synonym_aug_p = synonym_aug_p
        self.backtranslate_p = backtranslate_p
        self.llm_p = llm_p
        self.similarity_threshold = similarity_threshold

        # Load tokenizer and model for embeddings
        logger.info(f"Loading model and tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Initialize augmenters
        self._init_augmenters()

    def _init_augmenters(self):
        """Initialize the NLP augmenters."""
        # Synonym replacement augmenter using WordNet
        try:
            import nltk

            nltk.download("wordnet", quiet=True)
            nltk.download("punkt", quiet=True)
            nltk.download("averaged_perceptron_tagger", quiet=True)

            self.synonym_aug = naw.SynonymAug(
                aug_src="wordnet",
                aug_p=0.3,  # 30% of words will be replaced
                aug_max=None,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize SynonymAug: {e}")
            # Fallback to a simple random synonym replacement
            self.synonym_aug = (
                lambda text: text
            )  # Just return the original text as fallback

        # Financial synonym replacement (custom)
        self.financial_synonym_aug = self._create_financial_synonym_augmenter()

        # Back-translation augmenter
        try:
            self.back_translate_aug = BackTranslation(
                url=[
                    "translate.google.com",
                    "translate.google.co.kr",
                ],
                proxies={
                    "http": "127.0.0.1:1234",
                    "http://host.name": "127.0.0.1:4012",
                },
            )
        except Exception as e:
            logger.warning(f"Failed to load back-translation model: {e}")
            self.back_translate_aug = None

    def _create_financial_synonym_augmenter(self):
        """Create a custom augmenter for financial domain-specific synonyms."""
        # For financial terms, we'll create a custom function instead of using RandomWordAug
        # because it doesn't support direct dictionary mapping

        def financial_synonym_replacement(text):
            words = text.split()
            for i, word in enumerate(words):
                word_lower = word.lower()
                if (
                    word_lower in FINANCIAL_SYNONYMS and random.random() < 0.3
                ):  # 30% chance to replace
                    synonyms = FINANCIAL_SYNONYMS[word_lower]
                    replacement = random.choice(synonyms)

                    # Preserve capitalization
                    if word.istitle():
                        replacement = replacement.capitalize()
                    elif word.isupper():
                        replacement = replacement.upper()

                    words[i] = replacement

            return " ".join(words)

        # Return the function as our "augmenter"
        return financial_synonym_replacement

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

    def apply_synonym_augmentation(self, text: str) -> str:
        """
        Apply synonym replacement augmentation.

        Args:
            text: The text to augment

        Returns:
            str: Augmented text
        """
        try:
            # 50% chance to use financial synonym replacement
            if random.random() < 0.5:
                # Our financial synonym augmenter is a function, not an augmenter object
                augmented = self.financial_synonym_aug(text)
            else:
                augmented = self.synonym_aug.augment(text)
                # Return the first augmented text if multiple were generated
                if isinstance(augmented, list):
                    augmented = augmented[0] if augmented else text

            return augmented
        except Exception as e:
            logger.warning(f"Synonym augmentation failed: {e}")
            return text

    def apply_back_translation(self, text: str) -> str:
        """
        Apply back-translation augmentation.

        Args:
            text: The text to augment

        Returns:
            str: Augmented text
        """
        if self.back_translate_aug is None:
            return text

        try:
            augmented = self.back_translate_aug.translate(
                text=text, src="en", tmp="zh-cn"
            )

            # Return the first augmented text if multiple were generated
            if isinstance(augmented.result_text, list):
                augmented = augmented[0] if augmented else text

            return augmented.result_text
        except Exception as e:
            logger.warning(f"Back-translation failed: {e}")
            return text

    def apply_llm_paraphrasing(self, text: str, sentiment_label: str) -> str:
        """
        Apply LLM-based paraphrasing using GPT-4o-mini.

        Args:
            text: The text to augment
            sentiment_label: The sentiment label (positive, negative, neutral)

        Returns:
            str: Augmented text
        """
        # This function would call an API endpoint with GPT-4o-mini
        # Since we don't have direct API access, we'll return a placeholder
        logger.warning(
            "LLM paraphrasing requires API access. Using a simulated response."
        )

        # Simulated response (in a real implementation, this would call an API)
        # Use some word substitutions as a basic paraphrase simulation
        words = text.split()
        if len(words) > 3:
            idx = random.randint(0, len(words) - 1)
            if words[idx].lower() in FINANCIAL_SYNONYMS:
                synonyms = FINANCIAL_SYNONYMS[words[idx].lower()]
                words[idx] = random.choice(synonyms)

        return " ".join(words)

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

        # # Apply synonym replacement
        # if random.random() < self.synonym_aug_p:
        #     syn_augmented = self.apply_synonym_augmentation(text)
        #     if syn_augmented != text:
        #         similarity = self.check_similarity(text, syn_augmented)
        #         if similarity >= self.similarity_threshold:
        #             augmented_texts.append((syn_augmented, label))

        # Apply back-translation
        # if random.random() < self.backtranslate_p:
        #     bt_augmented = self.apply_back_translation(text)
        #     if bt_augmented != text:
        #         similarity = self.check_similarity(text, bt_augmented)
        #         if similarity >= self.similarity_threshold:
        #             augmented_texts.append((bt_augmented, label))

        # Apply LLM paraphrasing
        if True:
            # if random.random() < self.llm_p:
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
