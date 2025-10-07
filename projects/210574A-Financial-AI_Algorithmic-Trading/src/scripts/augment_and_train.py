#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to augment data and train FinBERT model.
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from dotenv import load_dotenv
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_augmentation import FinBertDataAugmenter
from finbert.finbert import Config, FinBert
from transformers import BertForSequenceClassification, AutoTokenizer

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def setup_training_directories(base_output_dir: str) -> dict:
    """
    Create output directories for the training run.

    Args:
        base_output_dir: Base directory for outputs

    Returns:
        dict: Dictionary of paths
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_output_dir, f"run_{timestamp}")

    paths = {
        "run_dir": run_dir,
        "augmented_data_dir": os.path.join(run_dir, "augmented_data"),
        "model_dir": os.path.join(run_dir, "model"),
        "logs_dir": os.path.join(run_dir, "logs"),
    }

    # Create directories
    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    return paths


def augment_data(args, paths):
    """
    Augment training data and save to output directory.

    Args:
        args: Command line arguments
        paths: Dictionary of paths

    Returns:
        dict: Paths to augmented files
    """
    logger.info("Starting data augmentation")

    # Create augmenter
    augmenter = FinBertDataAugmenter(
        model_name=args.model_name, similarity_threshold=args.similarity_threshold
    )

    # Define paths for augmented files
    augmented_files = {
        "train": os.path.join(paths["augmented_data_dir"], "train.csv"),
        "validation": os.path.join(paths["augmented_data_dir"], "validation.csv"),
        "test": os.path.join(paths["augmented_data_dir"], "test.csv"),
    }

    # Augment only training data
    augmenter.augment_and_save(
        input_path=os.path.join(args.data_dir, "train.csv"),
        output_path=augmented_files["train"],
        augment_percentage=args.augment_percentage,
    )

    # Copy validation and test sets without augmentation
    for dataset in ["validation", "test"]:
        input_path = os.path.join(args.data_dir, f"{dataset}.csv")
        output_path = augmented_files[dataset]
        if os.path.exists(input_path):
            df = pd.read_csv(input_path, sep="\t")
            df.to_csv(output_path, sep="\t", index=False)
            logger.info(f"Copied {dataset} set to {output_path}")

    return augmented_files


def train_model(args, paths, augmented_files):
    """
    Train the FinBERT model using augmented data.

    Args:
        args: Command line arguments
        paths: Dictionary of paths
        augmented_files: Paths to augmented data files
    """
    logger.info("Starting model training")

    # Initialize the BERT model
    model = BertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,  # positive, negative, neutral
        output_attentions=False,
        output_hidden_states=False,
    )

    # Create model configuration
    config = Config(
        data_dir=paths["augmented_data_dir"],
        bert_model=model,
        model_dir=paths["model_dir"],
        max_seq_length=args.max_seq_length,
        train_batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warm_up_proportion=args.warm_up_proportion,
        no_cuda=not args.use_cuda,
        do_lower_case=True,
        output_mode="classification",
    )

    # Initialize FinBERT trainer
    finbert = FinBert(config)
    # Set base_model attribute (missing in the original class)
    finbert.base_model = args.model_name

    # Prepare the model
    label_list = ["positive", "negative", "neutral"]
    finbert.prepare_model(label_list)

    # Get training data
    train_examples = finbert.get_data("train")

    # Train the model
    model = finbert.create_the_model()
    trained_model = finbert.train(train_examples, model)

    # Evaluate on test set
    test_examples = finbert.get_data("test")
    results = finbert.evaluate(trained_model, test_examples)

    # Save evaluation results
    results_file = os.path.join(paths["logs_dir"], "evaluation_results.csv")
    results.to_csv(results_file)
    logger.info(f"Evaluation results saved to {results_file}")

    # Print class distribution and evaluation metrics
    label_counts = results["labels"].value_counts()
    logger.info(f"Test set class distribution: {label_counts.to_dict()}")

    from finbert.utils import get_metrics

    metrics = get_metrics(results)

    # Save metrics to file
    metrics_file = os.path.join(paths["logs_dir"], "metrics.json")
    import json

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Model evaluation metrics: {metrics}")
    logger.info(f"Training completed. Model saved to {paths['model_dir']}")


def main():
    parser = argparse.ArgumentParser(description="Augment data and train FinBERT model")

    # Data augmentation parameters
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/sentiment_data",
        help="Path to data directory",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Path to output directory"
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

    # Model parameters
    parser.add_argument(
        "--model_name",
        type=str,
        default="ProsusAI/finbert",
        help="Model name for embeddings and base model",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=64, help="Maximum sequence length"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument(
        "--num_train_epochs", type=float, default=4.0, help="Number of training epochs"
    )
    parser.add_argument(
        "--warm_up_proportion", type=float, default=0.1, help="Warm up proportion"
    )
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=None,
        help="OpenAI API key for LLM paraphrasing",
    )

    args = parser.parse_args()

    # Set OpenAI API key if provided
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key

    # Setup directories
    paths = setup_training_directories(args.output_dir)

    # Log configuration
    logger.info(f"Run directory: {paths['run_dir']}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Model: {args.model_name}")

    # Augment data
    augmented_files = augment_data(args, paths)

    # Train model
    train_model(args, paths, augmented_files)

    logger.info("Augmentation and training completed successfully")


if __name__ == "__main__":
    main()
