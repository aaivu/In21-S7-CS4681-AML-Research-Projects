import pandas as pd
import numpy as np
import argparse
# Function to parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Subsample dataset by category")
    parser.add_argument('--input_csv', type=str, required=True, help="Path to the input CSV file")
    parser.add_argument('--output_csv', type=str, required=True, help="Path to the output CSV file")
    parser.add_argument('--remove_ratio', type=float, required=True, help="Fraction of categories to remove (e.g., 0.80 for 80%)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

# Main logic for subsampling
def main():
    # Parse command line arguments
    args = parse_args()

    # Load the training and validation data CSVs
    df = pd.read_csv(args.input_csv)

    # List of unique categories
    categories = df['category'].unique()

    # Set random seed for reproducibility
    np.random.seed(args.seed)

    # Choose a subset of categories to remove
    num_categories_to_remove = int(args.remove_ratio * len(categories))
    categories_to_remove = np.random.choice(categories, size=num_categories_to_remove, replace=False)

    # Filter out rows with these categories from both train and validation datasets
    df_subset = df[~df['category'].isin(categories_to_remove)]

    # Save the filtered data to new CSV files
    df_subset.to_csv(args.output_csv, index=False)

    # Optionally, print which categories were removed
    print(f"Removed categories: {categories_to_remove}")

if __name__ == "__main__":
    main()