import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Function to find the latest results folder
def get_latest_results_folder():
    results_base = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results'))
    subfolders = [f for f in os.listdir(results_base) if f.startswith('results_')]
    valid_folders = []

    for folder in subfolders:
        try:
            datetime.strptime(folder.split('_', 1)[1], '%Y%m%d_%H%M%S')
            valid_folders.append(folder)
        except ValueError:
            continue

    if not valid_folders:
        raise ValueError("No valid results folders found.")

    latest_folder = max(valid_folders, key=lambda x: datetime.strptime(x.split('_', 1)[1], '%Y%m%d_%H%M%S'))
    return os.path.join(results_base, latest_folder)

# Function to load the results summary file
def load_results_summary():
    latest_results_folder = get_latest_results_folder()
    summary_file = os.path.join(latest_results_folder, 'results_summary.csv')
    return pd.read_csv(summary_file)

def generate_variance_visualizations(df):
    """Generate plots to visualize variance across seeds and temperatures."""
    charts_folder = os.path.abspath(os.path.join(get_latest_results_folder(), 'visualization-charts'))
    os.makedirs(charts_folder, exist_ok=True)

    # List of metrics to visualize variance
    metrics = ['TPS', 'EPS', 'TE', 'CE', 'LE']

    for metric in metrics:
        # Boxplot for variance across seeds and temperatures for each metric
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df, x='temperature', y=metric, hue='seed', palette='Set3')
        plt.title(f'Variance of {metric} Across Seeds and Temperatures')
        plt.xlabel('Temperature')
        plt.ylabel(metric)
        plt.legend(title='Seed')
        plt.savefig(os.path.join(charts_folder, f'variance_{metric.lower()}.png'))
        plt.show()

    # Pairplot for relationships between metrics
    sns.pairplot(df, vars=metrics, hue='seed', palette='husl', diag_kind='kde')
    plt.suptitle('Pairwise Relationships Between Metrics', y=1.02, fontsize=16)
    plt.savefig(os.path.join(charts_folder, 'pairwise_metrics.png'))
    plt.show()

def main():
    # Load the results summary
    df = load_results_summary()

    # Generate variance visualizations
    generate_variance_visualizations(df)

if __name__ == "__main__":
    main()