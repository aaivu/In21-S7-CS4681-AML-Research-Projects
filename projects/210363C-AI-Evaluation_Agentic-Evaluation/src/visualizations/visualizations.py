import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path to the results file
RESULTS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results.csv'))

def load_results():
    """Load the results CSV file into a DataFrame."""
    return pd.read_csv(RESULTS_FILE)

def generate_visualizations(df):
    """Generate graphs, charts, and heatmaps."""
    metrics = ["TPS", "EPS", "TE", "CE", "LE"]
    charts_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results/visualization-charts'))
    os.makedirs(charts_folder, exist_ok=True)

    df[metrics].mean().plot(kind="bar", figsize=(10, 6), title="Average Metrics Across Runs")
    plt.ylabel("Metric Value")
    plt.savefig(os.path.join(charts_folder, "average_metrics.png"))
    plt.show()

    # Heatmap: Token usage or success rates across problem types
    if "problem_id" in df.columns:
        pivot_table = df.pivot_table(values="exec_successes", index="problem_id", columns="TE", aggfunc="mean")
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Heatmap of Success Rates Across Problem IDs")
        plt.savefig(os.path.join(charts_folder, "success_rate_heatmap.png"))
        plt.show()

def visualize_comparisons(df):
    """Generate visualizations to compare metrics across runs."""
    metrics = ["TPS", "EPS", "TE", "CE", "LE"]
    charts_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results/visualization-charts'))
    os.makedirs(charts_folder, exist_ok=True)

    # Pairplot for relationships between metrics
    sns.pairplot(df[metrics], diag_kind="kde", corner=True, plot_kws={"alpha": 0.6})
    plt.suptitle("Pairwise Relationships Between Metrics", y=1.02, fontsize=16)
    plt.savefig(os.path.join(charts_folder, "pairwise_relationships.png"))
    plt.show()

def main():
    # Ensure visualizations folder exists
    df = load_results()
    generate_visualizations(df)
    visualize_comparisons(df)

if __name__ == "__main__":
    main()