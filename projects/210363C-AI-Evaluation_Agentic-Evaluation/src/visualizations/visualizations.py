import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.stats import gaussian_kde

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def get_latest_results_folder():
    """Return the latest results folder based on timestamp in its name."""
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


def load_results_summary():
    """Load the latest results_summary.csv file."""
    latest_results_folder = get_latest_results_folder()
    summary_file = os.path.join(latest_results_folder, 'results_summary.csv')
    return pd.read_csv(summary_file)


def combine_all_results():
    """Concatenate all results_summary.csv files into a single combined dataset."""
    results_base = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results'))
    subfolders = [f for f in os.listdir(results_base) if f.startswith('results_')]
    dfs = []

    for folder in subfolders:
        folder_path = os.path.join(results_base, folder)
        summary_file = os.path.join(folder_path, 'results_summary.csv')
        if os.path.exists(summary_file):
            try:
                df = pd.read_csv(summary_file)
                df['source_folder'] = folder
                dfs.append(df)
            except Exception as e:
                print(f"Skipped {folder}: {e}")

    if not dfs:
        raise ValueError("No results_summary.csv files found in any results_* folder.")

    combined_df = pd.concat(dfs, ignore_index=True)
    output_path = os.path.join(results_base, 'data.csv')
    combined_df.to_csv(output_path, index=False)
    print(f"Combined data saved at: {output_path}")
    return combined_df


def generate_variance_visualizations(df):
    """Generate violin, boxen, ridgeline, mean+std, and efficiency/cost visualizations."""
    charts_folder = os.path.abspath(os.path.join(get_latest_results_folder(), 'visualization-charts'))
    os.makedirs(charts_folder, exist_ok=True)

    # Ensure numeric types
    df["total_tokens"] = pd.to_numeric(df["total_tokens"], errors="coerce")
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df["walltime_sec"] = pd.to_numeric(df.get("walltime_sec", 0), errors="coerce")

    sns.set_theme(style="whitegrid", font_scale=1.2)
    seeds = sorted(df["seed"].dropna().unique())

    # ----------------------------------------------------------------------
    # Per-seed plots (4)
    # ----------------------------------------------------------------------
    for seed in seeds:
        subset_df = df[df["seed"] == seed]
        print(f"Generating plots for Seed {seed}...")

        # 1. Violin Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.violinplot(data=subset_df, x="temperature", y="total_tokens", palette="viridis", ax=ax)
        ax.set_title(f"Token Distribution by Temperature (Violin) — Seed {seed}", fontsize=14, weight="bold")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Total Tokens")
        plt.tight_layout()
        plt.savefig(os.path.join(charts_folder, f"seed_{seed}_violin.png"))
        plt.close()

        # 2. Boxen Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxenplot(data=subset_df, x="temperature", y="total_tokens", palette="coolwarm", ax=ax)
        ax.set_title(f"Token Distribution by Temperature (Boxen) — Seed {seed}", fontsize=14, weight="bold")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Total Tokens")
        plt.tight_layout()
        plt.savefig(os.path.join(charts_folder, f"seed_{seed}_boxen.png"))
        plt.close()

        # 3. Ridgeline Density Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        temperatures = sorted(subset_df["temperature"].dropna().unique())
        colors = sns.color_palette("husl", len(temperatures))
        for i, (t, c) in enumerate(zip(temperatures, colors)):
            temp_data = subset_df[subset_df["temperature"] == t]["total_tokens"].dropna().values
            if len(temp_data) > 1:
                kde = gaussian_kde(temp_data)
                x_vals = np.linspace(temp_data.min(), temp_data.max(), 200)
                y_vals = kde(x_vals)
                ax.fill_between(x_vals, y_vals + i * 0.002, i * 0.002, color=c, alpha=0.6)
                ax.text(temp_data.mean(), i * 0.002 + max(y_vals) * 0.5, f"T={t}", fontsize=9, weight="bold")
        ax.set_title(f"Ridgeline Density Plot — Seed {seed}", fontsize=14, weight="bold")
        ax.set_xlabel("Total Tokens")
        ax.set_ylabel("Density (stacked)")
        plt.tight_layout()
        plt.savefig(os.path.join(charts_folder, f"seed_{seed}_ridgeline.png"))
        plt.close()

        # 4. Mean + STD Band Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        token_summary = subset_df.groupby("temperature")["total_tokens"].agg(["mean", "std"]).reset_index()
        x = token_summary["temperature"].values
        y = token_summary["mean"].values
        yerr = token_summary["std"].values

        ax.fill_between(x, y - yerr, y + yerr, alpha=0.2, color="#4C72B0")
        points = ax.scatter(x, y, c=y, cmap="viridis", s=100, edgecolor="black", zorder=3)
        ax.plot(x, y, color="#4C72B0", linewidth=2.5, zorder=2)

        ax.set_title(f"Token Usage Across Temperatures — Seed {seed}", fontsize=14, weight="bold")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Average Total Tokens")

        for xi, yi in zip(x, y):
            ax.text(xi, yi + 20, f"{int(yi)}", ha="center", fontsize=9, color="black")

        cbar = fig.colorbar(points, ax=ax)
        cbar.set_label("Average Token Usage", fontsize=11)

        plt.tight_layout()
        plt.savefig(os.path.join(charts_folder, f"seed_{seed}_mean_std.png"))
        plt.close()

    # ----------------------------------------------------------------------
    # Global analysis plots (3)
    # ----------------------------------------------------------------------

    # 1. Synthetic estimated cost (tokens + scaled walltime)
    df["est_cost"] = df["total_tokens"] * 1.0 + df["walltime_sec"] * 10.0

    # 2. Token Efficiency: Tokens per Correct
    efficiency = df.groupby("temperature").apply(
        lambda g: g["total_tokens"].sum() / max(g["correct"].sum(), 1)
    ).reset_index(name="tokens_per_correct")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=efficiency, x="temperature", y="tokens_per_correct", palette="viridis")
    plt.title("Token Efficiency: Tokens Spent per Correct Answer", fontsize=16, weight="bold")
    plt.xlabel("Temperature", fontsize=13)
    plt.ylabel("Tokens per Correct", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_folder, "token_efficiency.png"), dpi=300)
    plt.close()

    # 3. Heatmap of Estimated Cost by Seed & Temperature
    pivot_cost = df.pivot_table(
        values="est_cost",
        index="seed",
        columns="temperature",
        aggfunc="mean"
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        pivot_cost,
        cmap="YlGnBu",
        annot=True,
        fmt=".0f",
        cbar_kws={"label": "Estimated Cost"}
    )
    plt.title("Heatmap of Estimated Cost by Seed & Temperature", fontsize=16, weight="bold")
    plt.xlabel("Temperature", fontsize=13)
    plt.ylabel("Seed", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_folder, "cost_heatmap.png"), dpi=300)
    plt.close()

    # 4. Cost vs Temperature Trend (Mean ± Std)
    fig, ax = plt.subplots(figsize=(8, 6))
    cost_summary = df.groupby("temperature")["est_cost"].agg(["mean", "std"]).reset_index()
    x = cost_summary["temperature"].values
    y = cost_summary["mean"].values
    yerr = cost_summary["std"].values

    ax.fill_between(x, y - yerr, y + yerr, alpha=0.2, color="#2C7BB6")
    ax.plot(x, y, marker="o", color="#2C7BB6", linewidth=2.5)
    ax.set_title("Average Estimated Cost Across Temperatures", fontsize=16, weight="bold")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Average Estimated Cost")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_folder, "cost_trend.png"), dpi=300)
    plt.close()

    print(f"All visualizations saved in: {charts_folder}")


def main():
    combine_all_results()
    df = load_results_summary()
    generate_variance_visualizations(df)


if __name__ == "__main__":
    main()
