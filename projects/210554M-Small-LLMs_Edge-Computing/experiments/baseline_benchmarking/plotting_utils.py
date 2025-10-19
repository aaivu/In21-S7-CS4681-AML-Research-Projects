import os
import pandas as pd
import matplotlib.pyplot as plt

def safe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def generate_bar_plots(csv_file: str, plots_dir: str, bar_cfg: dict):
    safe_mkdir(plots_dir)
    df = pd.read_csv(csv_file)
    if df.empty:
        return

    if bar_cfg.get("model_size", False):
        plt.figure(figsize=(8, 6))
        plt.bar(df["Model"], df["Size (GB)"], color="skyblue")
        plt.title("Model Size (GB)")
        plt.ylabel("GB")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "bar_model_size.png"))
        plt.close()

    if bar_cfg.get("latency", False):
        plt.figure(figsize=(8, 6))
        plt.bar(df["Model"], df["Latency (ms_per_token)"], color="lightgreen")
        plt.title("Latency (ms_per_token)")
        plt.ylabel("ms_per_token")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "bar_latency.png"))
        plt.close()

    if bar_cfg.get("perplexity", False):
        plt.figure(figsize=(8, 6))
        plt.bar(df["Model"], df["Perplexity (WikiText-2)"], color="salmon")
        plt.title("Perplexity (WikiText-2)")
        plt.ylabel("Perplexity")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "bar_perplexity.png"))
        plt.close()

    if bar_cfg.get("metrics_grouped", False):
        metrics = [c for c in ["BoolQ Acc (%)", "SQuAD EM (%)", "SQuAD F1 (%)"] if c in df.columns]
        if metrics:
            ax = df.plot(x="Model", y=metrics, kind="bar", figsize=(10, 6))
            ax.set_title("Evaluation Metrics")
            ax.set_ylabel("Score (%)")
            plt.xticks(rotation=15, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "bar_metrics.png"))
            plt.close()

def generate_scatter_plots(csv_file: str, plots_dir: str, scatter_cfg: list):
    safe_mkdir(plots_dir)
    df = pd.read_csv(csv_file)
    if df.empty:
        return
    for combo in scatter_cfg:
        if not combo.get("enabled", False):
            continue
        x_col = combo.get("x")
        y_col = combo.get("y")
        if not x_col or not y_col:
            continue
        if x_col not in df.columns or y_col not in df.columns:
            continue
        plt.figure(figsize=(8, 6))
        plt.scatter(df[x_col], df[y_col], s=120)
        for i, row in df.iterrows():
            plt.text(row[x_col] + 0.01, row[y_col] + 0.01, row["Model"], fontsize=9)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"{x_col} vs {y_col}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        safe_name = f"scatter_{x_col.replace(' ', '_').replace('(', '').replace(')', '')}_vs_{y_col.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(os.path.join(plots_dir, safe_name))
        plt.close()