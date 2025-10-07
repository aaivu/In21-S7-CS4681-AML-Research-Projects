from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import torch


def save_results(results: Dict[str, Any], output_dir: Path) -> None:
    """Persist comparison metrics and summary statistics.

    Parameters
    ----------
    results:
        Dictionary produced by :func:`run_comparison` containing metrics for
        ``fedavg`` and ``fedavg_kd`` algorithms. Each algorithm entry stores
        per-epoch ``accuracy`` and ``loss`` lists.
    output_dir:
        Destination directory where CSV tables, plots and summary text are
        written. The directory is created if it does not already exist.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fedavg_metrics = results.get("fedavg", {}).get("metrics", {})
    kd_metrics = results.get("fedavg_kd", {}).get("metrics", {})

    epochs = range(1, max(
        len(fedavg_metrics.get("accuracy", [])),
        len(kd_metrics.get("accuracy", [])),
    ) + 1)

    # Side-by-side accuracy and loss tables
    acc_df = pd.DataFrame(
        {
            "FedAvg": fedavg_metrics.get("accuracy", []),
            "FedAvg+KD": kd_metrics.get("accuracy", []),
        },
        index=epochs,
    )
    acc_df.index.name = "epoch"
    acc_df.to_csv(output_dir / "accuracy.csv")

    loss_df = pd.DataFrame(
        {
            "FedAvg": fedavg_metrics.get("loss", []),
            "FedAvg+KD": kd_metrics.get("loss", []),
        },
        index=epochs,
    )
    loss_df.index.name = "epoch"
    loss_df.to_csv(output_dir / "loss.csv")

    # Publication-style accuracy plot
    plt.style.use("seaborn-v0_8")
    plt.rcParams.update({"figure.dpi": 300, "font.size": 12})
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, acc_df["FedAvg"], label="FedAvg", marker="o")
    plt.plot(epochs, acc_df["FedAvg+KD"], label="FedAvg+KD", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy.png")
    plt.savefig(output_dir / "accuracy.svg")
    plt.close()

    # Summary statistics
    final_acc_fedavg = acc_df["FedAvg"].iloc[-1] if not acc_df["FedAvg"].empty else float("nan")
    final_acc_kd = acc_df["FedAvg+KD"].iloc[-1] if not acc_df["FedAvg+KD"].empty else float("nan")
    avg_loss_fedavg = loss_df["FedAvg"].mean() if not loss_df["FedAvg"].empty else float("nan")
    avg_loss_kd = loss_df["FedAvg+KD"].mean() if not loss_df["FedAvg+KD"].empty else float("nan")

    summary = (
        f"FedAvg final accuracy: {final_acc_fedavg:.4f}\n"
        f"FedAvg average loss: {avg_loss_fedavg:.4f}\n"
        f"FedAvg+KD final accuracy: {final_acc_kd:.4f}\n"
        f"FedAvg+KD average loss: {avg_loss_kd:.4f}\n"
    )
    (output_dir / "summary.txt").write_text(summary)

    # Persist final model weights for each algorithm
    for alg in ["fedavg", "fedavg_kd"]:
        state = results.get(alg, {}).get("model_state")
        if state:
            torch.save(state, output_dir / f"{alg}_model.pt")


__all__ = ["save_results"]

