from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt  # import here so main can run without matplotlib installed

def _smooth(y: np.ndarray, k: int) -> np.ndarray:
    if k is None or k < 2 or y.size < 3:
        return y
    k = int(k)
    if k % 2 == 0:
        k += 1  # make odd
    k = min(k, y.size if y.size % 2 else y.size - 1)
    w = np.ones(k, dtype=np.float64) / k
    return np.convolve(y, w, mode="same")

def save_eval_plot(timesteps: Sequence[int], returns: Sequence[float], out_png: str, title: str = "", smooth_k: int = 0):
   
    x = np.asarray(timesteps, dtype=np.int64)
    y = np.asarray(returns, dtype=np.float64)

    plt.figure()
    plt.plot(x, y, label="Eval avg return", linewidth=1.6)
    if smooth_k and smooth_k > 1:
        ys = _smooth(y, smooth_k)
        plt.plot(x, ys, "--", label=f"Smoothed (k={smooth_k})")
    plt.xlabel("Timesteps")
    plt.ylabel("Average Return")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def save_train_plot(timesteps: Sequence[int], returns: Sequence[float], out_png: str, title: str = "", smooth_k: int = 0):
    x = np.asarray(timesteps, dtype=np.int64)
    y = np.asarray(returns, dtype=np.float64)

    plt.figure()
    plt.plot(x, y, alpha=0.6, label="Train episode return")
    if smooth_k and smooth_k > 1:
        ys = _smooth(y, smooth_k)
        plt.plot(x, ys, "--", label=f"Smoothed (k={smooth_k})")
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Return")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()