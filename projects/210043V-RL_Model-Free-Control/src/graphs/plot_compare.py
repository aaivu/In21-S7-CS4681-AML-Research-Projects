import argparse
from pathlib import Path
import numpy as np

def smooth(y: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or y.size < 3:
        return y
    # make k odd and ≤ len(y)
    if k % 2 == 0: k += 1
    k = min(k, len(y) if len(y) % 2 == 1 else len(y) - 1)
    w = np.ones(k, dtype=np.float64) / k
    return np.convolve(y, w, mode="same")

def main():
    ap = argparse.ArgumentParser(description="Plot Return vs Timesteps from .npy files")
    ap.add_argument("--files", nargs="+", required=True,
                    help="One or more .npy files (each: eval returns over time)")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="Optional labels (same count/order as --files). Defaults to file basenames.")
    ap.add_argument("--eval_freq", type=int, default=5000,
                    help="Eval frequency in env steps (x-axis spacing).")
    ap.add_argument("--smooth_k", type=int, default=0,
                    help="Moving-average window for smoothing (0/1 = off).")
    ap.add_argument("--out", default=None,
                    help="Optional output PNG path. If omitted, shows an interactive window.")
    ap.add_argument("--title", default="Return vs Timesteps",
                    help="Plot title.")
    args = ap.parse_args()

    import matplotlib.pyplot as plt  # imported here so the script still loads without matplotlib

    if args.labels and len(args.labels) != len(args.files):
        raise SystemExit("Error: --labels count must match --files count")

    plt.figure()

    for i, f in enumerate(args.files):
        path = Path(f)
        y = np.load(path)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        x = np.arange(len(y)) * args.eval_freq

        if args.smooth_k and args.smooth_k > 1:
            y_plot = smooth(y, args.smooth_k)
        else:
            y_plot = y

        label = args.labels[i] if args.labels else path.stem
        plt.plot(x, y_plot, label=label)

    plt.xlabel("Timesteps")
    plt.ylabel("Average Return")
    plt.title(args.title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=200)
        print(f"Saved: {args.out}")
    else:
        plt.show()

if _name_ == "_main_":
    main()