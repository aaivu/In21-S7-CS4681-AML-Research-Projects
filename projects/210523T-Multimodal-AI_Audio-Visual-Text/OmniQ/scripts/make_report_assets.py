import csv, math
from pathlib import Path
import matplotlib.pyplot as plt

RES = Path("results")
CSV = RES / "summary.csv"
RES.mkdir(exist_ok=True, parents=True)

if not CSV.exists():
    raise FileNotFoundError(f"Couldn't find {CSV}. Run your evals first to create it.")

# Load rows
rows = []
with CSV.open(newline="") as f:
    for r in csv.DictReader(f):
        rows.append(r)

def f(x, default=0.0):
    try: return float(x)
    except: return default

# Build convenience fields
for r in rows:
    r["Top1"]   = f(r.get("top1"))
    r["Top5"]   = f(r.get("top5"))
    r["Latency"] = f(r.get("avg_latency_ms"))
    r["VRAM"]   = f(r.get("peak_vram_GB"))
    r["ParamsM"]= f(r.get("params_M"))
    r["TrainM"] = f(r.get("trainable_params_M"))
    r["Frames"] = f(r.get("frames"), 32)
    r["Stride"] = f(r.get("stride"), 2)
    # Label for plots
    fusion = "Mamba" if "mamba" in r["model"] else ("Transformer" if "transformer" in r["model"] else "—")
    lora   = "✓" if str(r.get("lora_enabled","False")) in ("True","true","1") else "—"
    r["Label"] = f'{r["model"]} | {fusion} | LoRA {lora}'

# ---------- Plot 1: Top-1 vs. Latency ----------
plt.figure(figsize=(7,5))
xs = [r["Latency"] for r in rows]
ys = [r["Top1"] for r in rows]
for r in rows:
    plt.scatter(r["Latency"], r["Top1"])
    plt.annotate(r["Label"], (r["Latency"], r["Top1"]), xytext=(4,4), textcoords="offset points", fontsize=8)
plt.xlabel("Per-clip latency (ms)")
plt.ylabel("Top-1 accuracy (%)")
plt.title("UCF101: Accuracy vs Latency (lower is better on x)")
(RES / "acc_vs_latency.png").unlink(missing_ok=True)
plt.tight_layout()
plt.savefig(RES / "acc_vs_latency.png", dpi=160)
plt.close()

# ---------- Plot 2: Trainable params vs. Top-1 ----------
plt.figure(figsize=(7,5))
for r in rows:
    plt.scatter(r["TrainM"], r["Top1"])
    plt.annotate(r["Label"], (r["TrainM"], r["Top1"]), xytext=(4,4), textcoords="offset points", fontsize=8)
plt.xlabel("Trainable parameters (M)")
plt.ylabel("Top-1 accuracy (%)")
plt.title("UCF101: Trainable Params vs Accuracy")
(RES / "trainable_vs_top1.png").unlink(missing_ok=True)
plt.tight_layout()
plt.savefig(RES / "trainable_vs_top1.png", dpi=160)
plt.close()

# ---------- Make LaTeX + Markdown tables ----------
cols = ["Model","Fusion","LoRA","Frames/Stride","Top1","Top5","Params(M)","Trainable(M)","Peak VRAM(GB)","Latency(ms)"]
def fusion_of(r): return "Mamba" if "mamba" in r["model"] else ("Transformer" if "transformer" in r["model"] else "—")
def lora_of(r):   return "✓" if str(r.get("lora_enabled","False")) in ("True","true","1") else "—"
def row_vals(r):
    return [
        r["model"],
        fusion_of(r),
        lora_of(r),
        f'{int(r["Frames"])}/{int(r["Stride"])}',
        f'{r["Top1"]:.2f}',
        f'{r["Top5"]:.2f}',
        f'{r["ParamsM"]:.2f}',
        f'{r["TrainM"]:.2f}',
        f'{r["VRAM"]:.2f}',
        f'{r["Latency"]:.2f}',
    ]

# LaTeX
latex = []
latex.append("\\begin{tabular}{l l c c r r r r r r}")
latex.append("\\toprule")
latex.append(" & ".join(cols) + " \\\\")
latex.append("\\midrule")
for r in rows:
    latex.append(" & ".join(row_vals(r)) + " \\\\")
latex.append("\\bottomrule")
latex.append("\\end{tabular}")
(RES / "table_ucf101.tex").write_text("\n".join(latex))

# Markdown
md = []
md.append("| " + " | ".join(cols) + " |")
md.append("|" + "|".join(["---"]*len(cols)) + "|")
for r in rows:
    md.append("| " + " | ".join(row_vals(r)) + " |")
(RES / "table_ucf101.md").write_text("\n".join(md))

# ---------- Tiny summary (best vs baseline) ----------
def is_baseline(r): return "swin_tiny_2d_temporalavg" in r["model"]
base = max([r for r in rows if is_baseline(r)], key=lambda r: r["Top1"], default=None)
best = max(rows, key=lambda r: r["Top1"]) if rows else None
summary = []
if base and best:
    d_acc = best["Top1"] - base["Top1"]
    d_lat = base["Latency"] - best["Latency"]
    d_vram= base["VRAM"] - best["VRAM"]
    summary.append(f'Best model: {best["Label"]}')
    summary.append(f'ΔTop-1 vs baseline: {d_acc:+.2f} pts')
    summary.append(f'Latency change (ms): {d_lat:+.2f} (positive = faster)')
    summary.append(f'Peak VRAM change (GB): {d_vram:+.2f} (positive = uses less)')
(RES / "summary.txt").write_text("\n".join(summary) if summary else "Add baseline and rerun.")

print("Wrote:")
for p in ["acc_vs_latency.png","trainable_vs_top1.png","table_ucf101.tex","table_ucf101.md","summary.txt"]:
    print(" -", RES / p)
