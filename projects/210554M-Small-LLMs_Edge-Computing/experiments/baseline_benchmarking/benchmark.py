import os
import yaml
import pandas as pd
import zipfile
import datetime
from benchmark_utils import run_model_evaluation
from plotting_utils import generate_bar_plots, generate_scatter_plots, safe_mkdir

CONFIG_PATH = "config.yaml"

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config(CONFIG_PATH)

    output_dir = cfg.get("output_dir", "benchmark_results")
    safe_mkdir(output_dir)
    csv_filename = cfg.get("csv_filename", "baseline_results.csv")
    csv_path = os.path.join(output_dir, csv_filename)
    append_mode = bool(cfg.get("append", True))

    models_cfg = cfg.get("models", {})
    eval_cfg = cfg.get("evaluation", {})
    plots_cfg = cfg.get("plots", {})

    results = []
    # run enabled models
    for model_name, enabled in models_cfg.items():
        if not enabled:
            continue
        try:
            print(f"\n=== Running evaluation for {model_name} ===")
            res = run_model_evaluation(model_name, cfg)
            results.append(res)
        except Exception as e:
            print(f"[ERROR] model {model_name} failed: {e}")

    if len(results) == 0:
        print("No results collected (no models enabled or all failed). Exiting.")
        return

    df_new = pd.DataFrame(results)

    # Save CSV: append or overwrite canonical CSV
    if append_mode and os.path.exists(csv_path):
        try:
            df_existing = pd.read_csv(csv_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(csv_path, index=False)
            print(f"Appended {len(df_new)} rows to {csv_path}")
        except Exception as e:
            print(f"Failed to append existing CSV ({e}) — writing new file instead.")
            df_new.to_csv(csv_path, index=False)
            print(f"Wrote {csv_path}")
    else:
        df_new.to_csv(csv_path, index=False)
        print(f"Wrote new CSV: {csv_path}")

    # Plots folder
    plots_dir = os.path.join(output_dir, "plots")
    safe_mkdir(plots_dir)

    # Bar plots
    if plots_cfg.get("bar_plots", {}):
        generate_bar_plots(csv_path, plots_dir, plots_cfg.get("bar_plots", {}))

    # Scatter plots
    if plots_cfg.get("scatter_plots", []):
        generate_scatter_plots(csv_path, plots_dir, plots_cfg.get("scatter_plots", []))

    # Zip everything with timestamp to preserve multiple runs
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"results_{timestamp}.zip"
    zip_path = os.path.join(output_dir, zip_name)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(csv_path, arcname=os.path.basename(csv_path))
        for root, _, files in os.walk(plots_dir):
            for fname in files:
                full = os.path.join(root, fname)
                arc = os.path.join("plots", fname)
                z.write(full, arcname=arc)
    print(f"\n✅ Zipped results to: {zip_path}")

if __name__ == "__main__":
    main()