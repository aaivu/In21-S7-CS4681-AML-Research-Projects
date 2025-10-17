#!/bin/bash
CONFIG_DIR="$(dirname "$0")/configs"   # experiments/configs
DATA_DIR="$(dirname "$0")/../data/wikitext"  # project_root/data/wikitext

CONFIGS=("baseline.json" "hybrid.json" "no_moe_baseline.json")

for cfg in "${CONFIGS[@]}"; do
    echo "====================================="
    echo "Running experiment with config: $cfg"
    echo "====================================="
    python "$(dirname "$0")/run_experiments.py" --config "$CONFIG_DIR/$cfg" --data-dir "$DATA_DIR"
done
