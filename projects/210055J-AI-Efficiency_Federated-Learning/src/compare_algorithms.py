import os
from pathlib import Path
import yaml

from analysis.comparison_runner import run_comparison
from analysis.reporting import save_results

# Hardcoded values for the output directory and config file
BASE_OUTPUT_DIR = Path("analysis/comparisons")
CONFIG_FILE_PATH = Path("config/config.yaml") 

def get_next_versioned_dir(base_path: Path) -> Path:
    """
    Finds the next available versioned directory 
    """
    base_path.mkdir(parents=True, exist_ok=True)
    version = 1
    while True:
        versioned_dir = base_path / f"comparison_v{version}"
        if not versioned_dir.exists():
            return versioned_dir
        version += 1

def main() -> None:
    if not CONFIG_FILE_PATH.is_file():
        print(f"Error: Configuration file not found at {CONFIG_FILE_PATH}")
        return

    with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_dir = get_next_versioned_dir(BASE_OUTPUT_DIR)
    
    print(f"Running comparison and saving results to: {run_dir}")

    results = run_comparison(config, run_dir.parent, run_dir.name)

    save_results(results, run_dir)

    print(f"Artifacts stored in: {run_dir}")

if __name__ == "__main__":
    main()