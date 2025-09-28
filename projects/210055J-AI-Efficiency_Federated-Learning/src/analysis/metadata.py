from __future__ import annotations

from pathlib import Path
from datetime import datetime
import platform
import subprocess
import sys
from typing import Dict

import yaml


def create_run_dir(base_path: Path, config: Dict) -> Path:
    """Create a directory for a single experiment run and store metadata.

    Parameters
    ----------
    base_path: Path
        Root directory where the ``results`` folder will be created.
    config: dict
        Experiment configuration to be saved alongside metadata.

    Returns
    -------
    Path
        Path to the newly created run directory ``results/<tag>``.
    """
    tag = config.get("tag")
    if not tag:
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_path / "results" / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=base_path)
            .decode()
            .strip()
        )
    except Exception:
        git_commit = "unknown"

    metadata = {
        "config": config,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "git_commit": git_commit,
    }
    with (run_dir / "metadata.yaml").open("w") as f:
        yaml.safe_dump(metadata, f)

    return run_dir


__all__ = ["create_run_dir"]
