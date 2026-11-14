"""Reproducibility: fixed random seeds and a record of how each result was produced."""

from __future__ import annotations

import json
import platform
import random
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np

PACKAGES = ["numpy", "pandas", "scikit-learn", "xgboost"]


def set_seed(seed: int) -> None:
    """Fix the seeds of the random number generators used by the project."""
    random.seed(seed)
    np.random.seed(seed)


def git_commit() -> str:
    """Hash of the current commit, with ``-dirty`` if there are uncommitted changes."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"],
                                        text=True, stderr=subprocess.DEVNULL).strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_info() -> dict:
    """Commit, date, Python and package versions of the current run."""
    versions = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return {"git_commit": git_commit(),
            "date": datetime.now().isoformat(timespec="seconds"),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": versions}


def save_run(folder: str | Path, config: dict, result=None, extra: dict | None = None) -> Path:
    """Save the configuration, run information and (optionally) a ``CVResult``."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    info = {"config": config, "run": run_info(), **(extra or {})}
    if result is not None:
        info["summary"] = result.summary()
        result.predictions().to_csv(folder / "predictions.csv", index=False)
        result.fold_metrics().to_csv(folder / "fold_metrics.csv", index=False)
        result.selected_features().to_csv(folder / "selected_features.csv", index=False)
    (folder / "run.json").write_text(json.dumps(info, indent=2))
    return folder
