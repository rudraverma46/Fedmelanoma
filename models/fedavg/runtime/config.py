"""Loads configs/experiment_config.yaml and resolves paths against the experiment folder."""
from __future__ import annotations

import os
from typing import Any, Dict

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "configs", "experiment_config.yaml")


def load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["data"]["partitions_root"] = os.path.normpath(os.path.join(ROOT, cfg["data"]["partitions_root"]))
    cfg["data"]["cache_dir"] = os.path.normpath(os.path.join(ROOT, cfg["data"]["cache_dir"]))
    return cfg
