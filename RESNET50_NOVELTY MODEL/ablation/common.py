"""Registry, per-run configuration, hashes and output paths shared by every ablation script.

A run is (experiment id, seed). Its run id is "<exp>-s<seed>-<config hash[:8]>", where the config
hash covers everything that defines the experiment except the seed, so a changed setting can never
resume or overwrite an older run.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from typing import Any, Dict, List, Tuple

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_PATH = os.path.join(ROOT, "ablation", "experiments.yaml")
BASE_CONFIG_PATH = os.path.join(ROOT, "configs", "experiment_config.yaml")
RAW_DIR = os.path.join(ROOT, "results", "raw")

DEFAULT_DATA = "/kaggle/input/datasets/visheshkumarbansal/fedmel01/hospital_partitions"
DEFAULT_CACHE = "/tmp/fedmel_cache"   # outside /kaggle/working, so it is not saved as output

VARIANT_DEFAULTS = {"selection": "boltzmann", "aggregation": "fedmelanoma", "straggler_buffer": True,
                    "focal_gamma": 2.0}


def load_registry(path: str = REGISTRY_PATH) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def variant_of(registry: Dict[str, Any], exp_id: str) -> Dict[str, Any]:
    overrides = registry["experiments"][exp_id]
    unknown = set(overrides) - set(VARIANT_DEFAULTS)
    if unknown:
        raise ValueError(f"{exp_id}: unknown variant keys {sorted(unknown)}")
    return {**VARIANT_DEFAULTS, **overrides}


def data_paths() -> Tuple[str, str]:
    return os.environ.get("FEDMEL_DATA", DEFAULT_DATA), os.environ.get("FEDMEL_CACHE", DEFAULT_CACHE)


def _sha256_file(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def split_id(partitions_root: str) -> str:
    """Hash of every hospital's train/test CSV: identical partitions give an identical id."""
    digest = hashlib.sha256()
    for hospital in sorted(os.listdir(partitions_root)):
        for name in ("metadata_train.csv", "metadata_test.csv"):
            path = os.path.join(partitions_root, hospital, name)
            if os.path.isfile(path):
                digest.update(f"{hospital}/{name}:{_sha256_file(path)}".encode())
    return digest.hexdigest()[:16]


def code_hash() -> str:
    """Hash of every .py file in the experiment folder, recorded per run (not part of the run id)."""
    digest = hashlib.sha256()
    for folder, dirs, files in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d not in {"results", "checkpoints", "logs", "cache", "__pycache__"})
        for name in sorted(files):
            if name.endswith(".py"):
                path = os.path.join(folder, name)
                digest.update(os.path.relpath(path, ROOT).replace(os.sep, "/").encode())
                digest.update(_sha256_file(path).encode())
    return digest.hexdigest()[:16]


def build_run_config(exp_id: str, seed: int, registry: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Base config with this run's rounds, seed, data paths, focal gamma, variant and harness."""
    registry = registry or load_registry()
    with open(BASE_CONFIG_PATH, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    variant = variant_of(registry, exp_id)
    common = registry["common"]
    partitions_root, cache_dir = data_paths()

    cfg = copy.deepcopy(cfg)
    cfg["experiment"]["seed"] = int(seed)
    cfg["federation"]["num_server_rounds"] = int(common["num_server_rounds"])
    cfg["local_training"]["focal_gamma"] = float(variant["focal_gamma"])
    cfg["data"] = {"partitions_root": partitions_root, "cache_dir": cache_dir}
    cfg["runtime"]["dataloader_workers"] = int(os.environ.get("FEDMEL_WORKERS", 4))
    cfg["ablation"] = {"experiment": exp_id, "variant": variant, "harness": common["harness"]}
    cfg["experiment"]["run_id"] = run_id(exp_id, seed, cfg)
    return cfg


def config_hash(cfg: Dict[str, Any]) -> str:
    """Everything that defines the experiment, minus the seed and machine-specific paths/runtime."""
    keyed = {k: cfg[k] for k in ("federation", "local_training", "model", "ablation")}
    keyed["reference_constants"] = cfg.get("reference_constants")
    return hashlib.sha256(json.dumps(keyed, sort_keys=True).encode()).hexdigest()


def run_id(exp_id: str, seed: int, cfg: Dict[str, Any]) -> str:
    return f"{exp_id}-s{seed}-{config_hash(cfg)[:8]}"


def run_dir(exp_id: str, seed: int) -> str:
    return os.path.join(RAW_DIR, exp_id, f"seed_{seed}")


def batch_runs(batch: str, registry: Dict[str, Any] | None = None) -> List[Tuple[str, int]]:
    """Seed-major order, so a partial batch still holds complete seeds for every arm."""
    registry = registry or load_registry()
    return [(exp, seed) for seed in registry["common"]["seeds"] for exp in registry["batches"][batch]]


def atomic_write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)
