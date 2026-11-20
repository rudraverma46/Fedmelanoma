"""Records, ON THE MACHINE THAT HOLDS THE NOVELTY FOLDER, the SHA-256 of everything this baseline must share with the
ResNet-50 Novelty run, so tests/test_00 can prove identity on a machine where the novelty folder is absent.

Recorded in configs/reference_manifest.json:
  * pipeline files of the novelty folder (model, client, loss, data, evaluation, monitoring, config loader)
  * the novelty server/straggler.py (our sim/straggler.py, the environment)
  * the novelty config values the two runs share
  * every image-cache file (.npy pixels and .json row order) and every hospital split CSV used for training/testing

Usage (from this folder): python tools/make_reference_manifest.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from runtime.config import load_config  # noqa: E402

NOVELTY = os.path.abspath(os.path.join(ROOT, "..", "RESNET 50_ fed novelty", "RESNET50_NOVELTY MODEL"))
PIPELINE = ["models/model.py", "client/dataset.py", "client/focal_loss.py", "client/client.py",
            "client/local_trainer.py", "runtime/image_cache.py", "runtime/config.py",
            "evaluate.py", "evaluation/evaluate_checkpoint.py", "hospital_registry.py",
            "monitoring/checkpointing.py", "monitoring/experiment_logging.py", "monitoring/param_hash.py",
            "monitoring/subgroup_metrics.py", "tests/conftest.py"]


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    cfg = load_config()
    with open(os.path.join(NOVELTY, "configs", "experiment_config.yaml"), encoding="utf-8") as fh:
        base = yaml.safe_load(fh)
    novelty_cache = os.path.normpath(os.path.join(NOVELTY, base["data"]["cache_dir"]))
    partitions = cfg["data"]["partitions_root"]
    hospitals = sorted(d for d in os.listdir(partitions) if os.path.isdir(os.path.join(partitions, d)))
    manifest = {
        "source": NOVELTY,
        "pipeline_files": {rel: sha(os.path.join(NOVELTY, rel)) for rel in PIPELINE},
        "straggler_py": sha(os.path.join(NOVELTY, "server", "straggler.py")),
        "novelty_config": {"local_training": base["local_training"], "model": base["model"],
                           "runtime": {k: base["runtime"][k] for k in ("dataloader_workers", "pin_memory",
                                                                     "commit_floor_gb", "monitor_interval_seconds")},
                           "evaluation_thresholds": base["reference_constants"]["evaluation_thresholds"],
                           "seed": base["experiment"]["seed"], "local_epochs": base["federation"]["local_epochs"],
                           "num_clients": base["federation"]["num_clients"],
                           "num_server_rounds": base["federation"]["num_server_rounds"]},
        "cache_files": {f: sha(os.path.join(novelty_cache, f)) for f in sorted(os.listdir(novelty_cache))},
        "split_csvs": {f"{h}/{name}": sha(os.path.join(partitions, h, name)) for h in hospitals
                       for name in ("metadata_train.csv", "metadata_test.csv")},
    }
    out = os.path.join(ROOT, "configs", "reference_manifest.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {out}: {len(manifest['pipeline_files'])} files, {len(manifest['cache_files'])} cache files, "
          f"{len(manifest['split_csvs'])} split CSVs")


if __name__ == "__main__":
    main()
