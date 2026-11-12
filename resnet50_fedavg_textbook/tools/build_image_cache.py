"""Build the 224x224 uint8 image cache for all hospitals (CPU, one-time, about 15 minutes)."""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hospital_registry import discover_hospitals  # noqa: E402
from runtime.config import load_config  # noqa: E402
from runtime.image_cache import build_split_cache, cache_is_current  # noqa: E402


def main() -> None:
    cfg = load_config()
    root, cache_dir = cfg["data"]["partitions_root"], cfg["data"]["cache_dir"]
    for hospital in discover_hospitals(root):
        for split in ("train", "test"):
            if cache_is_current(root, cache_dir, hospital, split):
                print(f"current  {hospital} {split}", flush=True)
                continue
            start = time.time()
            build_split_cache(root, cache_dir, hospital, split)
            print(f"built    {hospital} {split} in {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
