"""Builds the pixel-identical 224x224 image cache (runtime/image_cache.py) in parallel, skipping
splits that are already current.

    FEDMEL_DATA=... FEDMEL_CACHE=... python -m ablation.build_cache [--workers 4]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ablation.common import data_paths  # noqa: E402
from hospital_registry import discover_hospitals  # noqa: E402
from runtime.image_cache import build_split_cache, cache_is_current  # noqa: E402


def _build(job):
    root, cache, hospital, split = job
    import cv2
    cv2.setNumThreads(1)
    start = time.time()
    build_split_cache(root, cache, hospital, split)
    return f"{hospital}/{split} {time.time() - start:.0f} s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    root, cache = data_paths()
    jobs = [(root, cache, h, s) for h in discover_hospitals(root) for s in ("train", "test")
            if not cache_is_current(root, cache, h, s)]
    print(f"{len(jobs)} splits to build into {cache}", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        for line in pool.map(_build, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
