"""Compute the participation schedule without any training.

The dropout draw reads only the seed, so all 50 rounds can be replayed in advance. Writes
results/schedule/planned_schedule.json, which the tests and the final report compare with the real run.

Usage: python tools/dry_run_schedule.py [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from hospital_registry import build_hospital_registry  # noqa: E402
from runtime.config import load_config  # noqa: E402


def simulate(sizes: Dict[int, int], seed: int, rounds: int, n_train: int) -> List[dict]:
    rng = np.random.default_rng(seed)
    client_ids = sorted(sizes)
    schedule = []
    for t in range(1, rounds + 1):
        dropped = sorted(int(c) for c in rng.choice(client_ids, size=len(client_ids) - n_train, replace=False))
        trained = [c for c in client_ids if c not in dropped]
        total = sum(sizes[c] for c in trained)
        schedule.append({"round": t, "addressed": client_ids, "dropped": dropped, "trained": trained, "weights": {str(c): sizes[c] / total for c in trained}})
    return schedule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config()
    seed = cfg["experiment"]["seed"] if args.seed is None else args.seed
    names, records, *_ = build_hospital_registry(cfg["data"]["partitions_root"])
    sizes = {i: r["n_samples"] for i, r in enumerate(records)}
    schedule = simulate(sizes, seed, cfg["federation"]["num_server_rounds"], cfg["federation"]["clients_per_round"])
    trained = Counter(c for r in schedule for c in r["trained"])
    out = {"seed": seed, "hospitals": [{"client_id": i, "hospital": names[i], "n_samples": sizes[i]} for i in sizes],
           "summary": {"rounds": len(schedule), "updates": sum(trained.values()), "trained_count": {str(c): trained[c] for c in sizes}}, "schedule": schedule}
    path = os.path.join(ROOT, "results", "schedule", "planned_schedule.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
