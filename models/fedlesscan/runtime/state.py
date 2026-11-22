"""Atomic save and restore of everything needed to continue a run at a round boundary (D10).

The FedLesScan object holds client histories, stored results, late updates in flight (with their weights),
per-client job counters (the position in the seeded environment trace) and the selection generator.
"""
from __future__ import annotations

import json
import os
import pickle
import random
import tempfile
from typing import Any, Dict

import numpy as np
import torch


def state_path(checkpoint_dir: str, run_id: str) -> str:
    return os.path.join(checkpoint_dir, run_id, "resume_state.pkl")


def capture_state(round_num: int, server, strategy, rng: np.random.Generator) -> Dict[str, Any]:
    return {
        "round": round_num,
        "parameters": server.parameters,
        "clients": strategy._clients,
        "best_state": strategy._best_checkpoint_state,
        # FedLesScan: histories, stored results, in-flight late updates, job counters and the selection generator.
        "fedlesscan": strategy._fedlesscan,
        "last_global_output_hash": strategy._last_global_output_hash,
        "numpy_rng": rng.bit_generator.state,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "python_rng": random.getstate(),
    }


def save_state(path: str, state: Dict[str, Any]) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp_state_", suffix=".pkl")
    try:
        with os.fdopen(fd, "wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def restore_state(path: str, server, strategy, rng: np.random.Generator) -> int:
    """Load state into a freshly built server and strategy. Returns the last completed round."""
    with open(path, "rb") as fh:
        state = pickle.load(fh)
    server.parameters = state["parameters"]
    strategy._clients = state["clients"]
    strategy._best_checkpoint_state = state["best_state"]
    strategy._fedlesscan = state["fedlesscan"]
    strategy._last_global_output_hash = state["last_global_output_hash"]
    rng.bit_generator.state = state["numpy_rng"]
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    random.setstate(state["python_rng"])
    return int(state["round"])


def truncate_jsonl_after_round(path: str, last_round: int) -> None:
    """Remove records of an unfinished round, so a resume cannot duplicate them."""
    if not os.path.isfile(path):
        return
    kept = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip() and json.loads(line).get("round", 0) <= last_round:
                kept.append(line)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.writelines(kept)
    os.replace(tmp, path)
