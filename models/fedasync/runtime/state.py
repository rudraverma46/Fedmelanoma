"""Atomic save/restore of the whole run at an evaluation boundary, and log truncation on resume.

The state holds the global model, the server version, the scheduler (virtual clock, idle set, every
in-flight update with its already-trained weights), all RNG states and the best-checkpoint record, so a
resumed run continues exactly where the saved one stopped.
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


def rng_states(numpy_rng: np.random.Generator) -> Dict[str, Any]:
    return {"numpy_rng": numpy_rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python_rng": random.getstate(), "np_global_rng": np.random.get_state()}


def restore_rngs(state: Dict[str, Any], numpy_rng: np.random.Generator) -> None:
    numpy_rng.bit_generator.state = state["numpy_rng"]
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["np_global_rng"])


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


def load_state(path: str) -> Dict[str, Any]:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def truncate_jsonl_after_step(path: str, step: int) -> int:
    """Drop records written after the saved step (field `step` = updates applied when written)."""
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as fh:
        lines = [l for l in fh if l.strip()]
    kept = [l for l in lines if json.loads(l).get("step", -1) <= step]
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(kept)
    return len(lines) - len(kept)
