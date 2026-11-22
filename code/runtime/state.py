"""Atomic save and restore of everything needed to continue a run at a round boundary (D10).

Only pending weights of clients waiting in the straggler buffer are stored: any other pending
entry can never be consumed, because a client only reaches A_t after returning fresh weights
in the same round or by draining from the buffer.
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
    manager = strategy._straggler_manager
    buffered = set(manager.get_buffer_state().keys())
    return {
        "round": round_num,
        "parameters": server.parameters,
        "clients": strategy._clients,
        "manager": {
            "latency_history": list(manager._latency_history),
            "buffer": manager._buffer,
            "current_W_t": manager.current_W_t,
            "window_history": manager.window_history,
            "straggler_counts": manager.straggler_counts,
            "on_time_counts": manager.on_time_counts,
            "failed_counts": manager.failed_counts,
            "quorum_met_history": manager.quorum_met_history,
        },
        "pending": {cid: arrs for cid, arrs in strategy._pending_client_ndarrays.items() if cid in buffered},
        "best_state": strategy._best_checkpoint_state,
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
    manager, saved = strategy._straggler_manager, state["manager"]
    manager._latency_history.clear()
    manager._latency_history.extend(saved["latency_history"])
    manager._buffer = saved["buffer"]
    manager.current_W_t = saved["current_W_t"]
    for key in ("window_history", "straggler_counts", "on_time_counts", "failed_counts", "quorum_met_history"):
        setattr(manager, key, saved[key])
    strategy._pending_client_ndarrays = state["pending"]
    strategy._best_checkpoint_state = state["best_state"]
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
