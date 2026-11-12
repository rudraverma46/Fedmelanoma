"""
monitoring/checkpointing.py
=============================
Server-side GLOBAL model checkpointing for the federated runtime.

This is deliberately separate from client/local_trainer.py's
save_checkpoint() (centralized, per-epoch, client/local model) -- that
function is untouched. This module saves the AGGREGATED GLOBAL model
produced by FedMelanomaStrategy.aggregate_fit(), never a client-local
model, mirroring the same on-disk dict shape
(epoch/round, model_state_dict, metrics) local_trainer.py already uses so
downstream tooling (evaluate.py, notebooks) can load either checkpoint
family the same way.

Primary selection criterion for best_global_model.pth: validation
ROC-AUC, exactly matching train.py's own criterion (`best_roc_auc`) --
per the blueprint (Section 5.1: "AUROC ... appropriate as the primary
metric" under 5.55% prevalence). No new selection criterion is invented.

Checkpoints are namespaced by run_id (checkpoints/<run_id>/...) so that
re-running the experiment never silently overwrites a previous run's
checkpoints, while each individual filename still follows the requested
`global_round_XXXX.pth` pattern within its run's folder.

Writes are atomic: a checkpoint is fully written to a temp file in the
same directory, then moved into place with os.replace(), which is atomic
on both POSIX and Windows -- a crash or interrupted write can never leave
a corrupt/partial `global_round_XXXX.pth` behind.
"""

from __future__ import annotations

import os
import tempfile
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from monitoring.param_hash import hash_ndarrays

DEFAULT_CHECKPOINT_DIR = "checkpoints"

# Blueprint Section 5.1 / train.py's own convention: ROC-AUC is the
# primary model-selection criterion.
BEST_MODEL_METRIC = "roc_auc"


def _atomic_torch_save(obj: dict, final_path: str) -> None:
    """Write *obj* to *final_path* atomically (temp file + os.replace)."""
    directory = os.path.dirname(final_path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp_ckpt_", suffix=".pth")
    os.close(fd)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, final_path)  # atomic on POSIX and Windows
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _build_checkpoint_dict(
    round_num: int,
    ndarrays: Sequence[np.ndarray],
    param_keys: Optional[Sequence[str]],
    metrics: Optional[Dict[str, float]],
    run_id: str,
) -> dict:
    param_hash = hash_ndarrays(ndarrays)

    tensors = [torch.tensor(np.asarray(arr)) for arr in ndarrays]

    if param_keys is not None and len(param_keys) == len(ndarrays):
        model_state_dict = OrderedDict(zip(param_keys, tensors))
    else:
        # No key names available -- still fully loadable (see
        # load_global_checkpoint), just under a generic key so the file
        # is self-describing rather than silently wrong.
        model_state_dict = None

    return {
        "round": int(round_num),
        "model_state_dict": model_state_dict,
        # Parameters stored as torch.Tensor (not raw numpy arrays): torch's
        # default torch.load(weights_only=True) (PyTorch >=2.6) only
        # allowlists a small set of safe globals, which does not include
        # numpy's array-reconstruction function -- tensors ARE in that
        # allowlist, so checkpoints stay loadable under the secure default
        # without relaxing it. Order == model.state_dict().values() order,
        # exactly like every parameter exchange elsewhere in this codebase,
        # so a checkpoint saved before param_keys were known is still
        # fully reconstructable.
        "parameters": tensors,
        "metrics": dict(metrics) if metrics else {},
        "param_hash": param_hash,
        "timestamp": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": run_id,
        "strategy": "FedMelanomaStrategy",
    }


def save_global_checkpoint(
    round_num: int,
    ndarrays: Sequence[np.ndarray],
    run_id: str,
    metrics: Optional[Dict[str, float]] = None,
    param_keys: Optional[Sequence[str]] = None,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
) -> str:
    """
    Save the aggregated GLOBAL model for one completed round.

    Safe to call twice for the same round (e.g. once right after
    aggregate_fit with fit-only metrics, once more after
    aggregate_evaluate adds validation metrics) -- each call atomically
    overwrites checkpoints/<run_id>/global_round_XXXX.pth with the fuller
    dict.

    Returns the path written.
    """
    run_dir = os.path.join(checkpoint_dir, run_id)
    path = os.path.join(run_dir, f"global_round_{round_num:04d}.pth")
    checkpoint = _build_checkpoint_dict(round_num, ndarrays, param_keys, metrics, run_id)
    _atomic_torch_save(checkpoint, path)
    return path


def maybe_update_best_checkpoint(
    round_num: int,
    ndarrays: Sequence[np.ndarray],
    run_id: str,
    metrics: Dict[str, float],
    best_state: Dict[str, float],
    param_keys: Optional[Sequence[str]] = None,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    metric_name: str = BEST_MODEL_METRIC,
) -> bool:
    """
    Update checkpoints/<run_id>/best_global_model.pth if *metrics[metric_name]*
    improves on the best seen so far.

    *best_state* is a small mutable dict the caller keeps across rounds,
    e.g. {"best_value": -inf}; this function reads/writes
    best_state["best_value"] in place instead of this module owning any
    global mutable state itself.

    Returns True if a new best was written.
    """
    if metric_name not in metrics:
        return False
    value = metrics[metric_name]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False

    current_best = best_state.get("best_value", float("-inf"))
    if value <= current_best:
        return False

    best_state["best_value"] = value
    run_dir = os.path.join(checkpoint_dir, run_id)
    path = os.path.join(run_dir, "best_global_model.pth")
    checkpoint = _build_checkpoint_dict(round_num, ndarrays, param_keys, metrics, run_id)
    checkpoint["best_metric_name"] = metric_name
    checkpoint["best_metric_value"] = value
    _atomic_torch_save(checkpoint, path)
    return True


def load_global_checkpoint(path: str, device: Optional[torch.device] = None):
    """
    Load a checkpoint saved by save_global_checkpoint()/
    maybe_update_best_checkpoint().

    Returns (ndarrays, checkpoint_dict). Loading into a fresh
    FedMelanomaModel:

        from models.model import FedMelanomaModel
        ndarrays, ckpt = load_global_checkpoint(path)
        model = FedMelanomaModel()
        state_dict = OrderedDict(zip(model.state_dict().keys(),
                                      [torch.tensor(a) for a in ndarrays]))
        model.load_state_dict(state_dict)

    (done this way, from the freshly-instantiated model's own key order,
    rather than trusting the checkpoint's own model_state_dict keys
    verbatim, so a checkpoint remains loadable even if it predates
    param_keys being threaded through).
    """
    checkpoint = torch.load(path, map_location=device or "cpu")
    ndarrays: List[np.ndarray] = [
        a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
        for a in checkpoint["parameters"]
    ]
    return ndarrays, checkpoint
