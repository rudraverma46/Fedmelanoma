"""FedStrag: straggler-aware weighted averaging of stale updates, time-bounded asynchronous protocol.

Kumar and Srirama, "FedStrag: Straggler-aware federated learning for low resource devices", Digital
Communications and Networks 11(4):1214-1224, 2025, DOI 10.1016/j.dcan.2024.12.004 (paper read 2026-11-14).
Official code: github.com/cloud-and-smart-labs/FedStrag, commit 9d645c216e491789213e3c2e8c6b95d8dce566fb,
FedStrag/async_FL_Server/server.py (async_w_fedAvg) and async_FL_Client/client.py.

AGGREGATION RULE USED HERE = THE PAPER (user decision 2026-11-14), which differs from the official code:

    W_f   = 1 / (t_g - t_l^f + 1)                                  paper Eq. (6)
    theta = sum_f theta_f * W_f * D_f / sum_f W_f * D_f             paper Eq. (7), Algorithm 1 lines 12-16

    t_g  : the current global iteration, i.e. the tag of the latest REAL broadcast (it does not advance in
           drain iterations, where nothing is broadcast)
    t_l  : the iteration (tag) of the broadcast the client trained on
    D_f  : the number of training samples the client used
The paper's worked example (Eq. 8: tags 4, 4, 3 at the aggregation -> W = [1, 1, 0.5]) fixes the indexing: an
update trained on the latest broadcast has staleness 0 and weight 1, one iteration older 1/2, then 1/3.

Where the official code differs (kept out of this run, listed for the report):
  * the code weights by staleness only (np.average with weights 1/(iter - tag + 1), iter already incremented,
    so the freshest update gets 1/2); the paper also weights by D_f and gives the freshest update weight 1.
    Equal D_f (the paper's clients all train on 60 samples) hides the first difference; our hospitals have
    155 to 1,716 samples, so it matters here.
  * the code runs 32 local epochs, the paper reports 16; this study uses 5 for every method.

Protocol parts from the paper (Algorithm 1, Section 5.1): every iteration the server sends the model and its
iteration number to ALL clients, waits T_d, then aggregates every update available and replaces the global
model with the result. Parts that come from the code only: the 4 s poll until >= 1 update is queued, and the
FIFO backlog (a client trains every broadcast it received, oldest first, so a slow client's updates grow
stale). Delayed updates are never discarded: they wait in the queue and are aggregated with a smaller weight.
In resnet_50_fedstrag_final the novelty failure draw is active (brief 2026-11-20): a FAILED job's update never
reaches the server, so it never enters this function; every update that does arrive is aggregated.
The official code's committed main() calls fedAvg() with async_w_fedAvg() commented out: the published repo
runs its FedAvg baseline by default; FedStrag is async_w_fedAvg (server.py lines 118-146 at 9d645c2).

Integer tensors (BatchNorm num_batches_tracked, which Keras models do not have) are averaged the same way
and rounded to the nearest integer; the counter is never read by the model.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

OFFICIAL_REPO = "https://github.com/cloud-and-smart-labs/FedStrag"
OFFICIAL_COMMIT = "9d645c216e491789213e3c2e8c6b95d8dce566fb"
WAIT_TIME_SECONDS = 4.0   # server.py:44  wait_time = 4
MINIMUM_CLIENTS = 1       # server.py:48  minimimum_clients = 1


def staleness_weights(tags: Sequence[int], iteration_now: int) -> np.ndarray:
    """Eq. (6): W = 1 / (t_g - t_l + 1), t_g = tag of the latest real broadcast = iteration_now - 1."""
    staleness = np.array([(iteration_now - 1) - int(t) for t in tags], dtype=np.int64)
    if (staleness < 0).any():
        raise ValueError("an update cannot be newer than the latest broadcast")
    return 1.0 / (staleness.astype(np.float64) + 1.0)


def async_w_fedavg(models: Sequence[Sequence[np.ndarray]], tags: Sequence[int], iteration_now: int,
                   num_examples: Sequence[int]) -> Tuple[List[np.ndarray], List[Dict]]:
    """Eq. (7): theta = sum(theta_f W_f D_f) / sum(W_f D_f), tensor by tensor; the result replaces the global model."""
    if len(models) < MINIMUM_CLIENTS:
        raise ValueError("aggregation runs only once at least one update is queued")
    if not (len(models) == len(tags) == len(num_examples)):
        raise ValueError("models, tags and sample counts must have the same length")
    w = staleness_weights(tags, iteration_now)
    d = np.array([float(n) for n in num_examples], dtype=np.float64)
    if (d <= 0).any():
        raise ValueError("every update needs a positive sample count")
    coeff = w * d
    out = []
    for tensors in zip(*models):
        stacked = np.stack([t.astype(np.float64) for t in tensors])
        avg = np.average(stacked, axis=0, weights=coeff)
        if tensors[0].dtype.kind in "iu":
            avg = np.rint(avg)
        out.append(avg.astype(tensors[0].dtype))
    total = float(coeff.sum())
    record = [{"tag": int(t), "staleness": int((iteration_now - 1) - t), "raw_weight": float(wi), "num_examples_weight": float(di),
               "normalised_weight": float(c / total)} for t, wi, di, c in zip(tags, w, d, coeff)]
    return out, record
