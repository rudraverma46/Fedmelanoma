"""
monitoring/param_hash.py
=========================
Deterministic parameter fingerprints, used to PROVE (not assume) the
round-to-round broadcast/aggregation lifecycle:

    server broadcasts w^t              -> hash_ndarrays(w^t)
    client receives parameters         -> hash_ndarrays(received) [computed
                                           CLIENT-SIDE, in client/client.py,
                                           and echoed back in fit() metrics --
                                           this is the only way to prove the
                                           client actually got what the
                                           server sent, rather than trusting
                                           the server's own copy of what it
                                           meant to send]
    client returns w_i^(t+1)           -> hash_ndarrays(returned)
    strategy aggregates -> w^(t+1)     -> hash_ndarrays(aggregated)
    next round broadcasts w^(t+1)      -> hash_ndarrays(next w^t) -- must
                                           equal the previous round's
                                           aggregated hash

No research module imports this -- it is pure adapter/monitoring code
consumed by server/strategy.py and client/client.py.
"""

from __future__ import annotations

import hashlib
from typing import List, Sequence

import numpy as np


def hash_ndarrays(arrays: Sequence[np.ndarray]) -> str:
    """
    SHA-256 fingerprint of an ordered list of ndarrays.

    Includes each array's shape and dtype in the hashed stream (not just
    raw bytes) so that two differently-shaped arrays that happen to share
    a byte pattern cannot collide, and empty arrays still contribute
    something to the digest.
    """
    hasher = hashlib.sha256()
    hasher.update(str(len(arrays)).encode("utf-8"))
    for arr in arrays:
        a = np.asarray(arr)
        hasher.update(str(a.shape).encode("utf-8"))
        hasher.update(str(a.dtype).encode("utf-8"))
        hasher.update(a.tobytes())
    return hasher.hexdigest()


def hash_parameters(parameters) -> str:
    """
    Hash a flwr.common.Parameters object directly (avoids every call site
    needing its own parameters_to_ndarrays import).
    """
    from flwr.common import parameters_to_ndarrays

    return hash_ndarrays(parameters_to_ndarrays(parameters))


def short_hash(full_hash: str, length: int = 12) -> str:
    """First N hex chars, for compact human-readable log lines."""
    return full_hash[:length]
