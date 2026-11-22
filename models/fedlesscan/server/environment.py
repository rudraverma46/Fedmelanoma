"""The straggler environment: per-client seeded Beta / Bernoulli / LogNormal trace (the three tiers of the Novelty run).

This is ENVIRONMENT only (what happens to a client's job), not an algorithm. It reproduces the draws of the novelty
StragglerManager.simulate_response, in the same order and with the same distributions:
    completion_prob ~ Beta(a, b) of the tier; failed = (uniform > completion_prob); latency ~ LogNormal(mu, sigma) if not failed
    tier reliable       Beta(19,1)  LogNormal(1.39, 0.30)   median ~4 s
    tier moderate       Beta(7,3)   LogNormal(2.77, 0.50)   median ~16 s
    tier high_straggler Beta(2,3)   LogNormal(4.09, 0.80)   median ~60 s
tests/test_00 checks draw-for-draw equality with the novelty simulate_response (seed 42, stream 7919, per client).
Nothing of the novelty scheduler (selector, adaptive window, buffer, aggregation) is imported or reproduced here.

Client c's k-th job (every dispatch, failed or not) always gets the same (failed, latency) pair, because client c has its
own generator seeded with [seed, TRACE_STREAM, c]. Same seed and stream as the FedAsync / FedStrag novelty-environment runs.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

TRACE_STREAM = 7919   # fixed salt so the trace stream is independent of every other generator
BETA_PARAMS: Dict[str, Tuple[float, float]] = {"reliable": (19.0, 1.0), "moderate": (7.0, 3.0), "high_straggler": (2.0, 3.0)}
DEFAULT_BETA = (10.0, 2.0)   # unknown tier, as in the novelty file


class TraceEnvironment:
    def __init__(self, seed: int, records: Sequence[Dict]) -> None:
        self.records = [dict(r) for r in records]
        self.rngs = [np.random.default_rng([int(seed), TRACE_STREAM, i]) for i in range(len(records))]
        self.trace: List[List[Tuple[bool, float]]] = [[] for _ in records]

    def draw(self, client_id: int, k: int) -> Tuple[bool, float]:
        """(failed, latency_seconds) of client `client_id`'s k-th job; latency is inf for a failed job."""
        t = self.trace[client_id]
        rec, rng = self.records[client_id], self.rngs[client_id]
        while len(t) <= k:
            a, b = BETA_PARAMS.get(rec["tier"], DEFAULT_BETA)
            completion_prob = float(rng.beta(a, b))
            failed = bool(rng.random() > completion_prob)
            if failed:
                t.append((True, float("inf")))
            else:
                t.append((False, float(rng.lognormal(mean=rec["latency_mu"], sigma=rec["latency_sigma"]))))
        return t[k]

    def tiers(self) -> List[str]:
        return [r["tier"] for r in self.records]
