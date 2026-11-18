"""The FedMelanoma-Novelty straggler environment, as a per-client paired trace for asynchronous algorithms.

sim/straggler.py is a byte-identical copy of the novelty server/straggler.py (tests/test_00 checks the
hash). Every draw goes through its StragglerManager.simulate_response, so the distributions, parameters
and draw order (Beta completion probability, Bernoulli failure, LogNormal latency) are the novelty ones:
    tier reliable       Beta(19,1)  LogNormal(1.39, 0.30)   median ~4 s
    tier moderate       Beta(7,3)   LogNormal(2.77, 0.50)   median ~16 s
    tier high_straggler Beta(2,3)   LogNormal(4.09, 0.80)   median ~60 s
Tiers: hospital index (sorted names) mod 3 over (reliable, moderate, high_straggler) -- hospital_registry.

What is NOT carried over: the adaptive window W_t, quorum, buffer and S_max. Those are the novelty's
reaction to stragglers; each algorithm here reacts by its own rules.

Pairing: client c's k-th job (k counts every dispatch of that client, failed or not) always gets the same
(failed, latency) pair, because client c has its own seeded generator. FedAsync and FedStrag therefore see the
same draws per client; only the order in which they consume them differs. The novelty run itself used ONE
shared generator consumed in selection order, so its exact draws cannot be replayed: same distributions and
parameters, not the same numbers.

FedAsync as published has no failure mechanism (arXiv:1903.03934, checked 2026-11-14): every triggered worker
trains and pushes. This copy therefore uses latency(c, k) only: the tier's LogNormal delay, no Beta/Bernoulli
failure. draw() is kept for reference and for the equality test with the novelty simulator.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np

from sim.straggler import StragglerManager

TRACE_STREAM = 7919   # fixed salt so the trace stream is independent of every other generator


class TraceEnvironment:
    def __init__(self, seed: int, records: Sequence[Dict]) -> None:
        self.stubs = [SimpleNamespace(client_id=i, tier=r["tier"], latency_mu=r["latency_mu"],
                                      latency_sigma=r["latency_sigma"]) for i, r in enumerate(records)]
        self.managers = [StragglerManager(rng=np.random.default_rng([int(seed), TRACE_STREAM, i]))
                         for i in range(len(records))]
        self.trace: List[List[Tuple[bool, float]]] = [[] for _ in records]
        # latency-only stream (FedAsync as published has no failure): LogNormal(mu, sigma) of the client's tier,
        # one seeded generator per client, independent of the failure trace above.
        self.lat_rngs = [np.random.default_rng([int(seed), TRACE_STREAM + 1, i]) for i in range(len(records))]
        self.lat_trace: List[List[float]] = [[] for _ in records]

    def draw(self, client_id: int, k: int) -> Tuple[bool, float]:
        """(failed, latency_seconds) for client `client_id`'s k-th job; deterministic in (seed, client, k)."""
        t = self.trace[client_id]
        while len(t) <= k:
            r = self.managers[client_id].simulate_response(self.stubs[client_id], len(t))
            t.append((bool(r.is_failed), float(r.response_time)))
        return t[k]

    def latency(self, client_id: int, k: int) -> float:
        """LogNormal latency of client `client_id`'s k-th job with NO failure draw: same distribution and
        parameters as draw(), but its own seeded stream, because draw() returns no latency for failed jobs."""
        t = self.lat_trace[client_id]
        stub = self.stubs[client_id]
        while len(t) <= k:
            t.append(float(self.lat_rngs[client_id].lognormal(mean=stub.latency_mu, sigma=stub.latency_sigma)))
        return t[k]

    def tiers(self) -> List[str]:
        return [s.tier for s in self.stubs]
