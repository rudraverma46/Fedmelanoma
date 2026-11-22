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

FedAsync as published has no failure mechanism (arXiv:1903.03934, checked 2026-11-14). Until 2026-11-21 this copy
therefore used latency(c, k) only (TraceEnvironment.latency, kept for the tests and for reference).

User decision 2026-11-21: simulate failures here as in every other folder. The paper gives no rule for handling a
failure, so the rule of the other paper-silent baseline (resnet_50_fedstrag, sim/environment.py, correction
approved 2026-11-20) is used unchanged, through FailureEnvironment below, a verbatim copy of FedStrag's
NoveltyEnvironment:
    1. completion probability  p ~ Beta(a, b) of the client's tier, then failed = U(0, 1) > p (novelty simulate_response)
    2. timing: the client trains FIRST (novelty code); a failed update is then lost, never applied, never retried
    3. a failed job keeps its client busy for one LogNormal draw of its tier (child stream [seed, 7921, c, 1]);
       then the slot frees and the scheduler triggers a uniformly random idle client, as after any arrival
    4. no penalty: the client stays eligible; the failed job still counts as one of the 300 dispatches
Same TRACE_STREAM (7921) as FedStrag, so client c's k-th job has the SAME outcome (failed or not, delay) in FedAsync
and FedStrag: paired environments.
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


# ---------------------------------------------------------------------------------------------------------------
# Failure environment (user decision 2026-11-21). Verbatim copy of resnet_50_fedstrag sim/environment.py
# (NoveltyEnvironment, renamed FailureEnvironment; constants renamed FAILURE_TRACE_STREAM so TraceEnvironment keeps 7919).
# ---------------------------------------------------------------------------------------------------------------
from dataclasses import dataclass  # noqa: E402

FAILURE_TRACE_STREAM = 7921   # = FedStrag TRACE_STREAM: paired failure/delay outcomes per (client, job index)
FAILED_DURATION_SUBSTREAM = 1   # child stream [seed, TRACE_STREAM, client, 1]: durations of failed jobs only


@dataclass(frozen=True)
class JobOutcome:
    failed: bool
    latency: float          # seconds until the update reaches the server; inf when failed (novelty response_time)
    busy_seconds: float     # seconds the client is occupied by this job: = latency if delivered, the failed-job
                            # duration (tier LogNormal, child stream) if failed


class FailureEnvironment:
    def __init__(self, seed: int, records: Sequence[Dict]) -> None:
        self.seed = int(seed)
        self.stubs = [SimpleNamespace(client_id=i, tier=r["tier"], latency_mu=r["latency_mu"],
                                      latency_sigma=r["latency_sigma"]) for i, r in enumerate(records)]
        self.managers = [StragglerManager(rng=np.random.default_rng([self.seed, FAILURE_TRACE_STREAM, i]))
                         for i in range(len(records))]
        self.failed_rngs = [np.random.default_rng([self.seed, FAILURE_TRACE_STREAM, i, FAILED_DURATION_SUBSTREAM])
                            for i in range(len(records))]
        self.trace: List[List[JobOutcome]] = [[] for _ in records]
        self.failed_durations: List[List[float]] = [[] for _ in records]

    def failed_duration(self, client_id: int, k: int) -> float:
        """Busy time of client `client_id`'s k-th job IF it fails: LogNormal(mu, sigma) of the client's tier.

        Drawn for every job index k (used only when job k fails), so the value depends on (seed, client, k) only.
        """
        t = self.failed_durations[client_id]
        stub = self.stubs[client_id]
        while len(t) <= k:
            t.append(float(self.failed_rngs[client_id].lognormal(mean=stub.latency_mu, sigma=stub.latency_sigma)))
        return t[k]

    def outcome(self, client_id: int, k: int) -> JobOutcome:
        """Outcome of client `client_id`'s k-th job (k counts every dispatch of that client, from 0)."""
        t = self.trace[client_id]
        while len(t) <= k:
            j = len(t)
            response = self.managers[client_id].simulate_response(client=self.stubs[client_id], current_round=j)
            failed, latency = bool(response.is_failed), float(response.response_time)
            t.append(JobOutcome(failed=failed, latency=latency,
                                busy_seconds=self.failed_duration(client_id, j) if failed else latency))
        return t[k]

    def tiers(self) -> List[str]:
        return [s.tier for s in self.stubs]
