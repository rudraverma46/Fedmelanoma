"""The FedMelanoma-Novelty straggler and failure environment, as a per-client trace for FedStrag.

sim/straggler.py is a byte-identical copy of the novelty server/straggler.py (tests/test_04 checks the hash).
Every job outcome is produced by calling the novelty's own StragglerManager.simulate_response, unmodified:

    1. completion probability  p ~ Beta(a, b) of the client's tier      reliable (19, 1)  moderate (7, 3)  high_straggler (2, 3)
    2. failure                 failed = U(0, 1) > p                     (no update; response_time = inf)
    3. delay, if not failed    LogNormal(mu, sigma) of the tier         (1.39, 0.30)  (2.77, 0.50)  (4.09, 0.80)
                                                                         medians ~4 s, ~16 s, ~60 s

Tiers and LogNormal parameters come from hospital_registry (byte-identical to the novelty file): hospital index
(sorted names) mod 3 over (reliable, moderate, high_straggler), i.e. 4 / 3 / 3 hospitals.

Only `is_failed` and `response_time` are read. simulate_response also labels the job on-time / straggler against
the manager's window W_t; that label belongs to the novelty's reaction to stragglers and is ignored here. The
novelty's window update, quorum, buffer, S_max expiry and staleness discount (manage_window, _update_window,
_drain_buffer, apply_staleness_weight) are never called (tests/test_04 checks this).

Time consumed by a FAILED job (correction approved 2026-11-20). The client trains first and the update is lost
afterwards (novelty timing), so the computation time already consumed is a full job. The existing delay model of
one job is the tier LogNormal(mu, sigma); a failed job therefore keeps its client busy for one LogNormal draw of its
own tier, after which the client takes its next queued broadcast. simulate_response draws no delay for a failure,
and drawing one from the main stream would shift every later draw; the durations of failed jobs come instead from a
per-client CHILD stream default_rng([seed, TRACE_STREAM, c, 1]), so the main stream 7921 -- every failure/success
outcome and every successful delay -- is exactly what it was before this correction. Failure probabilities and the
LogNormal parameters are unchanged; no new distribution or parameter is introduced.

Determinism: client c has its own generator default_rng([seed, TRACE_STREAM, c]) and its own manager, so the
k-th job of client c always gets the same outcome, whatever order clients are served in. The stream differs from
the delay-only stream of resnet50_fedasync / resnet50_fedstrag (salt 7919 + 1): with a failure draw each job
consumes three numbers (Beta, uniform, LogNormal) instead of one, exactly as in the novelty code. The novelty run
itself used ONE generator for all clients in selection order, so its exact draws cannot be replayed: same
distributions, not the same numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Sequence

import numpy as np

from sim.straggler import StragglerManager

TRACE_STREAM = 7921   # fixed salt, distinct from the delay-only stream 7919 + 1 used by the FedAsync / earlier FedStrag runs


FAILED_DURATION_SUBSTREAM = 1   # child stream [seed, TRACE_STREAM, client, 1]: durations of failed jobs only


@dataclass(frozen=True)
class JobOutcome:
    failed: bool
    latency: float          # seconds until the update reaches the server; inf when failed (novelty response_time)
    busy_seconds: float     # seconds the client is occupied by this job: = latency if delivered, the failed-job
                            # duration (tier LogNormal, child stream) if failed


class NoveltyEnvironment:
    def __init__(self, seed: int, records: Sequence[Dict]) -> None:
        self.seed = int(seed)
        self.stubs = [SimpleNamespace(client_id=i, tier=r["tier"], latency_mu=r["latency_mu"],
                                      latency_sigma=r["latency_sigma"]) for i, r in enumerate(records)]
        self.managers = [StragglerManager(rng=np.random.default_rng([self.seed, TRACE_STREAM, i]))
                         for i in range(len(records))]
        self.failed_rngs = [np.random.default_rng([self.seed, TRACE_STREAM, i, FAILED_DURATION_SUBSTREAM])
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
