"""FedAsync's scheduler and the event-driven virtual clock that makes the run asynchronous on one GPU.

The paper's server runs a scheduler thread (triggers idle workers, sending them the current model and
its version) and an updater thread (applies each pushed update as soon as it arrives). Here:

* `concurrency` workers are busy at any moment. Whenever one becomes idle, the scheduler triggers a
  worker chosen uniformly at random among the idle ones (seeded numpy Generator) and sends it the current
  global model x_v with its version v.
* The worker's training is computed immediately on the GPU, from exactly that x_v, and its measured wall
  time d becomes its duration on a virtual clock: it arrives at dispatch_time + d. Computing it early
  cannot change it: its inputs (x_v, the client's data, the RNG stream at that moment) are fixed at
  dispatch, exactly as a real worker that received x_v would use them.
* The updater pops updates in arrival-time order (ties: dispatch order) and applies each one at once.
  The server never waits for the other in-flight workers, so a fast client can contribute several
  updates while a slow one is still training on an old version: real staleness, not a synchronous round.

Only the scheduling and clock live here; the update rule is server/fedasync.py.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass(order=True)
class Job:
    arrival_time: float
    seq: int
    client_id: int = field(compare=False)
    tau: int = field(compare=False)
    dispatch_time: float = field(compare=False)
    duration: float = field(compare=False)
    arrays: Any = field(compare=False, repr=False)
    num_examples: int = field(compare=False)
    metrics: Dict = field(compare=False, default_factory=dict)


class VirtualClockScheduler:
    def __init__(self, client_ids: Sequence[int], concurrency: int, rng: np.random.Generator) -> None:
        if not 0 < concurrency <= len(client_ids):
            raise ValueError("concurrency must be between 1 and the number of clients")
        self.idle: List[int] = sorted(int(c) for c in client_ids)
        self.concurrency = int(concurrency)
        self.rng = rng
        self.clock: float = 0.0
        self.seq: int = 0
        self.in_flight: List[Job] = []

    def pick_idle(self) -> int:
        """Uniform over idle clients; the drawn client becomes busy."""
        client_id = int(self.rng.choice(self.idle))
        self.idle.remove(client_id)
        return client_id

    def submit(self, client_id: int, tau: int, duration: float, arrays, num_examples: int, metrics: Dict) -> Job:
        if duration <= 0:
            raise ValueError("a job needs a positive duration")
        job = Job(arrival_time=self.clock + float(duration), seq=self.seq, client_id=int(client_id), tau=int(tau),
                  dispatch_time=self.clock, duration=float(duration), arrays=arrays, num_examples=int(num_examples),
                  metrics=dict(metrics))
        self.seq += 1
        heapq.heappush(self.in_flight, job)
        return job

    def next_arrival(self) -> Job:
        job = heapq.heappop(self.in_flight)
        if job.arrival_time < self.clock:
            raise RuntimeError("virtual clock would run backwards")
        self.clock = job.arrival_time
        self.idle.append(job.client_id)
        self.idle.sort()
        return job

    def busy(self) -> List[int]:
        return sorted(j.client_id for j in self.in_flight)

    def peek(self) -> Optional[Job]:
        return self.in_flight[0] if self.in_flight else None
