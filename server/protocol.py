"""FedStrag's time-bounded asynchronous protocol (paper Algorithm 1 + official code), on an event-driven virtual
clock (one GPU), inside the FedMelanoma-Novelty straggler and failure environment (sim/environment.py).

Sources: Kumar & Srirama, Digital Communications and Networks 11(4):1214-1224, 2025 (Algorithm 1, Section 5.1,
Eqs. 6-8); official code github.com/cloud-and-smart-labs/FedStrag @ 9d645c2, FedStrag/async_FL_Server/server.py
(main, waitFunction, async_w_fedAvg) and FedStrag/async_FL_Client/client.py (on_message, train).

GLOBAL ITERATION g (g = 1 .. G, G = 50). One iteration = one broadcast followed by one server decision:
  1. Broadcast. The current global model is published to ALL 10 clients with tag t = g - 1 (server.py: the
     message carries 'iteration': iter, then iter += 1; Algorithm 1 lines 3-5). Every client appends it to its
     FIFO queue. t_g ("latest_tag") = t.
  2. Clients. A client works through its queue one broadcast at a time, OLDEST FIRST (client.py: paho runs
     on_message on one thread, so a broadcast that arrives while the client trains waits and is trained next,
     on its own, older model). Taking a broadcast off the queue = a DISPATCH. The client trains from that
     broadcast's model on the GPU (a completed local training). The environment then decides the outcome of the
     job, exactly as the novelty code does (it trains first, then simulates):
        * FAILED      -> the update is lost (never reaches the server). No penalty, no retry, no replacement.
                         The training already happened, so the job still occupies the client for one tier
                         LogNormal job duration (sim/environment.py, failed-job child stream); at that time the
                         loss is recorded and the client takes its next queued broadcast, if any.
        * DELIVERED   -> the update reaches the server `latency` seconds after the job started (tier LogNormal);
                         the client is busy until then. A successful update is never discarded for being old.
  3. Wait. The server polls every T_d = 4 s (server.py waitFunction: time.sleep(wait_time); wait_time = 4) and
     stops at the first poll at which >= 1 update (minimimum_clients = 1) has arrived.
  4. Aggregate. Every arrived update is combined with W*D weights, W = 1/(t_g - t_l + 1) (server/fedstrag.py);
     the result replaces the global model (model version + 1) and the server's update queue is emptied.
  EMPTY ITERATION. If at a poll no update has arrived AND none can arrive any more (no client is busy, neither
     with a job that will deliver nor with one that will fail; an idle client always has an empty queue), the
     iteration ends at that poll with the global model UNCHANGED (model version not incremented). While a failing
     job is still running the server keeps polling, as it cannot know the job will fail. The official server
     would poll forever once nothing can arrive; the paper's time-bounded rule ("aggregate all available updates
     after T_d") gives the unchanged model. It is counted (`empty_iterations`) and logged.

TERMINATION after iteration G (server.py: `if iter < no_iteration ... else: break`): nothing more is broadcast or
aggregated. Updates still in flight (already trained, not yet arrived), failing jobs still running (already trained
and already counted as failed attempts) and broadcasts still queued at clients (never started) are EXCLUDED and
reported. No update is ever waiting at the server at that point, because each iteration aggregates everything that
has arrived. Nothing is flushed, discarded or applied outside this rule.

Staleness: t_g = latest_tag (tag of the latest broadcast), t_l = tag the update was trained on. An update trained
on the latest broadcast has staleness 0 and weight 1 (paper Eq. 8 example).

Broadcast models waiting in client queues live on disk (SnapshotStore): with a FIFO backlog a slow client can hold
dozens of old ResNet-50 models. Files are deleted only when the run state is saved, so a resumed run always finds
every snapshot its queues reference.
"""
from __future__ import annotations

import math
import os
import pickle
import tempfile
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

import numpy as np

from server.fedstrag import MINIMUM_CLIENTS, WAIT_TIME_SECONDS, async_w_fedavg


@dataclass
class ClientJob:
    client_id: int
    tag: int
    job_index: int
    broadcast_time: float
    start_time: float
    finish_time: float
    duration: float
    arrays: object = field(repr=False)
    num_examples: int
    metrics: Dict
    started_in_iteration: int
    failed: bool = False     # True: trained, will be lost at finish_time (arrays is None)


class SnapshotStore:
    """Broadcast models by tag, on disk."""

    def __init__(self, directory: str) -> None:
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def path(self, tag: int) -> str:
        return os.path.join(self.directory, f"broadcast_{tag:06d}.pkl")

    def put(self, tag: int, arrays: Sequence[np.ndarray]) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.directory, prefix=".tmp_", suffix=".pkl")
        with os.fdopen(fd, "wb") as fh:
            pickle.dump(list(arrays), fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.path(tag))

    def get(self, tag: int) -> List[np.ndarray]:
        with open(self.path(tag), "rb") as fh:
            return pickle.load(fh)

    def tags(self) -> List[int]:
        return sorted(int(n[10:16]) for n in os.listdir(self.directory) if n.startswith("broadcast_") and n.endswith(".pkl"))

    def collect(self, keep: Sequence[int]) -> List[int]:
        keep = set(int(t) for t in keep)
        removed = [t for t in self.tags() if t not in keep]
        for t in removed:
            os.remove(self.path(t))
        return removed


TrainFn = Callable[[int, List[np.ndarray]], tuple]  # (client_id, arrays) -> (arrays, n, metrics, seconds)


class FedStragProtocol:
    # per-client counters; every dispatch trains, so dispatched == completed_trainings == failed_attempts + updates_generated
    COUNTERS = ("dispatched", "completed_trainings", "failed_attempts", "updates_generated", "updates_received", "applied")

    def __init__(self, initial: Sequence[np.ndarray], n_clients: int, env, global_iterations: int,
                 store: SnapshotStore, train_fn: TrainFn, wait_time: float = WAIT_TIME_SECONDS,
                 minimum_clients: int = MINIMUM_CLIENTS) -> None:
        if float(wait_time) <= 0 or int(minimum_clients) < 1 or int(global_iterations) < 1:
            raise ValueError("wait_time > 0, minimum_clients >= 1 and global_iterations >= 1 are required")
        self.x: List[np.ndarray] = [a.copy() for a in initial]
        self.n_clients = int(n_clients)
        self.env, self.global_iterations = env, int(global_iterations)
        self.store, self.train_fn = store, train_fn
        self.wait_time, self.minimum_clients = float(wait_time), int(minimum_clients)
        self.job_index: Dict[int, int] = {c: 0 for c in range(self.n_clients)}
        self.counts: Dict[str, Dict[int, int]] = {n: {c: 0 for c in range(self.n_clients)} for n in self.COUNTERS}
        self.iter: int = 0               # completed global iterations
        self.model_version: int = 0      # number of aggregations applied to the global model
        self.empty_iterations: int = 0
        self.latest_tag: int = -1        # t_g: tag of the latest broadcast (-1 before the first one)
        self.clock: float = 0.0
        self.queues: Dict[int, deque] = {c: deque() for c in range(self.n_clients)}
        self.busy: Dict[int, ClientJob] = {}
        self.arrived: List[ClientJob] = []
        self.events: List[Dict] = []     # drained by the runner for logging

    # ---- persistence: the trainer, the store and the environment are re-attached after loading --------------------
    def __getstate__(self):
        state = dict(self.__dict__)
        state["train_fn"], state["store"], state["env"] = None, None, None
        return state

    def totals(self) -> Dict[str, int]:
        t = {n: int(sum(self.counts[n].values())) for n in self.COUNTERS}
        t["in_flight_updates"] = sum(not j.failed for j in self.busy.values())   # trained, will arrive, not yet arrived
        t["failing_jobs_in_progress"] = sum(j.failed for j in self.busy.values())  # trained, will be lost (in failed_attempts)
        t["arrived_not_aggregated"] = len(self.arrived)            # 0 at every iteration end
        t["pending_updates"] = t["in_flight_updates"] + t["arrived_not_aggregated"]
        t["queued_broadcasts_not_started"] = int(sum(len(q) for q in self.queues.values()))
        t["updates_lost_to_failures"] = t["failed_attempts"]
        t["model_version"] = self.model_version
        t["empty_iterations"] = self.empty_iterations
        # invariant: every successful update is aggregated or still pending. Must be 0: nothing is discarded.
        t["unaccounted_updates"] = t["updates_generated"] - t["applied"] - t["pending_updates"]
        return t

    def done(self) -> bool:
        return self.iter >= self.global_iterations

    def referenced_tags(self) -> List[int]:
        return sorted({tag for q in self.queues.values() for tag, _ in q})

    def queue_lengths(self) -> Dict[int, int]:
        """Broadcasts each client still has to process: queued + the one it is working on."""
        return {c: len(self.queues[c]) + (c in self.busy) for c in range(self.n_clients)}

    # ---- one global iteration ---------------------------------------------------------------------------------------
    def run_iteration(self) -> Dict:
        if self.done():
            raise RuntimeError("all global iterations are done; nothing more is broadcast or aggregated")
        iteration = self.iter + 1
        tag = self.iter
        receivers = list(range(self.n_clients))
        broadcast_time = self.clock
        version_before = self.model_version
        self.latest_tag = tag
        self.store.put(tag, self.x)
        for c in receivers:
            self.queues[c].append((tag, broadcast_time))
        self.events.append({"event": "broadcast", "iteration": iteration, "tag": tag, "time": broadcast_time,
                            "model_version": version_before, "receivers": receivers,
                            "client_queue_lengths": self.queue_lengths()})
        for c in receivers:
            if c not in self.busy:
                self._start(c, broadcast_time, iteration)

        poll, poll_times, empty = broadcast_time, [], False
        while True:
            poll += self.wait_time
            poll_times.append(poll)
            self._advance(poll, iteration)
            self.events.append({"event": "poll", "iteration": iteration, "time": poll,
                                "server_queue_length": len(self.arrived), "clients_busy": sorted(self.busy)})
            if len(self.arrived) >= self.minimum_clients:
                break
            if not self.busy:
                empty = True     # nothing arrived and nothing can arrive: every job in flight failed
                break
        self.clock = poll

        aggregated = []
        if empty:
            self.empty_iterations += 1
            self.events.append({"event": "empty_iteration", "iteration": iteration, "time": poll,
                                "model_version": self.model_version})
        else:
            jobs = sorted(self.arrived, key=lambda j: (j.finish_time, j.client_id))   # server FIFO: arrival order
            self.arrived = []
            # t_g = latest_tag  <=>  iteration_now = latest_tag + 1 (server/fedstrag.py convention)
            self.x, weights = async_w_fedavg([j.arrays for j in jobs], [j.tag for j in jobs], self.latest_tag + 1,
                                             [j.num_examples for j in jobs])
            self.model_version += 1
            for position, (j, w) in enumerate(zip(jobs, weights)):
                aggregated.append({"client_id": j.client_id, "tag": j.tag, "job_index": j.job_index,
                                   "server_queue_position": position, "broadcast_time": j.broadcast_time,
                                   "start_time": j.start_time, "arrival_time": j.finish_time, "duration": j.duration,
                                   "num_examples": j.num_examples, "train_loss": j.metrics.get("train_loss"),
                                   "status": "applied", **w})
                j.arrays = None
                self.counts["applied"][j.client_id] += 1
        self.iter = iteration
        return {"iteration": iteration, "tag_broadcast": tag, "receivers": receivers, "broadcast_time": broadcast_time,
                "poll_times": poll_times, "polls": len(poll_times), "aggregated_at": self.clock,
                "aggregated": aggregated, "empty": empty, "latest_tag": self.latest_tag,
                "model_version_before": version_before, "model_version_after": self.model_version,
                "totals": self.totals(), "busy_after": sorted(self.busy), "client_queue_lengths": self.queue_lengths()}

    def _start(self, client_id: int, at: float, iteration: int) -> None:
        """The client takes its oldest queued broadcast, trains it, and the environment decides the job's outcome.

        Either way the client is busy for the job's duration: a delivered job until its update arrives, a failed job
        for one tier LogNormal job duration (outcome.busy_seconds), after which the loss is recorded in _advance.
        """
        if self.queues[client_id] and client_id not in self.busy:
            tag, broadcast_time = self.queues[client_id].popleft()
            k = self.job_index[client_id]
            self.job_index[client_id] += 1
            self.counts["dispatched"][client_id] += 1
            start = max(at, broadcast_time)
            arrays = self.store.get(tag)
            new_arrays, n, metrics, seconds = self.train_fn(client_id, arrays)
            del arrays
            self.counts["completed_trainings"][client_id] += 1
            outcome = self.env.outcome(client_id, k)
            record = {"iteration": iteration, "client_id": client_id, "tag": tag, "tier": self.env.stubs[client_id].tier,
                      "job_index": k, "broadcast_time": broadcast_time, "start_time": start,
                      "measured_fit_seconds": seconds, "num_examples": n,
                      "client_queue_left": len(self.queues[client_id]),
                      **{kk: v for kk, v in metrics.items() if isinstance(v, (int, float, bool))}}
            if outcome.failed:
                if not (math.isfinite(outcome.busy_seconds) and outcome.busy_seconds > 0):
                    raise RuntimeError(f"environment returned an invalid failed-job duration {outcome.busy_seconds}")
                self.counts["failed_attempts"][client_id] += 1
                del new_arrays
                self.busy[client_id] = ClientJob(client_id=client_id, tag=tag, job_index=k, broadcast_time=broadcast_time,
                                                 start_time=start, finish_time=start + outcome.busy_seconds,
                                                 duration=outcome.busy_seconds, arrays=None, num_examples=n,
                                                 metrics=dict(metrics), started_in_iteration=iteration, failed=True)
                self.events.append({"event": "client_failed", "status": "trained_update_will_be_lost", "failed": True,
                                    "duration": outcome.busy_seconds, "finish_time": start + outcome.busy_seconds,
                                    **record, **self.totals()})
                return
            if not (math.isfinite(outcome.latency) and outcome.latency > 0):
                raise RuntimeError(f"environment returned an invalid delay {outcome.latency} for a delivered job")
            self.counts["updates_generated"][client_id] += 1
            self.busy[client_id] = ClientJob(client_id=client_id, tag=tag, job_index=k, broadcast_time=broadcast_time,
                                             start_time=start, finish_time=start + outcome.latency,
                                             duration=outcome.latency, arrays=new_arrays, num_examples=n,
                                             metrics=dict(metrics), started_in_iteration=iteration)
            self.events.append({"event": "client_start", "status": "trained_in_flight", "failed": False,
                                "duration": outcome.latency, "finish_time": start + outcome.latency,
                                **record, **self.totals()})

    def _advance(self, until: float, iteration: int) -> None:
        """Deliver every update finishing by `until`, in finish-time order; a freed client starts its next job."""
        while True:
            due = [(job.finish_time, c) for c, job in self.busy.items() if job.finish_time <= until]
            if not due:
                return
            _, c = min(due)
            job = self.busy.pop(c)
            if job.failed:
                self.events.append({"event": "update_lost", "iteration": iteration, "client_id": c, "tag": job.tag,
                                    "job_index": job.job_index, "failure_time": job.finish_time,
                                    "status": "failed_update_lost"})
                if self.queues[c]:
                    self._start(c, job.finish_time, iteration)
                continue
            self.arrived.append(job)
            self.counts["updates_received"][c] += 1
            self.events.append({"event": "update_arrived", "iteration": iteration, "client_id": c, "tag": job.tag,
                                "job_index": job.job_index, "completion_time": job.finish_time,
                                "num_examples": job.num_examples, "server_queue_length": len(self.arrived),
                                "status": "arrived_waiting_for_aggregation"})
            if self.queues[c]:
                self._start(c, job.finish_time, iteration)
