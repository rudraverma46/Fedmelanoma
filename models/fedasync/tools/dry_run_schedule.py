"""CPU dry run of the FedAsync schedule with failures (no training, no data needed).

Every quantity that decides the schedule is virtual: the scheduler draws from default_rng(seed), and each job's
outcome and duration come from sim/environment.FailureEnvironment, whose tiers depend only on the hospital index.
This replays runtime/run_fedasync.py's dispatch / arrival / refill order exactly and prints the counts the real
run will log (dispatched, lost, applied, evaluation points), per tier and per client.

    python tools/dry_run_schedule.py [--clients 10] [--dispatches 300] [--concurrency 6] [--every 6] [--seed 42]
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from hospital_registry import _TIER_CYCLE, _TIER_LATENCY  # noqa: E402
from server.scheduler import VirtualClockScheduler  # noqa: E402
from sim.environment import FailureEnvironment  # noqa: E402


def simulate(clients: int, dispatches: int, concurrency: int, every: int, seed: int) -> dict:
    records = []
    for i in range(clients):
        tier = _TIER_CYCLE[i % len(_TIER_CYCLE)]
        mu, sigma = _TIER_LATENCY[tier]
        records.append({"tier": tier, "latency_mu": mu, "latency_sigma": sigma})
    env = FailureEnvironment(seed, records)
    sched = VirtualClockScheduler(range(clients), concurrency, np.random.default_rng(seed))
    job_index, dispatched, lost, applied = Counter(), Counter(), Counter(), Counter()
    evals, window = 0, 0

    def refill() -> None:
        while len(sched.in_flight) < concurrency and sum(dispatched.values()) < dispatches:
            c = sched.pick_idle()
            k = job_index[c]
            job_index[c] += 1
            dispatched[c] += 1
            o = env.outcome(c, k)
            sched.submit(c, 0, o.busy_seconds, None, 0, {"failed": True} if o.failed else {})

    refill()
    n_applied = 0
    while sched.in_flight:
        job = sched.next_arrival()
        is_lost = bool(job.metrics.get("failed"))
        if is_lost:
            lost[job.client_id] += 1
        else:
            applied[job.client_id] += 1
            n_applied += 1
            window += 1
        refill()
        finished = not sched.in_flight
        if (not is_lost and n_applied % every == 0) or (finished and window):
            evals += 1
            window = 0
    return {"records": records, "dispatched": dispatched, "lost": lost, "applied": applied,
            "evaluations": evals, "virtual_seconds": sched.clock}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clients", type=int, default=10)
    p.add_argument("--dispatches", type=int, default=300)
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--every", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    r = simulate(a.clients, a.dispatches, a.concurrency, a.every, a.seed)
    tot = {k: sum(r[k].values()) for k in ("dispatched", "lost", "applied")}
    print(f"dispatched {tot['dispatched']}  lost {tot['lost']}  applied {tot['applied']}  "
          f"evaluation points {r['evaluations']}  virtual time {r['virtual_seconds']:.0f} s")
    print("client tier            dispatched  lost  applied")
    for c, rec in enumerate(r["records"]):
        print(f"{c:>6} {rec['tier']:<16} {r['dispatched'][c]:>10} {r['lost'][c]:>5} {r['applied'][c]:>8}")


if __name__ == "__main__":
    main()
