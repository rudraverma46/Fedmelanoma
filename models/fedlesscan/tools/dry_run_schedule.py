"""Compute the whole 50-round FedLesScan schedule without any training.

In this experiment the schedule does not depend on the model: failures and latencies come from the seeded novelty
trace (server/environment.py), training times for clustering are those simulated latencies, and the deadline is
fixed. So selection, on-time/late/failed outcomes, late arrivals and the aggregation candidates of every round can
be computed in advance. This tool drives the REAL FedMelanomaStrategy with fake clients and tiny dummy parameters,
so the plan uses exactly the code of the run. tests/test_03_real_round.py compares the real run's logs to it.

Writes results/schedule/planned_schedule.json.
Usage: python tools/dry_run_schedule.py [--seed 42]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from flwr.common import Code, FitRes, Status, ndarrays_to_parameters  # noqa: E402
from flwr.server.client_manager import SimpleClientManager  # noqa: E402
from flwr.server.client_proxy import ClientProxy  # noqa: E402

from hospital_registry import build_hospital_registry  # noqa: E402
from monitoring.experiment_logging import ExperimentLogger  # noqa: E402
from runtime.config import load_config  # noqa: E402
from server.strategy import FedMelanomaStrategy, HospitalConfig  # noqa: E402


class FakeProxy(ClientProxy):
    def get_properties(self, ins, timeout, group_id): ...
    def get_parameters(self, ins, timeout, group_id): ...
    def fit(self, ins, timeout, group_id): ...
    def evaluate(self, ins, timeout, group_id): ...
    def reconnect(self, ins, timeout, group_id): ...


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config()
    seed = cfg["experiment"]["seed"] if args.seed is None else args.seed
    fed, fl = cfg["federation"], cfg["fedlesscan"]
    rounds = fed["num_server_rounds"]
    names, records = build_hospital_registry(cfg["data"]["partitions_root"])
    sizes = {i: r["n_samples"] for i, r in enumerate(records)}

    tmp = tempfile.mkdtemp(prefix="dry_run_")
    initial = [np.zeros(3, dtype=np.float32)]
    strategy = FedMelanomaStrategy(
        hospital_configs=[HospitalConfig(**r) for r in records],
        n_select=fed["clients_per_round"], initial_parameters=initial, rng=np.random.default_rng(seed),
        experiment_logger=ExperimentLogger("dry-run", logs_root=os.path.join(tmp, "logs")), run_id="dry-run",
        checkpoint_dir=os.path.join(tmp, "ckpt"), results_dir=os.path.join(tmp, "res"), seed=seed, max_rounds=rounds,
        round_deadline_seconds=fl["round_deadline_seconds"],
        allowed_stragglers=fl["allowed_stragglers"], staleness_tolerance=fl["staleness_tolerance"])
    manager = SimpleClientManager()
    proxies = {i: FakeProxy(f"{i:02d}") for i in sizes}
    for proxy in proxies.values():
        manager.register(proxy)

    parameters = ndarrays_to_parameters(initial)
    plan, counts = [], {k: Counter() for k in ("selected", "on_time", "late", "failed", "aggregated")}
    quiet = io.StringIO()
    for t in range(1, rounds + 1):
        with contextlib.redirect_stdout(quiet):
            strategy.configure_fit(t, parameters, manager)
            selected = strategy._last_round_selected_ids
            results = [(proxies[i], FitRes(status=Status(code=Code.OK, message=""), parameters=ndarrays_to_parameters(initial),
                                           num_examples=sizes[i], metrics={"train_loss": 1.0, "fit_seconds": 1.0})) for i in selected]
            strategy.aggregate_fit(t, results, [])
        p = strategy._round_plan[t]
        plan.append({"round": t, **p})
        for key in counts:
            counts[key].update(p[key] if key != "aggregated" else [c for c, _ in p[key]])

    out = {
        "seed": seed, "deadline_seconds": fl["round_deadline_seconds"],
        "hospitals": [{"client_id": i, "hospital": names[i], "n_samples": sizes[i], "tier": records[i]["tier"]} for i in sizes],
        "round_1_selected": plan[0]["selected"], "rounds": plan,
        "totals": {k: {str(c): v[c] for c in sorted(sizes)} for k, v in counts.items()},
        "late_total": sum(len(p["late"]) for p in plan), "failed_total": sum(len(p["failed"]) for p in plan),
        "on_time_total": sum(len(p["on_time"]) for p in plan), "selected_total": sum(len(p["selected"]) for p in plan),
    }
    path = os.path.join(ROOT, "results", "schedule", "planned_schedule.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in ("selected_total", "on_time_total", "late_total", "failed_total", "round_1_selected")}))
    print(json.dumps(out["totals"]["selected"]))


if __name__ == "__main__":
    main()
