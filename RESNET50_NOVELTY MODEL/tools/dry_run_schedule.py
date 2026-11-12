"""Compute the participation schedule without any training.

Selection and the straggler simulation read only the seed and the hospital registry: no
training output feeds back into them. So the whole 50-round schedule can be replayed on the
CPU in a second, using the vendored modules in the same call order as
FedMelanomaStrategy.configure_fit / aggregate_fit (one shared RNG).

Writes results/schedule/planned_schedule.json. The verification suite and the final report
compare the real run against this file.

Usage: python tools/dry_run_schedule.py [--seed 42] [--monte-carlo 2000]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
from typing import Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hospital_registry import build_hospital_registry  # noqa: E402
from runtime.config import load_config  # noqa: E402
from server.aggregation import compute_aggregation_weights  # noqa: E402
from server.selector import Client, Selector  # noqa: E402
from server.straggler import StragglerManager  # noqa: E402
from server.strategy import HospitalConfig, compute_normalized_diversity_priors  # noqa: E402


def build_clients(partitions_root: str):
    names, records, g_fitz, g_mod, g_conf = build_hospital_registry(partitions_root)
    configs = [HospitalConfig(**r) for r in records]
    priors = compute_normalized_diversity_priors(configs, g_fitz, g_mod, g_conf)
    clients = {
        i: Client(client_id=i, n_samples=hc.n_samples, D=priors[i], tier=hc.tier,
                  latency_mu=hc.latency_mu, latency_sigma=hc.latency_sigma)
        for i, hc in enumerate(configs)
    }
    return names, clients


def simulate(clients: Dict[int, Client], seed: int, rounds: int, n_select: int, trace_round: int = 0) -> List[dict]:
    rng = np.random.default_rng(seed)
    selector = Selector()
    manager = StragglerManager(rng=rng)
    schedule: List[dict] = []
    for t in range(1, rounds + 1):
        buffered = set(manager.get_buffer_state().keys())
        candidates = [c for c in clients.values() if c.client_id not in buffered]
        trace = {}
        if t == trace_round:
            probs = selector.boltzmann_probs(candidates)
            trace = {
                "buffered_at_start": sorted(buffered),
                "history_before": list(manager._latency_history),
                "before": {c.client_id: {"R": c.R, "N_eff": c.N_eff, "score": selector.compute_score(c),
                                         "p_first_draw": float(probs[i])} for i, c in enumerate(candidates)},
            }
        selected = selector.boltzmann_selection(
            clients=candidates, n_select=min(n_select, len(candidates)), rng=rng)
        selected_ids = [c.client_id for c in selected]
        window_used = manager.current_W_t
        responses = {cid: manager.simulate_response(client=clients[cid], current_round=t) for cid in selected_ids}
        on_time, stragglers = manager.manage_window(
            clients=[clients[cid] for cid in responses], responses=responses,
            current_round=t, n_selected=len(selected_ids))
        weights = compute_aggregation_weights(on_time=on_time, client_map=clients)
        for cid in selected_ids:
            selector.update_client_state(client=clients[cid], staleness=responses[cid].staleness,
                                         round_idx=t, was_selected=True)
        if trace:
            trace["history_after"] = list(manager._latency_history)
            trace["after"] = {cid: {"R": clients[cid].R, "N_eff": clients[cid].N_eff} for cid in selected_ids}
            trace["staleness"] = {cid: (None if math.isinf(r.staleness) else r.staleness) for cid, r in responses.items()}
        schedule.append({
            "round": t,
            "trace": trace,
            "selected": selected_ids,
            "window_used_for_classification": window_used,
            "window_after_round": manager.current_W_t,
            "latency": {cid: (None if math.isinf(r.response_time) else r.response_time) for cid, r in responses.items()},
            "failed": [cid for cid, r in responses.items() if r.is_failed],
            "new_stragglers": [r.client_id for r in stragglers],
            "aggregated": sorted(weights.keys()),
            "drained": [r.client_id for r in on_time if r.client_id not in responses],
            "weights": {int(k): v for k, v in weights.items()},
            "quorum_threshold": max(1, math.ceil(0.70 * len(selected_ids))),
            "quorum_met": manager.quorum_met_history[-1],
        })
    return schedule


def summarize(schedule: List[dict], clients: Dict[int, Client]) -> dict:
    return {
        "rounds": len(schedule),
        "empty_rounds": sum(1 for r in schedule if not r["aggregated"]),
        "quorum_met_rounds": sum(1 for r in schedule if r["quorum_met"]),
        "mean_selected": float(np.mean([len(r["selected"]) for r in schedule])),
        "mean_aggregated": float(np.mean([len(r["aggregated"]) for r in schedule])),
        "mean_failed": float(np.mean([len(r["failed"]) for r in schedule])),
        "mean_new_stragglers": float(np.mean([len(r["new_stragglers"]) for r in schedule])),
        "mean_window": float(np.mean([r["window_after_round"] for r in schedule])),
        "train_images_total": int(sum(clients[c].n_samples for r in schedule for c in r["selected"]) * 5),
        "selected_count": {c: sum(c in r["selected"] for r in schedule) for c in clients},
        "aggregated_count": {c: sum(c in r["aggregated"] for r in schedule) for c in clients},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--monte-carlo", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config()
    seed = cfg["experiment"]["seed"] if args.seed is None else args.seed
    rounds, n_select = cfg["federation"]["num_server_rounds"], cfg["federation"]["clients_per_round"]
    root = cfg["data"]["partitions_root"]

    with contextlib.redirect_stdout(io.StringIO()):
        names, clients = build_clients(root)
        hospitals = [{"client_id": i, "hospital": names[i], "tier": c.tier, "n_samples": c.n_samples, "D": c.D}
                     for i, c in clients.items()]
        schedule = simulate(clients, seed, rounds, n_select, trace_round=5)
    out = {"seed": seed, "hospitals": hospitals, "summary": summarize(schedule, clients), "schedule": schedule}

    if args.monte_carlo:
        rows = []
        with contextlib.redirect_stdout(io.StringIO()):
            for s in range(args.monte_carlo):
                _, fresh = build_clients(root) if s == 0 else (None, None)
                base = fresh if s == 0 else base
                clones = {i: Client(client_id=i, n_samples=c.n_samples, D=c.D, tier=c.tier,
                                    latency_mu=c.latency_mu, latency_sigma=c.latency_sigma)
                          for i, c in base.items()}
                rows.append(summarize(simulate(clones, 1000 + s, rounds, n_select), clones))
        out["monte_carlo"] = {
            "runs": args.monte_carlo,
            **{k: float(np.mean([r[k] for r in rows]))
               for k in ("empty_rounds", "quorum_met_rounds", "mean_selected", "mean_aggregated",
                         "mean_failed", "mean_new_stragglers", "mean_window", "train_images_total")},
            "aggregated_count": {c: float(np.mean([r["aggregated_count"][c] for r in rows])) for c in clients},
            "selected_count": {c: float(np.mean([r["selected_count"][c] for r in rows])) for c in clients},
        }

    path = os.path.join(ROOT, "results", "schedule", "planned_schedule.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "schedule"}, indent=1))


if __name__ == "__main__":
    main()
