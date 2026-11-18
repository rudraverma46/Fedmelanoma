"""Runs FedAsync on the ten hospitals with the textbook FedAvg model, data and local training, inside the
FedMelanoma-Novelty tier delays (sim/environment.py: per-tier LogNormal latency; FedAsync as published has no failure).

Run from the experiment folder:
    python -m runtime.run_fedasync [--run-id ID] [--resume] [--stop-after-update S]
    smoke test: --run-id smoke-... --smoke-fraction 0.05 --total-dispatches 12 --local-epochs 1

Exit codes: 0 finished (or stopped as requested); 10 safe stop (low commit or logs/STOP); else a crash.

Order of operations for every update (server/fedasync.py, server/scheduler.py):
    1. the earliest in-flight update arrives (virtual clock moves to its arrival time)
    2. the updater applies it at once: x_v+1 = (1 - alpha s(v - tau)) x_v + alpha s(v - tau) x_new
    3. the freed slot triggers a random idle client with (x_v+1, v+1). Its job-th simulator draw decides:
         the client trains (completed local training) and its update arrives `latency` virtual seconds later
         (LogNormal draw of its tier). There is NO failure: FedAsync as published (arXiv:1903.03934) has no
         failure or lost-update mechanism, so none is simulated (user decision 2026-11-14).
       The run ends when the dispatch budget is spent and every in-flight update has arrived.
       Counters kept apart: dispatched, pre_training_failures, completed_trainings, post_training_failures,
       rejected_or_expired, applied.
    4. every `eval_every_updates` applied updates: evaluate x on the local test partitions of the clients
       whose updates were applied since the last evaluation (the textbook FedAvg procedure: the clients
       that contributed), save the checkpoint, update the best checkpoint (ROC-AUC), save resume state.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
import traceback
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding="utf-8", line_buffering=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from hospital_registry import discover_hospitals  # noqa: E402
from models.model import FedMelanomaModel  # noqa: E402
from monitoring.checkpointing import maybe_update_best_checkpoint, save_global_checkpoint  # noqa: E402
from monitoring.experiment_logging import ExperimentLogger  # noqa: E402
from monitoring.param_hash import hash_ndarrays  # noqa: E402
from runtime.clients import build_clients, timed_fit  # noqa: E402
from runtime.config import load_config  # noqa: E402
from runtime.metrics import weighted_average  # noqa: E402
from runtime.resources import ResourceMonitor, keep_awake  # noqa: E402
from runtime.state import load_state, restore_rngs, rng_states, save_state, state_path, truncate_jsonl_after_step  # noqa: E402
from server.fedasync import PAPER, FedAsyncServer, make_staleness_function  # noqa: E402
from hospital_registry import build_hospital_registry  # noqa: E402
from sim.environment import TraceEnvironment  # noqa: E402
from server.scheduler import VirtualClockScheduler  # noqa: E402

EXIT_SAFE_STOP = 10
STOP_FILE = os.path.join("logs", "STOP")


def seed_everything(seed: int) -> None:
    """resnet50_fedavg_textbook/runtime/run_experiment.py seed_everything, unchanged."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def jsonl_writer(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def write(record: dict) -> None:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"logged_at": time.time(), **record}, default=str) + "\n")
    return write


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-update", type=int, help="exit cleanly after the evaluation at this update count")
    parser.add_argument("--total-dispatches", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--local-epochs", type=int)
    parser.add_argument("--smoke-fraction", type=float, default=0.0, help="smoke test only: share of training rows kept")
    args = parser.parse_args()

    cfg = load_config()
    fa = cfg["fedasync"]
    run_id = args.run_id or cfg["experiment"]["run_id"]
    seed = cfg["experiment"]["seed"]
    total = args.total_dispatches or fa["total_dispatches"]
    eval_every = args.eval_every or fa["eval_every_updates"]
    local_epochs = args.local_epochs or cfg["federation"]["local_epochs"]
    if args.smoke_fraction and not run_id.startswith("smoke"):
        raise SystemExit("--smoke-fraction is only allowed for run ids starting with 'smoke'")
    if fa["reject_stale"]:
        raise SystemExit("reject_stale is not part of FedAsync; this implementation does not support it")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; refusing to train on the CPU.")
    device = torch.device("cuda")

    # Same construction order as textbook FedAvg: seed, initial global model, then the training model.
    seed_everything(seed)
    initial_model = FedMelanomaModel()
    initial = [v.detach().cpu().numpy().copy() for v in initial_model.state_dict().values()]
    param_keys = list(initial_model.state_dict().keys())
    del initial_model
    hospitals = discover_hospitals(cfg["data"]["partitions_root"])
    model = FedMelanomaModel().to(device)
    clients = build_clients(cfg, hospitals, model, device, proximal_mu=fa["rho"], smoke_fraction=args.smoke_fraction)

    logger = ExperimentLogger(run_id)
    paths = {"updates": os.path.join("logs", "updates", f"{run_id}.jsonl"),
             "clients": os.path.join("logs", "clients", f"{run_id}.jsonl"),
             "rounds": os.path.join("logs", "rounds", f"{run_id}.jsonl")}
    write = {k: jsonl_writer(p) for k, p in paths.items()}

    rng = np.random.default_rng(seed)
    staleness_fn = make_staleness_function(fa["staleness_function"], fa["staleness_a"], fa["staleness_b"])
    server = FedAsyncServer(initial, fa["alpha"], staleness_fn)
    scheduler = VirtualClockScheduler(range(len(hospitals)), fa["concurrency"], rng)
    _, records, *_ = build_hospital_registry(cfg["data"]["partitions_root"])
    env = TraceEnvironment(cfg["environment"]["trace_seed"], records)
    applied, window, best_state = 0, [], {"best_value": float("-inf")}
    names = ("dispatched", "pre_training_failures", "completed_trainings", "post_training_failures",
             "rejected_or_expired", "applied")
    counts = {n: Counter() for n in names}
    job_index = Counter()   # per client: how many times it has been dispatched (indexes its trace)

    def totals() -> dict:
        return {n: int(sum(counts[n].values())) for n in names}

    def dispatch() -> None:
        client_id = scheduler.pick_idle()
        k = job_index[client_id]
        job_index[client_id] += 1
        counts["dispatched"][client_id] += 1
        x_tau, tau = server.snapshot()
        sent_hash = hash_ndarrays(x_tau)
        arrays, n, metrics, seconds, peak = timed_fit(clients[client_id], x_tau, local_epochs)   # always trains first
        del x_tau
        counts["completed_trainings"][client_id] += 1
        latency = env.latency(client_id, k)   # paper-literal: no failure draw, only the tier's LogNormal delay
        base = {"step": applied, "client_id": client_id, "hospital": hospitals[client_id],
                "tier": env.stubs[client_id].tier, "job_index": k, "tau": tau, "measured_fit_seconds": seconds,
                "num_examples": n, "train_loss": metrics.get("train_loss"), "gpu_peak_mb": peak}
        job = scheduler.submit(client_id, tau, latency, arrays, n, metrics)
        write["clients"]({
            "event": "client_fit", **base,
            "dispatch_time": job.dispatch_time, "duration_seconds": latency, "arrival_time": job.arrival_time,
            "proximal_mu": metrics.get("proximal_mu"),
            "proximal_penalty": metrics.get("proximal_penalty"), "local_drift_l2": metrics.get("local_drift_l2"),
            "client_received_matches_sent": metrics.get("client_received_param_hash") == sent_hash,
            "busy_after_dispatch": scheduler.busy(), **totals(),
        })

    def refill() -> None:
        while len(scheduler.in_flight) < fa["concurrency"] and int(sum(counts["dispatched"].values())) < total:
            dispatch()

    path = state_path("checkpoints", run_id)
    if args.resume and os.path.exists(path):
        state = load_state(path)
        server.x, server.version = state["x"], state["version"]
        scheduler.idle, scheduler.clock, scheduler.seq, scheduler.in_flight = (
            state["idle"], state["clock"], state["seq"], state["in_flight"])
        applied, best_state, counts, job_index = state["applied"], state["best_state"], state["counts"], state["job_index"]
        restore_rngs(state["rngs"], rng)
        for p in list(paths.values()) + [logger.op_path, logger.math_path]:
            truncate_jsonl_after_step(p, applied)
        logger.log_audit({"event": "resumed", "after_update": applied, "step": applied})
        print(f"[run] resumed after update {applied}")
    else:
        if os.path.exists(path):
            raise SystemExit(f"{path} exists; pass --resume or use a new run id.")
        logger.log_run_start({
            "strategy": "FedAsync", "paper": PAPER, "step": 0, "fedasync": fa, "total_dispatches": total, "environment": cfg["environment"],
            "eval_every_updates": eval_every, "local_epochs": local_epochs, "smoke_fraction": args.smoke_fraction,
            "initial_parameters_hash": hash_ndarrays(initial), "hospitals": hospitals, "seed": seed,
        })
        logger.log_audit({"event": "runtime_config", "step": 0, "config": cfg, "torch": torch.__version__,
                          "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                          "initial_parameters_hash": hash_ndarrays(initial)})
        refill()

    monitor = ResourceMonitor(os.path.join("logs", "resources", f"{run_id}.csv"),
                              cfg["runtime"]["monitor_interval_seconds"], cfg["runtime"]["commit_floor_gb"])
    monitor.start()
    keep_awake(True)
    window_start = time.time()
    try:
        while scheduler.in_flight:
            job = scheduler.next_arrival()
            before = hash_ndarrays(server.x)
            record = server.apply(job.client_id, job.arrays, job.tau, job.arrival_time, job.dispatch_time)
            after = hash_ndarrays(server.x)
            del job.arrays
            applied += 1
            counts["applied" if record.accepted else "rejected_or_expired"][job.client_id] += 1
            window.append(job.client_id)
            write["updates"]({
                "event": "update_applied", "step": applied, **record.__dict__, "duration_seconds": job.duration,
                "num_examples": job.num_examples, "train_loss": job.metrics.get("train_loss"),
                "global_hash_before": before, "global_hash_after": after, "global_changed": before != after,
                "still_in_flight": scheduler.busy(), **totals(),
            })
            monitor.current_round = applied
            refill()
            finished = not scheduler.in_flight
            if applied % eval_every == 0 or finished:
                k = (applied + eval_every - 1) // eval_every
                evaluated = sorted(set(window))
                results = [clients[c].evaluate(server.x, {}) for c in evaluated]
                n_total = sum(n for _, n, _ in results)
                loss = sum(n * l for l, n, _ in results) / n_total
                metrics = weighted_average([(n, m) for _, n, m in results])
                ckpt_metrics = {**metrics, "updates_applied": applied, "server_version": server.version,
                                "virtual_time": scheduler.clock, **totals()}
                save_global_checkpoint(k, server.x, run_id, ckpt_metrics, param_keys, "checkpoints")
                became_best = maybe_update_best_checkpoint(k, server.x, run_id, ckpt_metrics, best_state, param_keys, "checkpoints")
                if finished:
                    shutil.copyfile(os.path.join("checkpoints", run_id, f"global_round_{k:04d}.pth"),
                                    os.path.join("checkpoints", run_id, "final_global_model.pth"))
                logger.log_math({"event": "round_evaluate", "round": k, "step": applied, "eval_loss": loss,
                                 **{f"eval_{m}": metrics.get(m) for m in ("accuracy", "roc_auc", "pr_auc", "recall", "specificity", "f1", "balanced_accuracy")},
                                 "evaluated_clients": evaluated, "became_best": became_best})
                write["rounds"]({"round": k, "step": applied, "seconds": time.time() - window_start,
                                 "virtual_time": scheduler.clock, "server_version": server.version,
                                 "updates_in_window": list(window), "evaluated_clients": evaluated,
                                 "eval_loss": loss, "eval": metrics, "became_best": became_best, **totals(),
                                 "global_hash": hash_ndarrays(server.x)})
                save_state(path, {"x": server.x, "version": server.version, "idle": scheduler.idle,
                                  "clock": scheduler.clock, "seq": scheduler.seq, "in_flight": scheduler.in_flight,
                                  "applied": applied, "best_state": best_state, "counts": counts, "job_index": job_index, "rngs": rng_states(rng)})
                print(f"[run] eval {k}: {applied} applied / {totals()['dispatched']} of {total} dispatched, v={server.version}, virtual {scheduler.clock:.0f} s, "
                      f"wall {time.time() - window_start:.0f} s, roc_auc={metrics.get('roc_auc')}")
                window, window_start = [], time.time()
                if not finished and (monitor.low_commit.is_set() or os.path.exists(STOP_FILE)):
                    reason = "low commit" if monitor.low_commit.is_set() else "logs/STOP"
                    logger.log_audit({"event": "safe_stop", "after_update": applied, "reason": reason, "step": applied})
                    print(f"[run] safe stop after update {applied}: {reason}")
                    return EXIT_SAFE_STOP
                if args.stop_after_update == applied:
                    print(f"[run] stopping after update {applied} as requested")
                    return 0
        logger.log_run_end({"step": applied, "updates_applied": applied, "server_version": server.version,
                            "virtual_time": scheduler.clock, "totals": totals(),
                            "per_client": {n: dict(counts[n]) for n in names}, "best": best_state})
        print("[run] finished")
        return 0
    except BaseException as exc:
        logger.log_exception("update loop", exc)
        traceback.print_exc()
        raise
    finally:
        monitor.halt.set()
        keep_awake(False)


if __name__ == "__main__":
    sys.exit(main())
