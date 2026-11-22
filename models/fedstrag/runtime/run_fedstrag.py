"""Runs standard FedStrag (paper Algorithm 1 + official code) on the ten hospitals with the ResNet-50 Novelty
model, data and local training, inside the Novelty straggler AND failure environment (sim/environment.py).

Run from the experiment folder:
    python -m runtime.run_fedstrag [--run-id ID] [--resume] [--stop-after-iteration I]
    smoke test: --run-id smoke-... --smoke-fraction 0.10 --global-iterations 6 --local-epochs 1

Exit codes: 0 finished (or stopped as requested); 10 safe stop (low commit or logs/STOP); else a crash.

Per global iteration (server/protocol.py): broadcast to all 10 clients with tag t = iteration - 1; clients work
through their FIFO queues (each job is trained, then the novelty environment decides: failed -> update lost;
delivered -> arrives after the tier LogNormal delay); the server polls every 4 s until >= 1 update has arrived,
then replaces the global model with the W*D-weighted average of the arrived updates, W = 1/(t_g - t_l + 1).
After EVERY iteration: evaluate the global model on the local test partitions of the clients aggregated in that
iteration (the novelty run evaluated each round on that round's participating clients), save the checkpoint,
update the best checkpoint (ROC-AUC), save the resume state, delete broadcast snapshots no queue references.
An empty iteration (no update could arrive) leaves the model unchanged and is not evaluated.
After iteration 50: stop; in-flight updates and unstarted queued broadcasts are reported, not applied.

Logs (logs/<kind>/<run_id>.jsonl, every record has `step` = global iteration):
  iterations : broadcast, poll, update_arrived, empty_iteration and one `aggregate` record per iteration
  clients    : one record per dispatched job (client_start = delivered later, client_failed = update lost)
  updates    : one record per APPLIED update (iteration, poll time, client, broadcast tag, completion time,
               staleness, samples, raw and normalised weight, queue position, model version)
  rounds     : one evaluation record per iteration
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import shutil
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding="utf-8", line_buffering=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from hospital_registry import build_hospital_registry, discover_hospitals  # noqa: E402
from models.model import FedMelanomaModel  # noqa: E402
from monitoring.checkpointing import maybe_update_best_checkpoint, save_global_checkpoint  # noqa: E402
from monitoring.experiment_logging import ExperimentLogger  # noqa: E402
from monitoring.param_hash import hash_ndarrays  # noqa: E402
from runtime.clients import build_clients, timed_fit  # noqa: E402
from runtime.config import load_config  # noqa: E402
from runtime.metrics import weighted_average  # noqa: E402
from runtime.resources import ResourceMonitor, keep_awake  # noqa: E402
from runtime.state import load_state, restore_rngs, rng_states, save_state, state_path, truncate_jsonl_after_step  # noqa: E402
from server.fedstrag import OFFICIAL_COMMIT, OFFICIAL_REPO  # noqa: E402
from server.protocol import FedStragProtocol, SnapshotStore  # noqa: E402
from sim.environment import NoveltyEnvironment  # noqa: E402

EXIT_SAFE_STOP = 10
STOP_FILE = os.path.join("logs", "STOP")
NOVELTY_INITIAL_HASH = "330768fe201714e3f26b9973f70fd32cbec63e2bce8b8569194ef65641876f4a"   # logged by the novelty run
EVAL_KEYS = ("accuracy", "roc_auc", "pr_auc", "precision", "recall", "specificity", "f1", "balanced_accuracy")


def seed_everything(seed: int) -> None:
    """The novelty runtime/run_experiment.py seed_everything, unchanged."""
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


def check_config(cfg: dict) -> None:
    fs, env = cfg["fedstrag"], cfg["environment"]
    expected = {"broadcast_to": "all", "client_queue": "fifo", "wait_time_seconds": 4, "minimum_clients": 1,
                "staleness_weight": "inverse_staleness_plus_one", "sample_weighting": "paper_eq7_W_times_D",
                "empty_iteration": "keep_global_unchanged", "termination": "exclude_pending",
                "arrival_time": "novelty_lognormal", "eval_every_iterations": 1}
    wrong = {k: fs.get(k) for k, v in expected.items() if fs.get(k) != v}
    if wrong or env.get("kind") != "novelty_simulate_response" or env.get("failure") != "novelty_beta_bernoulli" \
            or env.get("failure_timing") != "after_local_training" \
            or env.get("failed_job_duration") != "tier_lognormal_child_stream":
        raise SystemExit(f"only the configured standard FedStrag is implemented; unexpected settings: {wrong or env}")
    init = cfg.get("initialisation") or {}
    if init.get("mode") != "seed" and (not init.get("file") or init.get("param_hash") != NOVELTY_INITIAL_HASH):
        raise SystemExit("configs: initialisation.file and the novelty initial param_hash are required")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-iteration", type=int, help="exit cleanly after this global iteration")
    parser.add_argument("--global-iterations", type=int, help="smoke tests only")
    parser.add_argument("--local-epochs", type=int, help="smoke tests only")
    parser.add_argument("--smoke-fraction", type=float, default=0.0, help="smoke test only: share of training rows kept")
    args = parser.parse_args()

    cfg = load_config()
    check_config(cfg)
    fs = cfg["fedstrag"]
    run_id = args.run_id or cfg["experiment"]["run_id"]
    seed = cfg["experiment"]["seed"]
    iterations = args.global_iterations or fs["global_iterations"]
    local_epochs = args.local_epochs or cfg["federation"]["local_epochs"]
    smoke = run_id.startswith("smoke")
    if (args.smoke_fraction or args.global_iterations or args.local_epochs) and not smoke:
        raise SystemExit("--smoke-fraction / --global-iterations / --local-epochs are only allowed for run ids starting with 'smoke'")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; refusing to train on the CPU.")
    device = torch.device("cuda")
    ckpt_root = cfg["runtime"]["checkpoint_root"]
    ckpt_root = ckpt_root if os.path.isabs(ckpt_root) else os.path.join(ROOT, ckpt_root)

    # Same construction order as the earlier FedStrag / textbook runs: seed, initial global model, training model.
    seed_everything(seed)
    initial_model = FedMelanomaModel()
    initial = [v.detach().cpu().numpy().copy() for v in initial_model.state_dict().values()]
    param_keys = list(initial_model.state_dict().keys())
    del initial_model
    # The NOVELTY run's exact initial model (user decision 2026-11-20): newer PyTorch builds initialise the new layers
    # differently from torch 2.2.1 even with seed 42, so the starting weights are loaded from a file and checked by
    # hash. The model above is still constructed first so the RNG stream is consumed exactly as before.
    if cfg["initialisation"].get("mode") != "seed":   # 2026-11-21: seed mode keeps the seed-42 model built above
        with open(os.path.join(ROOT, cfg["initialisation"]["file"]), "rb") as fh:
            saved = pickle.load(fh)
        if saved["keys"] != param_keys or [a.shape for a in saved["arrays"]] != [a.shape for a in initial] \
                or hash_ndarrays(saved["arrays"]) != cfg["initialisation"]["param_hash"]:
            raise SystemExit("initial model file does not match the novelty initial model (keys, shapes or hash)")
        if hash_ndarrays(initial) != cfg["initialisation"]["param_hash"]:
            print(f"[run] this PyTorch builds initial hash {hash_ndarrays(initial)[:12]}; using the novelty initial model "
                  f"{cfg['initialisation']['param_hash'][:12]} from {cfg['initialisation']['file']}")
        initial = [a.copy() for a in saved["arrays"]]
        del saved
    hospitals = discover_hospitals(cfg["data"]["partitions_root"])
    model = FedMelanomaModel().to(device)
    clients = build_clients(cfg, hospitals, model, device, proximal_mu=0.0, smoke_fraction=args.smoke_fraction)

    logger = ExperimentLogger(run_id)
    paths = {k: os.path.join("logs", k, f"{run_id}.jsonl") for k in ("iterations", "clients", "updates", "rounds")}
    write = {k: jsonl_writer(p) for k, p in paths.items()}

    def train_fn(client_id, arrays):
        new_arrays, n, metrics, seconds, peak = timed_fit(clients[client_id], arrays, local_epochs)
        metrics = {**metrics, "gpu_peak_mb": peak,
                   "client_received_matches_sent": metrics.get("client_received_param_hash") == hash_ndarrays(arrays)}
        return new_arrays, n, metrics, seconds

    rng = np.random.default_rng(seed)   # not used by the algorithm; kept so the RNG state set is the same as before
    store = SnapshotStore(os.path.join(ckpt_root, run_id, "broadcast_snapshots"))
    _, records, *_ = build_hospital_registry(cfg["data"]["partitions_root"])
    env = NoveltyEnvironment(cfg["environment"]["trace_seed"], records)
    protocol = FedStragProtocol(initial, len(hospitals), env, iterations, store, train_fn,
                                fs["wait_time_seconds"], fs["minimum_clients"])
    run = {"best_state": {"best_value": float("-inf")}, "wall_seconds": 0.0, "peak_gpu_mb": 0.0}

    path = state_path(ckpt_root, run_id)
    if args.resume and os.path.exists(path):
        state = load_state(path)
        protocol, run = state["protocol"], state["run"]
        protocol.store, protocol.train_fn, protocol.env = store, train_fn, env
        restore_rngs(state["rngs"], rng)
        missing = [t for t in protocol.referenced_tags() if not os.path.exists(store.path(t))]
        if missing:
            raise SystemExit(f"broadcast snapshots missing for tags {missing}; cannot resume")
        for p in list(paths.values()) + [logger.op_path, logger.math_path]:
            truncate_jsonl_after_step(p, protocol.iter)
        logger.log_audit({"event": "resumed", "after_iteration": protocol.iter, "step": protocol.iter})
        print(f"[run] resumed after iteration {protocol.iter}")
    else:
        if os.path.exists(path):
            raise SystemExit(f"{path} exists; pass --resume or use a new run id.")
        logger.log_run_start({
            "strategy": "FedStrag (standard)", "official_repo": OFFICIAL_REPO, "official_commit": OFFICIAL_COMMIT,
            "step": 0, "fedstrag": fs, "environment": cfg["environment"], "global_iterations": iterations,
            "local_epochs": local_epochs, "smoke_fraction": args.smoke_fraction,
            "initial_parameters_hash": hash_ndarrays(initial), "hospitals": hospitals,
            "tiers": env.tiers(), "seed": seed, "checkpoint_root": ckpt_root})
        logger.log_audit({"event": "runtime_config", "step": 0, "config": cfg, "torch": torch.__version__,
                          "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                          "initial_parameters_hash": hash_ndarrays(initial)})
        os.makedirs(os.path.join("results", run_id), exist_ok=True)
        with open(os.path.join("results", run_id, "run_config.json"), "w", encoding="utf-8") as fh:
            json.dump({"run_id": run_id, "config": cfg, "global_iterations": iterations, "local_epochs": local_epochs,
                       "smoke_fraction": args.smoke_fraction, "hospitals": hospitals, "tiers": env.tiers(),
                       "official_repo": OFFICIAL_REPO, "official_commit": OFFICIAL_COMMIT}, fh, indent=2, default=str)

    monitor = ResourceMonitor(os.path.join("logs", "resources", f"{run_id}.csv"),
                              cfg["runtime"]["monitor_interval_seconds"], cfg["runtime"]["commit_floor_gb"])
    monitor.start()
    keep_awake(True)
    try:
        while not protocol.done():
            started = time.time()
            before = hash_ndarrays(protocol.x)
            info = protocol.run_iteration()
            after = hash_ndarrays(protocol.x)
            step = protocol.iter
            for event in protocol.events:
                if event["event"] in ("client_start", "client_failed"):
                    write["clients"]({"step": step, "client_id_hospital": hospitals[event["client_id"]], **event})
                    run["peak_gpu_mb"] = max(run["peak_gpu_mb"], float(event.get("gpu_peak_mb", 0.0)))
                else:
                    write["iterations"]({"step": step, **event})
            protocol.events = []
            for a in info["aggregated"]:
                write["updates"]({"step": step, "iteration": step, "poll_time": info["aggregated_at"],
                                  "client_id_hospital": hospitals[a["client_id"]],
                                  "model_version_after": info["model_version_after"], **a})
            write["iterations"]({"step": step, "event": "aggregate", **info, "global_hash_before": before,
                                 "global_hash_after": after, "global_changed": before != after,
                                 "snapshots_on_disk": len(store.tags())})
            monitor.current_round = step

            evaluated = sorted({a["client_id"] for a in info["aggregated"]})
            metrics, loss, became_best = {}, None, False
            if evaluated:
                results = [clients[c].evaluate(protocol.x, {}) for c in evaluated]
                n_total = sum(n for _, n, _ in results)
                loss = sum(n * l for l, n, _ in results) / n_total
                metrics = weighted_average([(n, m) for _, n, m in results])
            ckpt_metrics = {**metrics, "iteration": step, "virtual_time": protocol.clock, **protocol.totals()}
            save_global_checkpoint(step, protocol.x, run_id, ckpt_metrics, param_keys, ckpt_root)
            if evaluated:
                became_best = maybe_update_best_checkpoint(step, protocol.x, run_id, ckpt_metrics, run["best_state"],
                                                           param_keys, ckpt_root)
            logger.log_math({"event": "round_evaluate", "round": step, "step": step, "eval_loss": loss,
                             **{f"eval_{m}": metrics.get(m) for m in EVAL_KEYS},
                             "evaluated_clients": evaluated, "became_best": became_best, "empty_iteration": info["empty"]})
            run["wall_seconds"] += time.time() - started
            write["rounds"]({"round": step, "step": step, "seconds": time.time() - started,
                             "virtual_time": protocol.clock, "evaluated_clients": evaluated, "eval_loss": loss,
                             "eval": metrics, "became_best": became_best, "empty_iteration": info["empty"],
                             **protocol.totals(), "global_hash": after})
            if protocol.done():
                shutil.copyfile(os.path.join(ckpt_root, run_id, f"global_round_{step:04d}.pth"),
                                os.path.join(ckpt_root, run_id, "final_global_model.pth"))
            save_state(path, {"protocol": protocol, "run": run, "rngs": rng_states(rng)})
            removed = store.collect(protocol.referenced_tags())
            t = protocol.totals()
            print(f"[run] iteration {step}/{iterations}: applied {len(info['aggregated'])} (total {t['applied']}), "
                  f"failed {t['failed_attempts']}, in flight {t['in_flight_updates']}, queued {t['queued_broadcasts_not_started']}, "
                  f"virtual {protocol.clock:.0f} s, wall {time.time() - started:.0f} s, snapshots {len(store.tags())} "
                  f"(-{len(removed)}), roc_auc={metrics.get('roc_auc')}")
            if protocol.done():
                break
            if monitor.low_commit.is_set() or os.path.exists(STOP_FILE):
                reason = "low commit" if monitor.low_commit.is_set() else "logs/STOP"
                logger.log_audit({"event": "safe_stop", "after_iteration": step, "reason": reason, "step": step})
                print(f"[run] safe stop after iteration {step}: {reason}")
                return EXIT_SAFE_STOP
            if args.stop_after_iteration is not None and step >= args.stop_after_iteration:
                print(f"[run] stopping after iteration {step} as requested")
                return 0

        # ---- termination (fedstrag.termination = exclude_pending) -------------------------------------------------
        t = protocol.totals()
        summary = {
            "run_id": run_id, "global_iterations": protocol.iter, "model_version": protocol.model_version,
            "empty_iterations": protocol.empty_iterations, "virtual_time_seconds": protocol.clock,
            "training_wall_seconds": run["wall_seconds"], "peak_gpu_mb_local_training": run["peak_gpu_mb"],
            "totals": t, "per_client": {n: {hospitals[c]: v for c, v in protocol.counts[n].items()} for n in protocol.COUNTERS},
            "tiers": dict(zip(hospitals, env.tiers())),
            "termination_policy": "after the last iteration nothing more is broadcast or aggregated; pending updates are excluded",
            "excluded_in_flight_updates": [{"client_id": c, "hospital": hospitals[c], "tag": j.tag, "job_index": j.job_index,
                                            "would_arrive_at": j.finish_time} for c, j in sorted(protocol.busy.items())
                                           if not j.failed],
            "failing_jobs_in_progress_at_end": [{"client_id": c, "hospital": hospitals[c], "tag": j.tag,
                                                 "job_index": j.job_index, "would_fail_at": j.finish_time}
                                                for c, j in sorted(protocol.busy.items()) if j.failed],
            "primary_checkpoint": f"final_global_model.pth (= global_round_{protocol.iter:04d}.pth, iteration {protocol.iter})",
            "checkpoint_note": ("best_global_model.pth is chosen by ROC-AUC on local TEST partitions; it is kept only as a "
                                "secondary, test-selected diagnostic and is never the reported result"),
            "excluded_queued_broadcasts_not_started": {hospitals[c]: [tag for tag, _ in q] for c, q in protocol.queues.items() if q},
            "best": run["best_state"],
        }
        with open(os.path.join("results", run_id, "run_summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)
        logger.log_run_end({"step": protocol.iter, **summary})
        print(f"[run] finished: {t}")
        return 0
    except BaseException as exc:
        logger.log_exception("iteration loop", exc)
        traceback.print_exc()
        raise
    finally:
        monitor.halt.set()
        keep_awake(False)


if __name__ == "__main__":
    sys.exit(main())
