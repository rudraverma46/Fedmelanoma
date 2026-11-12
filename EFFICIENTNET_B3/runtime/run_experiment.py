"""Runs the federated experiment with the reference strategy and Flower's round methods.

Run from the experiment folder:
    python -m runtime.run_experiment [--run-id ID] [--rounds N] [--stop-after R] [--resume]

Exit codes: 0 finished; 10 safe stop (low commit or logs/STOP); anything else is a crash.
The body of the round loop is flwr.server.Server.fit's, written out only because
Server.fit cannot start at a later round.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding="utf-8", line_buffering=True)  # vendored prints use non-ASCII glyphs

import numpy as np  # noqa: E402
import torch  # noqa: E402
from flwr.common import Context, RecordSet, parameters_to_ndarrays  # noqa: E402
from flwr.server import Server  # noqa: E402

from hospital_registry import discover_hospitals  # noqa: E402
from models.model import FedMelanomaModel  # noqa: E402
from monitoring.param_hash import hash_ndarrays  # noqa: E402
from runtime.config import load_config  # noqa: E402
from runtime.in_process import assert_identity_mapping, build_federation  # noqa: E402
from runtime.resources import ResourceMonitor, keep_awake  # noqa: E402
from runtime.state import capture_state, restore_state, save_state, state_path, truncate_jsonl_after_round  # noqa: E402
from server.server import server_fn  # noqa: E402

EXIT_SAFE_STOP = 10
STOP_FILE = os.path.join("logs", "STOP")


class ClientFailure(RuntimeError):
    """A genuine client error. The reference would count it as s = inf; this run must stop (D8)."""


def seed_everything(seed: int) -> None:
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


def release_memory_before_aggregation(strategy) -> None:
    """Aggregation is the commit peak: crashes in the full run were MemoryError there.

    Changes no value used by the method. Cached CUDA blocks are returned to the driver, and
    pending weights of clients outside the buffer are dropped: those belong to failed clients
    and can never be consumed, which is also why runtime/state.py does not save them.
    """
    buffered = set(strategy._straggler_manager.get_buffer_state().keys())
    for client_id in list(strategy._pending_client_ndarrays):
        if client_id not in buffered:
            del strategy._pending_client_ndarrays[client_id]
    gc.collect()
    torch.cuda.empty_cache()


def fail_fast(strategy) -> None:
    """Wrap the two aggregate callbacks so client errors stop the run before any state changes."""
    fit, evaluate = strategy.aggregate_fit, strategy.aggregate_evaluate

    def check(failures, stage: str) -> None:
        if failures:
            raise ClientFailure(f"{stage}: {[repr(f) for f in failures]}")

    def guarded_fit(server_round, results, failures):
        check(failures, f"round {server_round} fit")
        release_memory_before_aggregation(strategy)
        return fit(server_round, results, failures)

    def guarded_evaluate(server_round, results, failures):
        check(failures, f"round {server_round} evaluate")
        return evaluate(server_round, results, failures)

    strategy.aggregate_fit, strategy.aggregate_evaluate = guarded_fit, guarded_evaluate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--stop-after", type=int, help="exit cleanly after this round (test of resume)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    run_id = args.run_id or cfg["experiment"]["run_id"]
    seed = cfg["experiment"]["seed"]
    rounds = args.rounds or cfg["federation"]["num_server_rounds"]
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; refusing to train on the CPU.")
    device = torch.device("cuda")
    seed_everything(seed)

    # Arguments exactly as the reference's run_config; server_fn is the vendored one.
    context = Context(run_id=0, node_id=0, node_config={}, state=RecordSet(), run_config={
        "num-server-rounds": rounds, "local-epochs": cfg["federation"]["local_epochs"],
        "clients-per-round": cfg["federation"]["clients_per_round"],
        "partitions-root": cfg["data"]["partitions_root"], "seed": seed, "run-id": run_id,
        "checkpoint-dir": "checkpoints", "results-dir": "results",
    })
    components = server_fn(context)
    strategy = components.strategy
    fail_fast(strategy)
    hospitals = discover_hospitals(cfg["data"]["partitions_root"])
    logger = strategy._logger

    client_log_path = os.path.join("logs", "clients", f"{run_id}.jsonl")
    round_log_path = os.path.join("logs", "rounds", f"{run_id}.jsonl")
    model = FedMelanomaModel().to(device)
    manager = build_federation(cfg, hospitals, model, device, jsonl_writer(client_log_path))
    server = Server(client_manager=manager, strategy=strategy)
    server.set_max_workers(1)  # one GPU, one client at a time, in selection order

    start_round = 1
    path = state_path("checkpoints", run_id)
    if args.resume and os.path.exists(path):
        last = restore_state(path, server, strategy, strategy._rng)
        for log_file in (logger.op_path, logger.math_path, logger.straggler_path, client_log_path, round_log_path):
            truncate_jsonl_after_round(log_file, last)
        logger.log_audit({"event": "resumed", "after_round": last})
        start_round = last + 1
        print(f"[run] resumed after round {last}")
    else:
        if os.path.exists(path):
            raise SystemExit(f"{path} exists; pass --resume or use a new run id.")
        server.parameters = strategy.initialize_parameters(manager)
        logger.log_audit({"event": "runtime_config", "config": cfg, "rounds": rounds,
                          "torch": torch.__version__, "cuda": torch.version.cuda,
                          "gpu": torch.cuda.get_device_name(0),
                          "initial_parameters_hash": hash_ndarrays(parameters_to_ndarrays(server.parameters))})

    monitor = ResourceMonitor(os.path.join("logs", "resources", f"{run_id}.csv"),
                              cfg["runtime"]["monitor_interval_seconds"], cfg["runtime"]["commit_floor_gb"])
    monitor.start()
    keep_awake(True)
    write_round = jsonl_writer(round_log_path)
    try:
        for t in range(start_round, rounds + 1):
            monitor.current_round = t
            started = time.time()
            res_fit = server.fit_round(server_round=t, timeout=None)
            fit_metrics = {}
            if res_fit is not None:
                parameters_prime, fit_metrics, _ = res_fit
                if parameters_prime:
                    server.parameters = parameters_prime
            assert_identity_mapping(strategy, hospitals)
            res_eval = server.evaluate_round(server_round=t, timeout=None)
            eval_loss, eval_metrics = (res_eval[0], res_eval[1]) if res_eval else (None, {})
            save_state(path, capture_state(t, server, strategy, strategy._rng))
            write_round({"round": t, "seconds": time.time() - started, "fit": fit_metrics,
                         "eval_loss": eval_loss, "eval": eval_metrics})
            print(f"[run] round {t}/{rounds} {time.time() - started:.0f} s "
                  f"train_loss={fit_metrics.get('train_loss')} roc_auc={eval_metrics.get('roc_auc')}")
            if t < rounds and (monitor.low_commit.is_set() or os.path.exists(STOP_FILE)):
                reason = "low commit" if monitor.low_commit.is_set() else "logs/STOP"
                logger.log_audit({"event": "safe_stop", "after_round": t, "reason": reason})
                print(f"[run] safe stop after round {t}: {reason}")
                return EXIT_SAFE_STOP
            if args.stop_after == t:
                print(f"[run] stopping after round {t} as requested")
                return 0
        logger.log_run_end({"completed_rounds": rounds, "best": strategy._best_checkpoint_state})
        print("[run] finished")
        return 0
    except BaseException as exc:
        logger.log_exception("round loop", exc)
        traceback.print_exc()
        raise
    finally:
        monitor.halt.set()
        keep_awake(False)


if __name__ == "__main__":
    sys.exit(main())
