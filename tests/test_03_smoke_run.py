"""Tests 17-22: assertions on the artefacts of the GPU smoke run `smoke-seed42` (tools/run_verification.py).

The smoke run keeps 10% of each hospital's training rows, 1 local epoch and 6 global iterations; it stops after
iteration 2 and a NEW process resumes it. Smoke results verify behaviour only; they are not experimental results.
"""
import json
import os
import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest
import torch

from models.model import FedMelanomaModel
from monitoring.checkpointing import load_global_checkpoint
from monitoring.experiment_logging import read_jsonl
from monitoring.param_hash import hash_ndarrays
from runtime.config import load_config

RUN = "smoke-seed42"
ITERATIONS = 6
NOVELTY_INITIAL_HASH = "330768fe201714e3f26b9973f70fd32cbec63e2bce8b8569194ef65641876f4a"
CFG = load_config()
WAIT = float(CFG["fedstrag"]["wait_time_seconds"])
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = CFG["runtime"]["checkpoint_root"]
CKPT = os.path.join(CKPT if os.path.isabs(CKPT) else os.path.join(ROOT, CKPT), RUN)


def log(name):
    return list(read_jsonl({"iterations": f"logs/iterations/{RUN}.jsonl", "clients": f"logs/clients/{RUN}.jsonl",
                            "updates": f"logs/updates/{RUN}.jsonl", "rounds": f"logs/rounds/{RUN}.jsonl",
                            "audit": f"logs/audit/{RUN}.jsonl"}[name]))


def aggregates():
    return [r for r in log("iterations") if r["event"] == "aggregate"]


def jobs():
    return [r for r in log("clients") if r["event"] in ("client_start", "client_failed")]


def test_17_same_starting_model_and_gpu():
    audit = log("audit")
    start = next(r for r in audit if r["event"] == "run_start")
    config = next(r for r in audit if r["event"] == "runtime_config")
    assert start["initial_parameters_hash"] == config["initial_parameters_hash"] == NOVELTY_INITIAL_HASH
    assert config["gpu"] and config["cuda"] and "resumed" in [r["event"] for r in audit]   # any CUDA GPU


def test_18_iterations_tags_polls_and_weights():
    aggs = aggregates()
    assert [a["iteration"] for a in aggs] == list(range(1, ITERATIONS + 1))       # resume: no gap, no duplicate
    for a in aggs:
        assert a["receivers"] == list(range(10)) and a["tag_broadcast"] == a["latest_tag"] == a["iteration"] - 1
        assert a["poll_times"] == pytest.approx([a["broadcast_time"] + WAIT * (m + 1) for m in range(a["polls"])])
        if a["empty"]:
            assert a["aggregated"] == [] and a["global_changed"] is False
            assert a["model_version_after"] == a["model_version_before"]
            continue
        assert a["global_changed"] is True and a["model_version_after"] == a["model_version_before"] + 1
        coeff = []
        for u in a["aggregated"]:
            assert u["staleness"] == a["latest_tag"] - u["tag"] >= 0
            assert u["raw_weight"] == pytest.approx(1 / (u["staleness"] + 1))
            assert a["broadcast_time"] < u["arrival_time"] <= a["aggregated_at"]
            coeff.append(u["raw_weight"] * u["num_examples_weight"])
        assert [u["normalised_weight"] for u in a["aggregated"]] == pytest.approx([c / sum(coeff) for c in coeff])
        assert a["totals"]["unaccounted_updates"] == 0


def test_19_fifo_order_per_client():
    for c in range(10):
        tags = [j["tag"] for j in jobs() if j["client_id"] == c]
        assert tags == list(range(len(tags)))                                    # oldest first, none skipped
        idx = [j["job_index"] for j in jobs() if j["client_id"] == c]
        assert idx == list(range(len(idx)))


def test_20_failures_and_accounting():
    all_jobs = jobs()
    for j in all_jobs:
        assert j["client_received_matches_sent"] is True and j["start_time"] >= j["broadcast_time"]
        assert 0 < j["gpu_peak_mb"] < torch.cuda.get_device_properties(0).total_memory / 2**20 and np.isfinite(j["train_loss"])
        assert j["duration"] > 0 and j["finish_time"] == pytest.approx(j["start_time"] + j["duration"])
        if j["event"] == "client_failed":                     # trained, occupies the client, then lost
            assert j["failed"] is True and j["status"] == "trained_update_will_be_lost"
        else:
            assert j["failed"] is False
    for c in range(10):                                       # a client's next job starts when the previous one ends
        own = sorted((j for j in all_jobs if j["client_id"] == c), key=lambda j: j["job_index"])
        for prev, nxt in zip(own, own[1:]):
            assert nxt["start_time"] >= prev["finish_time"] - 1e-9
    failed = {(j["client_id"], j["job_index"]) for j in all_jobs if j["failed"]}
    applied = {(u["client_id"], u["job_index"]) for u in log("updates")}
    assert not (failed & applied)
    lost = {(e["client_id"], e["job_index"]) for e in log("iterations") if e["event"] == "update_lost"}
    assert lost <= failed
    with open(f"results/{RUN}/run_summary.json", encoding="utf-8") as fh:
        summary = json.load(fh)
    t = summary["totals"]
    assert t["dispatched"] == t["completed_trainings"] == len(all_jobs) == t["failed_attempts"] + t["updates_generated"]
    assert t["failed_attempts"] == len(failed) and t["applied"] == len(applied)
    assert t["updates_generated"] == t["applied"] + t["in_flight_updates"] and t["unaccounted_updates"] == 0
    assert len(summary["excluded_in_flight_updates"]) == t["in_flight_updates"]
    assert len(summary["failing_jobs_in_progress_at_end"]) == t["failing_jobs_in_progress"]
    assert len(lost) == t["failed_attempts"] - t["failing_jobs_in_progress"]
    assert summary["primary_checkpoint"].startswith("final_global_model.pth") and "iteration 6" in summary["primary_checkpoint"]
    assert summary["global_iterations"] == ITERATIONS and summary["peak_gpu_mb_local_training"] > 0


def test_21_checkpoints_and_resume_state():
    rounds = log("rounds")
    aggs = {a["iteration"]: a for a in aggregates()}
    assert [r["round"] for r in rounds] == list(range(1, ITERATIONS + 1))
    for r in rounds:
        arrays, checkpoint = load_global_checkpoint(os.path.join(CKPT, f"global_round_{r['round']:04d}.pth"))
        assert hash_ndarrays(arrays) == checkpoint["param_hash"] == r["global_hash"] == aggs[r["step"]]["global_hash_after"]
        assert r["evaluated_clients"] == sorted({u["client_id"] for u in aggs[r["step"]]["aggregated"]})
    final, final_ckpt = load_global_checkpoint(os.path.join(CKPT, "final_global_model.pth"))
    assert final_ckpt["param_hash"] == rounds[-1]["global_hash"]
    model = FedMelanomaModel()
    model.load_state_dict(OrderedDict((k, torch.tensor(v)) for k, v in zip(model.state_dict().keys(), final)), strict=True)
    with open(os.path.join(CKPT, "resume_state.pkl"), "rb") as fh:
        state = pickle.load(fh)
    protocol = state["protocol"]
    assert protocol.iter == ITERATIONS and hash_ndarrays(protocol.x) == rounds[-1]["global_hash"]
    snaps = os.path.join(CKPT, "broadcast_snapshots")
    on_disk = sorted(int(n[10:16]) for n in os.listdir(snaps) if n.startswith("broadcast_"))
    assert on_disk == protocol.referenced_tags()


def test_22_final_evaluation_outputs():
    out = f"results/{RUN}/final"
    for name in ("confusion_matrix.png", "roc_curve.png", "threshold_analysis_requested.csv", "test_predictions.csv", "metrics.json"):
        assert os.path.getsize(os.path.join(out, name)) > 0, name
    predictions = pd.read_csv(f"{out}/test_predictions.csv")
    assert len(predictions) == 1952 and predictions.y_true.sum() == 109 and predictions.isic_id.is_unique
    with open(f"{out}/metrics.json", encoding="utf-8") as fh:
        metrics = json.load(fh)
    for key in ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"):
        assert key in metrics
