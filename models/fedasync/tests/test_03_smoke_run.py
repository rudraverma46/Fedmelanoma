"""Tests 17-21: assertions on the artefacts of the GPU smoke run `smoke-seed42` (tools/run_verification.py).

The smoke run keeps 5% of each hospital's training rows (proportional, so fit times keep the real pattern),
1 local epoch, 18 updates, evaluation every 6; it stops after update 12, and a NEW process resumes it.
These are smoke-test results only: they verify behaviour, they are not experimental results.
"""
import json
import os
import pickle
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd
import pytest
import torch

from models.model import FedMelanomaModel
from monitoring.checkpointing import load_global_checkpoint
from monitoring.experiment_logging import read_jsonl
from monitoring.param_hash import hash_ndarrays
from server.fedasync import make_staleness_function

RUN = "smoke-seed42"
TOTAL, EVERY, CONCURRENCY = 18, 6, 6
HINGE = make_staleness_function("hinge", 10, 4)
NOVELTY_INITIAL_HASH = "330768fe201714e3f26b9973f70fd32cbec63e2bce8b8569194ef65641876f4a"


def log(name):
    return list(read_jsonl({"updates": f"logs/updates/{RUN}.jsonl", "clients": f"logs/clients/{RUN}.jsonl",
                            "rounds": f"logs/rounds/{RUN}.jsonl", "audit": f"logs/audit/{RUN}.jsonl",
                            "math": f"logs/math/{RUN}.jsonl"}[name]))


def test_17_same_starting_model_and_gpu():
    audit = log("audit")
    start = next(r for r in audit if r["event"] == "run_start")
    config = next(r for r in audit if r["event"] == "runtime_config")
    assert start["initial_parameters_hash"] == config["initial_parameters_hash"] == NOVELTY_INITIAL_HASH
    assert "RTX" in config["gpu"]
    assert "resumed" in [r["event"] for r in audit]


def test_18_every_update_applied_once_in_arrival_order_with_correct_math():
    updates = log("updates")
    assert [u["step"] for u in updates] == list(range(1, TOTAL + 1))          # resume left no gap or duplicate
    assert [u["arrival_time"] for u in updates] == sorted(u["arrival_time"] for u in updates)
    for u in updates:
        assert u["version_before"] == u["step"] - 1 and u["version_after"] == u["step"]
        assert u["staleness"] == u["version_before"] - u["tau"] >= 0
        assert u["s_value"] == pytest.approx(HINGE(u["staleness"]))
        assert u["alpha_t"] == pytest.approx(0.6 * HINGE(u["staleness"]))
        assert u["accepted"] is True and u["global_changed"] is True
        # logged right after the arrival is applied, before the freed slot is refilled
        assert len(u["still_in_flight"]) == CONCURRENCY - 1 and u["client_id"] not in u["still_in_flight"]
    assert any(u["staleness"] > 0 for u in updates)
    # the initial fill dispatches 1..CONCURRENCY clients; after every refill exactly CONCURRENCY train, each once
    for i, f in enumerate(log("clients")):
        expected = i + 1 if f["step"] == 0 else CONCURRENCY
        assert len(f["busy_after_dispatch"]) == len(set(f["busy_after_dispatch"])) == expected


def test_19_asynchronous_not_synchronous():
    """The server applied updates while earlier-dispatched clients were still training."""
    updates = log("updates")
    overtaken = sum(1 for a, b in zip(updates, updates[1:]) if b["dispatch_time"] < a["dispatch_time"])
    assert overtaken > 0
    # No synchronous barrier: the 6 first-dispatched clients (all at t = 0) were not applied as a block of 6
    # identical-version updates; versions advance one update at a time.
    assert len({u["version_after"] for u in updates}) == TOTAL


def test_20_dispatches_and_client_side():
    fits = log("clients")
    updates = log("updates")
    assert len(fits) == CONCURRENCY + TOTAL - 1                               # no dispatch after the last update
    for f in fits:
        assert f["tau"] == f["step"]                                         # sent the version current at dispatch
        assert f["client_received_matches_sent"] is True
        assert f["proximal_mu"] == pytest.approx(0.005) and f["proximal_penalty"] >= 0
        assert f["duration_seconds"] > 0 and f["gpu_peak_mb"] < 3900 and np.isfinite(f["train_loss"])
        assert f["arrival_time"] == pytest.approx(f["dispatch_time"] + f["duration_seconds"])
    assert any(f["proximal_penalty"] > 0 for f in fits)
    sizes = {f["client_id"]: f["num_examples"] for f in fits}
    assert all(sizes[c] == n for c, n in {u["client_id"]: u["num_examples"] for u in updates}.items())
    run_end = next(r for r in log("audit") if r["event"] == "run_end")
    assert sum(run_end["accepted_per_client"].values()) == TOTAL and not run_end["rejected_per_client"]
    assert Counter(str(u["client_id"]) for u in updates) == Counter(run_end["received_per_client"])
    assert len(run_end["in_flight_at_stop_not_applied"]) == CONCURRENCY - 1


def test_21_evaluation_checkpoints_and_resume_state():
    rounds = log("rounds")
    updates = log("updates")
    assert [r["round"] for r in rounds] == [1, 2, 3] and [r["step"] for r in rounds] == [6, 12, 18]
    for r in rounds:
        window = [u["client_id"] for u in updates if r["step"] - EVERY < u["step"] <= r["step"]]
        assert r["updates_in_window"] == window and r["evaluated_clients"] == sorted(set(window))
        arrays, checkpoint = load_global_checkpoint(f"checkpoints/{RUN}/global_round_{r['round']:04d}.pth")
        assert hash_ndarrays(arrays) == checkpoint["param_hash"] == r["global_hash"]
        assert r["global_hash"] == updates[r["step"] - 1]["global_hash_after"]
        assert checkpoint["metrics"]["updates_applied"] == r["step"] and 0 <= checkpoint["metrics"]["roc_auc"] <= 1
    final, final_ckpt = load_global_checkpoint(f"checkpoints/{RUN}/final_global_model.pth")
    assert final_ckpt["param_hash"] == rounds[-1]["global_hash"]
    model = FedMelanomaModel()
    model.load_state_dict(OrderedDict((k, torch.tensor(v)) for k, v in zip(model.state_dict().keys(), final)), strict=True)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224), torch.zeros(2, 13))
    assert torch.isfinite(out).all()
    with open(f"checkpoints/{RUN}/resume_state.pkl", "rb") as fh:
        state = pickle.load(fh)
    assert state["applied"] == TOTAL and state["version"] == TOTAL and len(state["in_flight"]) == CONCURRENCY - 1
    assert hash_ndarrays(state["x"]) == rounds[-1]["global_hash"]


def test_22_final_evaluation_outputs():
    out = f"results/{RUN}/final"
    for name in ("confusion_matrix.png", "roc_curve.png", "threshold_analysis_requested.csv", "test_predictions.csv", "metrics.json"):
        assert os.path.getsize(os.path.join(out, name)) > 0, name
    predictions = pd.read_csv(f"{out}/test_predictions.csv")
    assert len(predictions) == 1952 and predictions.y_true.sum() == 109 and predictions.isic_id.is_unique
    assert list(pd.read_csv(f"{out}/threshold_analysis_requested.csv").threshold) == [0.1, 0.3, 0.5, 0.7, 0.9]
    resources = pd.read_csv(f"logs/resources/{RUN}.csv")
    assert len(resources) > 0 and pd.to_numeric(resources.gpu_mem_mb, errors="coerce").max() > 0
