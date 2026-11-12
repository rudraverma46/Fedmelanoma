"""Tests 16-18: assertions on the artefacts of a real two-round GPU run with a restart in between.

tools/run_verification.py runs `verify-seed42` (round 1, exit, new process resumes round 2) and
evaluates its checkpoint before this file runs.
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

RUN = "verify-seed42"


def log(tier):
    path = {"op": f"logs/op/{RUN}.jsonl", "math": f"logs/math/{RUN}.jsonl", "audit": f"logs/audit/{RUN}.jsonl",
            "straggler": f"logs/op/stragglers/{RUN}.jsonl", "clients": f"logs/clients/{RUN}.jsonl", "rounds": f"logs/rounds/{RUN}.jsonl"}[tier]
    return list(read_jsonl(path))


NOVELTY_INITIAL_HASH = "330768fe201714e3f26b9973f70fd32cbec63e2bce8b8569194ef65641876f4a"  # logs/audit/resnet50-seed42.jsonl


def test_16_same_starting_model_as_novelty():
    events = {r["event"]: r for r in log("audit")}
    assert events["run_start"]["initial_parameters_hash"] == NOVELTY_INITIAL_HASH
    assert events["runtime_config"]["initial_parameters_hash"] == NOVELTY_INITIAL_HASH


def test_16_fedavg_weights_in_the_real_run(planned):
    aggregates = {r["round"]: r for r in log("math") if r["event"] == "round_aggregate"}
    sizes = {h["client_id"]: h["n_samples"] for h in planned["hospitals"]}
    returned = [r for r in log("math") if r["event"] == "client_fit_returned"]
    assert all(r["num_examples"] == sizes[r["client_id"]] for r in returned)
    for want in planned["schedule"][:2]:
        got = aggregates[want["round"]]
        weights = {int(k): v for k, v in got["aggregation_c_i"].items()}
        counts = {int(k): v for k, v in got["aggregation_n_i"].items()}
        assert got["dropped_before_training"] == want["dropped"]
        assert sorted(weights) == want["trained"] and counts == {c: sizes[c] for c in want["trained"]}
        assert sum(weights.values()) == pytest.approx(1.0)
        assert all(weights[c] == pytest.approx(counts[c] / sum(counts.values())) for c in weights)


def test_16_real_round_matches_schedule_and_hash_chain(planned):
    summaries = [r for r in log("straggler") if r["event"] == "round_summary"]
    assert [r["round"] for r in summaries] == [1, 2]
    for got, want in zip(summaries, planned["schedule"]):
        assert got["selected_set"] == want["addressed"] and got["straggler_set"] == want["dropped"]
        assert got["on_time_set"] == want["trained"] and got["failed_set"] == [] and got["drained_set"] == []

    aggregates = {r["round"]: r for r in log("math") if r["event"] == "round_aggregate"}
    returned = [r for r in log("math") if r["event"] == "client_fit_returned"]
    assert len(returned) == 12
    assert all(r["client_received_matches_broadcast"] is True for r in returned)
    broadcasts = {r["round"]: r for r in log("math") if r["event"] == "round_broadcast"}
    assert broadcasts[2]["broadcast_matches_previous_round_output"] is True
    assert all(aggregates[t]["global_input_hash"] != aggregates[t]["global_output_hash"] for t in (1, 2))  # no empty rounds

    fits = [r for r in log("clients") if r["event"] == "client_fit"]
    assert sorted(r["client_id"] for r in fits if r["round"] == 1) == planned["schedule"][0]["trained"]
    assert all(np.isfinite(r["train_loss"]) and r["gpu_peak_mb"] < 3900 for r in fits)
    sizes = {h["client_id"]: h["n_samples"] for h in planned["hospitals"]}
    assert all(r["num_examples"] == sizes[r["client_id"]] for r in fits)


def test_17_checkpoint_reload_and_resume(planned):
    path = f"checkpoints/{RUN}/global_round_0002.pth"
    arrays, checkpoint = load_global_checkpoint(path)
    assert hash_ndarrays(arrays) == checkpoint["param_hash"]
    aggregates = {r["round"]: r for r in log("math") if r["event"] == "round_aggregate"}
    assert checkpoint["param_hash"] == aggregates[2]["global_output_hash"]
    assert checkpoint["metrics"].get("roc_auc") is not None

    model = FedMelanomaModel()
    model.load_state_dict(OrderedDict((k, v) for k, v in checkpoint["model_state_dict"].items()), strict=True)
    other = FedMelanomaModel()
    other.load_state_dict(OrderedDict(zip(other.state_dict().keys(), [torch.tensor(a) for a in arrays])), strict=True)
    model.eval(), other.eval()
    image, meta = torch.randn(2, 3, 224, 224), torch.zeros(2, 13)
    with torch.no_grad():
        assert torch.equal(model(image, meta), other(image, meta))

    events = [r["event"] for r in log("audit")]
    assert "resumed" in events
    with open(f"checkpoints/{RUN}/resume_state.pkl", "rb") as fh:
        state = pickle.load(fh)
    assert state["round"] == 2
    assert state["manager"]["buffer"] == {} and state["pending"] == {}  # nothing carries over between rounds


def test_18_evaluation_outputs_exist_and_agree():
    out = f"results/{RUN}/final_round2"
    for name in ("confusion_matrix.png", "roc_curve.png", "pr_curve.png", "threshold_analysis.csv", "classification_report.txt",
                 "final_test_results.txt", "threshold_analysis_requested.csv", "threshold_confusion_requested.csv",
                 "test_predictions.csv", "metrics.json", "results/metrics/overall_metrics.csv",
                 "results/metrics/skin_type_metrics.csv", "results/metrics/image_type_metrics.csv",
                 "results/metrics/confirm_type_metrics.csv", "results/metrics/hospital_metrics.csv",
                 "results/metrics/fairness_summary.txt"):
        assert os.path.getsize(os.path.join(out, name)) > 0, name
    predictions = pd.read_csv(f"{out}/test_predictions.csv")
    assert len(predictions) == 1952 and predictions.y_true.sum() == 109 and predictions.isic_id.is_unique
    from sklearn.metrics import roc_auc_score
    with open(f"{out}/metrics.json", encoding="utf-8") as fh:
        metrics = json.load(fh)
    assert metrics["roc_auc"] == pytest.approx(roc_auc_score(predictions.y_true, predictions.y_prob))
    assert metrics["tp"] + metrics["fn"] == 109
    assert list(pd.read_csv(f"{out}/threshold_analysis_requested.csv").threshold) == [0.1, 0.3, 0.5, 0.7, 0.9]
    assert list(pd.read_csv(f"{out}/threshold_analysis.csv").threshold) == [0.3, 0.4, 0.5, 0.6, 0.7]
