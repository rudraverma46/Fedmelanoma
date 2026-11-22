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


def test_16_same_starting_model_as_novelty_and_textbook():
    events = {r["event"]: r for r in log("audit")}
    assert events["run_start"]["initial_parameters_hash"] == NOVELTY_INITIAL_HASH
    assert events["runtime_config"]["initial_parameters_hash"] == NOVELTY_INITIAL_HASH


def test_16_selection_and_outcomes_match_the_dry_run_plan(planned):
    """The schedule is deterministic (seeded trace, simulated latency): the real run must equal the plan."""
    selections = {r["round"]: r for r in log("math") if r["event"] == "fedlesscan_selection"}
    jobs = {r["round"]: r for r in log("math") if r["event"] == "fedlesscan_jobs"}
    aggs = {r["round"]: r for r in log("math") if r["event"] == "fedlesscan_aggregation"}
    assert selections[1]["selected"] == planned["round_1_selected"]
    fits = [r for r in log("clients") if r["event"] == "client_fit"]
    for t in (1, 2):
        want = planned["rounds"][t - 1]
        assert selections[t]["selected"] == want["selected"]
        assert sorted(r["client_id"] for r in fits if r["round"] == t) == want["selected"]   # train first, fail after
        assert jobs[t]["on_time"] == want["on_time"] and jobs[t]["late"] == want["late"] and jobs[t]["failed"] == want["failed"]
        assert [[c["client_id"], c["round_id"] + 1] for c in aggs[t]["candidates"]] == want["aggregated"]


def test_16_aggregation_follows_the_official_staleness_formula_and_hash_chain(planned):
    sizes = {h["client_id"]: h["n_samples"] for h in planned["hospitals"]}
    for agg in (r for r in log("math") if r["event"] == "fedlesscan_aggregation"):
        counts = {(c["client_id"], c["round_id"]): c["cardinality"] for c in agg["candidates"]}
        assert all(n == sizes[c] for (c, _), n in counts.items())
        total = sum(counts.values())
        for c in agg["candidates"]:
            assert c["score"] == pytest.approx((c["round_id"] + 1) / (agg["round0"] + 1))   # zero-based
            assert agg["round0"] - c["round_id"] <= agg["tolerance"] == 2
            assert c["effective_weight"] == pytest.approx(c["cardinality"] * c["score"] / total)
    returned = [r for r in log("math") if r["event"] == "client_fit_returned"]
    assert all(r["client_received_matches_broadcast"] is True for r in returned)
    broadcasts = {r["round"]: r for r in log("math") if r["event"] == "round_broadcast"}
    assert broadcasts[2]["broadcast_matches_previous_round_output"] is True
    aggregates = {r["round"]: r for r in log("math") if r["event"] == "round_aggregate"}
    assert all(aggregates[t]["global_input_hash"] != aggregates[t]["global_output_hash"] for t in (1, 2)
               if aggregates[t]["n_aggregated"] > 0)
    fits = [r for r in log("clients") if r["event"] == "client_fit"]
    assert all(np.isfinite(r["train_loss"]) and r["gpu_peak_mb"] < 3900 and r["fit_seconds"] > 0 for r in fits)


def test_16_evaluation_uses_the_selected_clients(planned):
    evaluated = [r for r in log("clients") if r["event"] == "client_evaluate"]
    for t in (1, 2):
        assert sorted(r["client_id"] for r in evaluated if r["round"] == t) == planned["rounds"][t - 1]["selected"]


def test_17_checkpoint_reload_and_resume():
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
    image, meta = torch.randn(2, 3, 224, 224), torch.zeros(2, 9)
    with torch.no_grad():
        assert torch.equal(model(image, meta), other(image, meta))

    assert "resumed" in [r["event"] for r in log("audit")]
    with open(f"checkpoints/{RUN}/resume_state.pkl", "rb") as fh:
        state = pickle.load(fh)
    assert state["round"] == 2
    assert state["fedlesscan"].results == []  # every result was consumed by its round's aggregation
    assert sum(state["fedlesscan"].job_counts.values()) == len([r for r in log("clients") if r["event"] == "client_fit"])


def test_18_evaluation_outputs_exist_and_agree():
    out = f"results/{RUN}/final_round2"
    for name in ("confusion_matrix.png", "roc_curve.png", "pr_curve.png", "threshold_analysis.csv", "classification_report.txt",
                 "final_test_results.txt", "threshold_analysis_requested.csv", "threshold_confusion_requested.csv",
                 "test_predictions.csv", "metrics.json", "results/metrics/overall_metrics.csv",
                 "results/metrics/skin_type_metrics.csv", "results/metrics/image_type_metrics.csv",
                 "results/metrics/hospital_metrics.csv",
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
