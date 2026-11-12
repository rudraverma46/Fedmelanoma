"""Test P20: FedProx assertions on the artefacts of the real two-round GPU run `verify-seed42`.

test_03_real_round.py, unchanged, already checks the starting model, weights, schedule, hash chain,
reload and resume on the same run.
"""
import json
import os

import numpy as np
import pytest

from monitoring.experiment_logging import read_jsonl

RUN = "verify-seed42"
FEDAVG = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "resnet50_fedavg_textbook"))


def fits():
    return [r for r in read_jsonl(f"logs/clients/{RUN}.jsonl") if r["event"] == "client_fit"]


def test_p20_every_fit_ran_with_the_configured_mu():
    records = fits()
    assert len(records) == 12
    assert all(r["proximal_mu"] == 0.01 for r in records)
    assert all(r["proximal_penalty"] > 0 and np.isfinite(r["proximal_penalty"]) for r in records)
    assert all(r["local_drift_l2"] > 0 and np.isfinite(r["local_drift_l2"]) for r in records)


def test_p20_penalty_is_consistent_with_the_logged_drift():
    """The last-epoch mean penalty cannot exceed (mu/2) * final drift^2 by much: drift grows during training."""
    for r in fits():
        assert r["proximal_penalty"] < 0.5 * r["proximal_mu"] * r["local_drift_l2"] ** 2 * 1.5


def test_p20_mu_is_in_the_audited_runtime_config():
    events = [r for r in read_jsonl(f"logs/audit/{RUN}.jsonl") if r["event"] == "runtime_config"]
    assert events and events[0]["config"]["fedprox"]["mu"] == 0.01


@pytest.mark.skipif(not os.path.isdir(FEDAVG), reason="FedAvg folder not present")
def test_p20_same_clients_order_and_examples_as_the_fedavg_run_in_both_rounds():
    theirs = [r for r in read_jsonl(os.path.join(FEDAVG, "logs", "clients", "verify-seed42.jsonl"))
              if r["event"] == "client_fit" and r["round"] <= 2]
    ours = fits()
    assert [(r["round"], r["client_id"], r["hospital"], r["num_examples"]) for r in ours] == [
        (r["round"], r["client_id"], r["hospital"], r["num_examples"]) for r in theirs]


def test_p20_gpu_peak_within_the_card():
    assert max(r["gpu_peak_mb"] for r in fits()) < 3900
    print(json.dumps({"gpu_peak_mb": max(r["gpu_peak_mb"] for r in fits())}))
