"""Test 24: from the logs of the real verification run, each client used its own hospital's properties."""
import json
import os

from tests_expected import EXPECTED

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = "verify-seed42"


def read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_24_trained_hospital_and_example_count_per_client(planned):
    fits = [r for r in read("logs", "clients", f"{RUN}.jsonl") if r["event"] == "client_fit"]
    assert fits
    counts = {h["client_id"]: h["n_samples"] for h in planned["hospitals"]}
    for fit in fits:
        assert fit["hospital"] == EXPECTED[fit["client_id"]]
        assert fit["num_examples"] == counts[fit["client_id"]]


def test_24_simulation_used_each_clients_tier_and_diversity(planned):
    outcomes = [r for r in read("logs", "op", "stragglers", f"{RUN}.jsonl") if r["event"] == "client_round_outcome"]
    assert outcomes
    facts = {h["client_id"]: h for h in planned["hospitals"]}
    for outcome in outcomes:
        fact = facts[outcome["client_id"]]
        assert outcome["tier"] == fact["tier"]
        assert abs(outcome["diversity_D_static"] - fact["D"]) < 1e-12
