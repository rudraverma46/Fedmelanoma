"""Test 4: the straggler environment is the novelty one, and paired across algorithms.

* sim/straggler.py is byte-identical to the novelty server/straggler.py.
* Tiers come from hospital_registry (index mod 3 over reliable, moderate, high_straggler); all 10 hospitals.
* Per-client traces are deterministic in (seed, client, job index) and independent of draw order.
* Draws equal StragglerManager.simulate_response called directly with the same generator.
* The empirical failure rate and latency medians match the tier parameters.
"""
import hashlib
import os

import numpy as np
import pytest

from hospital_registry import build_hospital_registry
from sim.environment import TRACE_STREAM, TraceEnvironment
from sim.straggler import StragglerManager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOVELTY = os.path.abspath(os.path.join(ROOT, "..", "RESNET 50_ fed novelty", "RESNET50_NOVELTY MODEL", "server", "straggler.py"))


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def records(cfg):
    return build_hospital_registry(cfg["data"]["partitions_root"])[1]


def test_straggler_module_is_the_novelty_file():
    assert sha(os.path.join(ROOT, "sim", "straggler.py")) == sha(NOVELTY)


def test_tiers_and_parameters(cfg):
    recs = records(cfg)
    assert len(recs) == 10
    assert [r["tier"] for r in recs] == ["reliable", "moderate", "high_straggler"] * 3 + ["reliable"]
    assert {r["tier"]: (r["latency_mu"], r["latency_sigma"]) for r in recs} == {
        "reliable": (1.39, 0.30), "moderate": (2.77, 0.50), "high_straggler": (4.09, 0.80)}


def test_trace_is_deterministic_and_order_independent(cfg):
    recs = records(cfg)
    a, b = TraceEnvironment(42, recs), TraceEnvironment(42, recs)
    forward = [a.draw(c, k) for c in range(10) for k in range(30)]
    shuffled = {(c, k): b.draw(c, k) for k in reversed(range(30)) for c in reversed(range(10))}
    assert forward == [shuffled[(c, k)] for c in range(10) for k in range(30)]
    other = TraceEnvironment(43, recs)
    assert [other.draw(0, k) for k in range(30)] != [a.draw(0, k) for k in range(30)]


def test_draws_equal_the_novelty_simulator(cfg):
    recs = records(cfg)
    env = TraceEnvironment(42, recs)
    for c in (0, 1, 2):
        mgr = StragglerManager(rng=np.random.default_rng([42, TRACE_STREAM, c]))
        for k in range(20):
            r = mgr.simulate_response(env.stubs[c], k)
            assert env.draw(c, k) == (bool(r.is_failed), float(r.response_time))


def test_failure_rate_and_latency_follow_the_tier(cfg):
    env = TraceEnvironment(42, records(cfg))
    for c, (fail_rate, median) in {0: (0.05, 4.0), 1: (0.30, 16.0), 2: (0.60, 60.0)}.items():
        draws = [env.draw(c, k) for k in range(4000)]
        assert np.mean([f for f, _ in draws]) == pytest.approx(fail_rate, abs=0.03)
        ok = [t for f, t in draws if not f]
        assert np.median(ok) == pytest.approx(median, rel=0.1)
        assert all(np.isfinite(t) and t > 0 for t in ok)


def test_latency_only_stream_is_deterministic_and_follows_the_tier(cfg):
    recs = records(cfg)
    a, b = TraceEnvironment(42, recs), TraceEnvironment(42, recs)
    assert [a.latency(3, k) for k in range(25)] == [b.latency(3, k) for k in reversed(range(25))][::-1]
    for c, median in {0: 4.0, 1: 16.0, 2: 60.0}.items():
        lat = [a.latency(c, k) for k in range(4000)]
        assert np.median(lat) == pytest.approx(median, rel=0.1) and min(lat) > 0
