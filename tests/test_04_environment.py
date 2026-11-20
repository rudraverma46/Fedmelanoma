"""Test 4: the straggler and failure environment is the novelty one.

* Tiers come from hospital_registry (index mod 3 over reliable, moderate, high_straggler); all 10 hospitals.
* Every outcome equals a direct call of the novelty StragglerManager.simulate_response on the client's own stream.
* Per-client traces are deterministic in (seed, client, job index) and independent of draw order.
* Failure rates follow the tier Beta means (0.05 / 0.30 / 0.60); delays follow the tier LogNormal medians.
* Only simulate_response runs: the novelty window, buffer, quorum and discount are never touched.
"""
import numpy as np
import pytest

from hospital_registry import build_hospital_registry
from sim.environment import TRACE_STREAM, NoveltyEnvironment
from sim.straggler import StragglerManager


def records(cfg):
    return build_hospital_registry(cfg["data"]["partitions_root"])[1]


def test_tiers_and_parameters(cfg):
    recs = records(cfg)
    assert len(recs) == 10
    assert [r["tier"] for r in recs] == ["reliable", "moderate", "high_straggler"] * 3 + ["reliable"]
    assert {r["tier"]: (r["latency_mu"], r["latency_sigma"]) for r in recs} == {
        "reliable": (1.39, 0.30), "moderate": (2.77, 0.50), "high_straggler": (4.09, 0.80)}


def test_outcomes_equal_direct_novelty_calls(cfg):
    recs = records(cfg)
    env = NoveltyEnvironment(42, recs)
    for c in range(10):
        manager = StragglerManager(rng=np.random.default_rng([42, TRACE_STREAM, c]))
        stub = env.stubs[c]
        for k in range(40):
            r = manager.simulate_response(client=stub, current_round=k)
            o = env.outcome(c, k)
            assert o.failed == r.is_failed
            assert (np.isinf(o.latency) and r.is_failed) or o.latency == r.response_time


def test_failed_job_duration_reuses_the_tier_lognormal_on_a_child_stream(cfg):
    """Correction 2026-11-20: a failed job occupies its client for one LogNormal(mu, sigma) draw of its tier, taken from
    child stream [42, 7921, c, 1]; a delivered job is busy exactly until it arrives."""
    from sim.environment import FAILED_DURATION_SUBSTREAM
    recs = records(cfg)
    env = NoveltyEnvironment(42, recs)
    for c in range(10):
        child = np.random.default_rng([42, TRACE_STREAM, c, FAILED_DURATION_SUBSTREAM])
        expected = [float(child.lognormal(mean=recs[c]["latency_mu"], sigma=recs[c]["latency_sigma"])) for _ in range(60)]
        for k in range(60):
            o = env.outcome(c, k)
            assert o.busy_seconds == (expected[k] if o.failed else o.latency)
    for c, median in {0: 4.0, 1: 16.0, 2: 60.0}.items():   # same tier medians as successful jobs
        assert np.median([env.failed_duration(c, k) for k in range(4000)]) == pytest.approx(median, rel=0.1)


def test_main_stream_unchanged_by_the_failed_duration_correction(cfg):
    """failed / latency of every job equal a plain replay of simulate_response on stream 7921 alone."""
    recs = records(cfg)
    env = NoveltyEnvironment(42, recs)
    for c in range(10):
        manager = StragglerManager(rng=np.random.default_rng([42, TRACE_STREAM, c]))
        for k in range(200):
            r = manager.simulate_response(client=env.stubs[c], current_round=k)
            o = env.outcome(c, k)
            assert o.failed == r.is_failed and (o.latency == r.response_time or (np.isinf(o.latency) and r.is_failed))


def test_trace_is_deterministic_and_order_independent(cfg):
    recs = records(cfg)
    a, b = NoveltyEnvironment(42, recs), NoveltyEnvironment(42, recs)
    forward = [a.outcome(c, k) for c in range(10) for k in range(30)]
    shuffled = {(c, k): b.outcome(c, k) for k in reversed(range(30)) for c in reversed(range(10))}
    assert forward == [shuffled[(c, k)] for c in range(10) for k in range(30)]
    other = NoveltyEnvironment(43, recs)
    assert [other.outcome(0, k) for k in range(30)] != [a.outcome(0, k) for k in range(30)]


def test_failure_rates_and_delays_follow_the_tiers(cfg):
    env = NoveltyEnvironment(42, records(cfg))
    n = 4000
    for c, (fail_rate, median) in {0: (0.05, 4.0), 1: (0.30, 16.0), 2: (0.60, 60.0)}.items():
        out = [env.outcome(c, k) for k in range(n)]
        failed = np.mean([o.failed for o in out])
        assert failed == pytest.approx(fail_rate, abs=0.03)
        lat = [o.latency for o in out if not o.failed]
        assert all(np.isfinite(lat)) and min(lat) > 0
        assert np.median(lat) == pytest.approx(median, rel=0.1)
        assert all(np.isinf(o.latency) for o in out if o.failed)


def test_novelty_reaction_to_stragglers_is_never_used(cfg):
    env = NoveltyEnvironment(42, records(cfg))
    for c in range(10):
        for k in range(50):
            env.outcome(c, k)
    for m in env.managers:
        assert m.window_history == [] and m.straggler_counts == [] and m.on_time_counts == []
        assert m.get_buffer_state() == {} and m.current_W_t == m.w_min and len(m._latency_history) == 0
