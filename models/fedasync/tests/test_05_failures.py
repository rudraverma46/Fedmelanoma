"""Test 5 (added 2026-11-21): failure simulation with the FedStrag handling rule (FedAsync's paper is silent).

* FailureEnvironment is deterministic per (seed, client, job index) and independent of serving order.
* Its outcomes equal FedStrag's NoveltyEnvironment job for job (paired environments), when the FedStrag folder
  is present next to this one (skipped otherwise).
* Failure rates follow the tier Beta means; a failed job still occupies its client for a positive duration.
* The replayed schedule conserves jobs: dispatched = lost + applied, budget respected, every lost update was
  dispatched, evaluation points = ceil(applied / 6).
"""
import importlib.util
import math
import os

import numpy as np
import pytest

from hospital_registry import _TIER_CYCLE, _TIER_LATENCY
from sim.environment import FAILURE_TRACE_STREAM, FailureEnvironment
from tools.dry_run_schedule import simulate

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEDSTRAG_ENV = os.path.join(os.path.dirname(HERE), "resnet_50_fedstrag_final", "sim", "environment.py")


def records(n=10):
    out = []
    for i in range(n):
        tier = _TIER_CYCLE[i % 3]
        mu, sigma = _TIER_LATENCY[tier]
        out.append({"tier": tier, "latency_mu": mu, "latency_sigma": sigma})
    return out


def test_stream_is_fedstrags():
    assert FAILURE_TRACE_STREAM == 7921


def test_deterministic_and_order_independent():
    a, b = FailureEnvironment(42, records()), FailureEnvironment(42, records())
    fwd = [a.outcome(c, k) for c in range(10) for k in range(30)]
    rev = {(c, k): b.outcome(c, k) for c in reversed(range(10)) for k in reversed(range(30))}
    assert fwd == [rev[(c, k)] for c in range(10) for k in range(30)]
    assert [FailureEnvironment(43, records()).outcome(0, k) for k in range(30)] != fwd[:30]


@pytest.mark.skipif(not os.path.exists(FEDSTRAG_ENV), reason="FedStrag folder not next to this one")
def test_outcomes_equal_fedstrag_job_for_job():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(FEDSTRAG_ENV)))
    spec = importlib.util.spec_from_file_location("fedstrag_environment", FEDSTRAG_ENV)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # dataclasses need the module registered while it executes
    spec.loader.exec_module(mod)          # its `sim.straggler` import resolves to ours: byte-identical novelty file
    ours, theirs = FailureEnvironment(42, records()), mod.NoveltyEnvironment(42, records())
    for c in range(10):
        for k in range(60):
            o, t = ours.outcome(c, k), theirs.outcome(c, k)
            assert (o.failed, o.latency, o.busy_seconds) == (t.failed, t.latency, t.busy_seconds)


def test_failure_rates_follow_tiers_and_failed_jobs_take_time():
    env = FailureEnvironment(42, records())
    expected = {"reliable": 1 - 19 / 20, "moderate": 1 - 7 / 10, "high_straggler": 1 - 2 / 5}
    for c, rec in enumerate(records()):
        outs = [env.outcome(c, k) for k in range(3000)]
        rate = np.mean([o.failed for o in outs])
        assert rate == pytest.approx(expected[rec["tier"]], abs=0.03)
        assert all(o.busy_seconds > 0 and math.isfinite(o.busy_seconds) for o in outs)
        assert all(math.isinf(o.latency) for o in outs if o.failed)
        assert all(o.busy_seconds == o.latency for o in outs if not o.failed)


def test_schedule_conserves_jobs():
    r = simulate(10, 300, 6, 6, 42)
    d, lost, applied = (sum(r[k].values()) for k in ("dispatched", "lost", "applied"))
    assert d == 300 and d == lost + applied and lost > 0
    assert r["evaluations"] == math.ceil(applied / 6)
    for c in range(10):
        assert r["dispatched"][c] == r["lost"][c] + r["applied"][c]
