"""Tests 11-16: FedAsync's equations, staleness, version tracking, asynchrony and the proximal worker.

Expected values are computed by hand or with plain numpy, not with the code under test.
"""
import numpy as np
import pytest
import torch

from server.fedasync import FedAsyncServer, make_staleness_function, mix, staleness_hinge, staleness_polynomial
from server.scheduler import VirtualClockScheduler

HINGE = make_staleness_function("hinge", 10, 4)


# ---- Test 11: staleness functions, by hand (paper Section 5) --------------------------------

def test_11_hinge_by_hand():
    assert [HINGE(s) for s in range(5)] == [1.0] * 5          # t - tau <= b = 4
    assert HINGE(5) == pytest.approx(1 / 11)                     # 1 / (10 * 1 + 1)
    assert HINGE(6) == pytest.approx(1 / 21)
    assert HINGE(10) == pytest.approx(1 / 61)
    values = [HINGE(s) for s in range(30)]
    assert all(a >= b for a, b in zip(values, values[1:]))       # non-increasing


def test_11_other_functions_by_hand():
    assert staleness_polynomial(0, 0.5) == 1.0
    assert staleness_polynomial(3, 0.5) == pytest.approx(0.5)    # (3 + 1)^-0.5
    assert make_staleness_function("constant")(17) == 1.0
    assert staleness_hinge(4, 10, 4) == 1.0 and staleness_hinge(5, 10, 4) < 1.0
    with pytest.raises(ValueError):
        make_staleness_function("exponential")


# ---- Test 12: the mixing update, by hand -----------------------------------------------------

def test_12_mix_by_hand_float_and_integer():
    g = [np.array([1.0, 2.0], np.float32), np.array(10, np.int64)]
    c = [np.array([3.0, 6.0], np.float32), np.array(15, np.int64)]
    out = mix(g, c, 0.6)
    assert np.allclose(out[0], [0.4 * 1 + 0.6 * 3, 0.4 * 2 + 0.6 * 6])
    assert out[0].dtype == np.float32
    assert out[1].dtype == np.int64 and int(out[1]) == 13      # 0.4*10 + 0.6*15 = 13
    with pytest.raises(ValueError):
        mix(g, [c[0]], 0.5)
    with pytest.raises(ValueError):
        mix([np.zeros(2)], [np.zeros(3)], 0.5)


def test_12_server_sequence_by_hand():
    """Three updates; staleness = applied-since-dispatch; alpha_t = 0.6 * s; x tracked by hand."""
    server = FedAsyncServer([np.array([0.0])], 0.6, HINGE)
    x = 0.0
    r1 = server.apply(0, [np.array([10.0])], tau=0)            # fresh: staleness 0, alpha 0.6
    x = 0.4 * x + 0.6 * 10.0
    assert (r1.staleness, r1.alpha_t, r1.version_before, r1.version_after) == (0, 0.6, 0, 1)
    for _ in range(5):                                        # five more fresh updates from client 1
        server.apply(1, [np.array([0.0])], tau=server.version)
        x = 0.4 * x
    r2 = server.apply(2, [np.array([100.0])], tau=0)          # dispatched at v=0, applied at v=6: staleness 6
    alpha = 0.6 / (10 * (6 - 4) + 1)
    x = (1 - alpha) * x + alpha * 100.0
    assert r2.staleness == 6 and r2.s_value == pytest.approx(1 / 21) and r2.alpha_t == pytest.approx(alpha)
    assert server.x[0][0] == pytest.approx(x) and server.version == 7
    assert all(r.accepted for r in (r1, r2))
    with pytest.raises(ValueError):
        server.apply(0, [np.array([0.0])], tau=99)            # an update from the future


def test_12_not_fedavg():
    """Two equal-size clients from the same start: FedAvg gives their mean, FedAsync does not."""
    a, b = np.array([1.0]), np.array([3.0])
    server = FedAsyncServer([np.array([0.0])], 0.6, HINGE)
    server.apply(0, [a], tau=0)
    server.apply(1, [b], tau=0)                               # staleness 1, s = 1
    expected = 0.4 * (0.4 * 0.0 + 0.6 * 1.0) + 0.6 * 3.0      # = 1.896
    assert server.x[0][0] == pytest.approx(expected)
    assert server.x[0][0] != pytest.approx((1.0 + 3.0) / 2)
    # order matters (it would not for an average)
    other = FedAsyncServer([np.array([0.0])], 0.6, HINGE)
    other.apply(1, [b], tau=0)
    other.apply(0, [a], tau=0)
    assert other.x[0][0] != pytest.approx(server.x[0][0])


# ---- Test 13-14: the scheduler is asynchronous and tracks versions ----------------------------

def simulate(durations, total, concurrency=6, seed=42):
    """The runner's loop (apply, then dispatch into the freed slot) with fixed per-client durations."""
    rng = np.random.default_rng(seed)
    server = FedAsyncServer([np.array([0.0])], 0.6, HINGE)
    sched = VirtualClockScheduler(range(len(durations)), concurrency, rng)
    dispatched = []

    def dispatch():
        cid = sched.pick_idle()
        x, tau = server.snapshot()
        dispatched.append((cid, tau, sched.clock))
        sched.submit(cid, tau, durations[cid], [x[0] + 1.0], 1, {})

    for _ in range(concurrency):
        dispatch()
    records = []
    while len(records) < total:
        job = sched.next_arrival()
        records.append((job, server.apply(job.client_id, job.arrays, job.tau, job.arrival_time, job.dispatch_time)))
        if len(records) < total:
            dispatch()
        assert len(sched.in_flight) == (concurrency if len(records) < total else concurrency - 1)
    return records, dispatched, sched


SIZES_SECONDS = [44, 16, 16, 40, 119, 124, 132, 18, 13, 84]   # textbook FedAvg's mean fit time per hospital


def test_13_updates_applied_in_arrival_order_without_waiting():
    records, _, sched = simulate(SIZES_SECONDS, 300)
    arrivals = [job.arrival_time for job, _ in records]
    assert arrivals == sorted(arrivals)
    # An update dispatched EARLIER is applied AFTER a later-dispatched one: the server did not wait for it.
    overtaken = sum(1 for (j1, _), (j2, _) in zip(records, records[1:]) if j2.dispatch_time < j1.dispatch_time)
    assert overtaken > 0
    counts = np.bincount([job.client_id for job, _ in records], minlength=10)
    assert counts.sum() == 300 and counts.max() > counts.min()   # fast clients contribute more often


def test_14_version_and_staleness_tracking():
    records, dispatched, _ = simulate(SIZES_SECONDS, 300)
    for i, (job, rec) in enumerate(records):
        assert rec.version_before == i and rec.version_after == i + 1
        assert rec.staleness == rec.version_before - job.tau >= 0
        assert rec.alpha_t == pytest.approx(0.6 * HINGE(rec.staleness))
    stale = [rec.staleness for _, rec in records]
    assert max(stale) > 4                       # the hinge actually down-weights some updates
    assert any(rec.alpha_t < 0.6 for _, rec in records) and any(rec.alpha_t == 0.6 for _, rec in records)
    # every dispatch carried the server version at that moment
    assert [tau for _, tau, _ in dispatched][:6] == [0] * 6


def test_14_scheduler_draw_is_uniform_over_idle_and_seeded():
    sched = VirtualClockScheduler(range(10), 6, np.random.default_rng(42))
    expected_rng = np.random.default_rng(42)
    pool = list(range(10))
    for _ in range(6):
        want = int(expected_rng.choice(pool))
        pool.remove(want)
        assert sched.pick_idle() == want
    assert sched.idle == sorted(pool)
    with pytest.raises(ValueError):
        sched.submit(0, 0, 0.0, None, 1, {})


def test_14_ties_break_by_dispatch_order():
    sched = VirtualClockScheduler(range(3), 3, np.random.default_rng(0))
    for cid in (2, 0, 1):
        sched.idle.remove(cid)
        sched.submit(cid, 0, 5.0, None, 1, {})
    assert [sched.next_arrival().client_id for _ in range(3)] == [2, 0, 1]


# ---- Test 15: the worker objective g = f + (rho/2)||x - x_tau||^2 ----------------------------

def test_15_proximal_term_value_and_gradient():
    from client.proximal import proximal_penalty, snapshot_global_reference
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 2)
    reference = snapshot_global_reference(model)
    with torch.no_grad():
        model.weight.add_(0.5)
    rho = 0.005
    penalty = proximal_penalty(model, reference, rho)
    by_hand = 0.5 * rho * sum(float((p - r).pow(2).sum()) for p, r in zip(model.parameters(), reference))
    assert float(penalty) == pytest.approx(by_hand)
    penalty.backward()
    assert torch.allclose(model.weight.grad, rho * (model.weight - reference[0]).detach())
    assert torch.allclose(model.bias.grad, torch.zeros_like(model.bias))


# ---- Test 16: the full ResNet-50 state, all 329 tensors -----------------------------------------

def test_16_real_model_mix_all_tensors():
    from models.model import FedMelanomaModel
    torch.manual_seed(0)
    model = FedMelanomaModel()
    g = [v.detach().cpu().numpy().copy() for v in model.state_dict().values()]
    rng = np.random.default_rng(0)
    c = [a + rng.normal(0, 0.01, a.shape).astype(a.dtype) if a.dtype.kind == "f" else a + 7 for a in g]
    server = FedAsyncServer(g, 0.6, HINGE)
    server.apply(3, c, tau=0)
    assert len(server.x) == 329
    for a, b, out in zip(g, c, server.x):
        assert out.dtype == a.dtype and out.shape == a.shape
        if a.dtype.kind == "f":
            assert np.allclose(out, 0.4 * a.astype(np.float64) + 0.6 * b.astype(np.float64), rtol=1e-6, atol=1e-7)
        else:
            assert int(out) == int(np.rint(0.4 * int(a) + 0.6 * int(b)))
    from collections import OrderedDict
    FedMelanomaModel().load_state_dict(OrderedDict((k, torch.tensor(v)) for k, v in zip(model.state_dict().keys(), server.x)), strict=True)
