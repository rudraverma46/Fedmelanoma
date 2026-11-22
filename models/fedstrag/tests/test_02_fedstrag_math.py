"""Tests 11-16: FedStrag's weights and aggregation, the time-bounded protocol, the FIFO backlog, failures,
empty iterations and termination.

Expected values are computed by hand or with plain numpy, not with the code under test. The protocol is driven
by a fake trainer and a scripted environment (delay and failure per client and job), so event timing and every
counter can be checked exactly.
"""
import pickle

import numpy as np
import pytest

from server.fedstrag import WAIT_TIME_SECONDS, async_w_fedavg, staleness_weights
from server.protocol import FedStragProtocol, SnapshotStore
from sim.environment import JobOutcome


# ---- Test 11: weights and aggregation, by hand (paper Eq. 6, 7, 8) ----------------------------------------------

def test_11_weights_by_hand():
    # the latest broadcast has tag 4 (iteration_now = 5): staleness 0, 0, 1, 4
    assert np.allclose(staleness_weights([4, 4, 3, 0], 5), [1.0, 1.0, 1 / 2, 1 / 5])
    with pytest.raises(ValueError):
        staleness_weights([5], 5)                       # trained on a broadcast that does not exist yet


def test_11_paper_worked_example_eq8():
    """Section 5.2: D = [60, 20, 80], tags 4, 4, 3 -> W = [1, 1, 0.5]; theta = (60 t1 + 20 t2 + 40 t3) / 120."""
    t = [[np.array([2.0])], [np.array([5.0])], [np.array([11.0])]]
    out, record = async_w_fedavg(t, [4, 4, 3], 5, [60, 20, 80])
    assert np.allclose(staleness_weights([4, 4, 3], 5), [1.0, 1.0, 0.5])
    assert out[0][0] == pytest.approx((1 * 2.0 * 60 + 1 * 5.0 * 20 + 0.5 * 11.0 * 80) / (60 * 1 + 20 * 1 + 80 * 0.5))
    assert [round(r["normalised_weight"], 6) for r in record] == [0.5, round(1 / 6, 6), round(1 / 3, 6)]
    assert [r["staleness"] for r in record] == [0, 0, 1]


def test_11_aggregation_replaces_global_and_weights_by_samples():
    a = [np.array([1.0, 2.0], np.float32), np.array(10, np.int64)]
    b = [np.array([4.0, 8.0], np.float32), np.array(20, np.int64)]
    out, record = async_w_fedavg([a, b], [4, 3], 5, [50, 150])        # staleness 0 and 1
    ca, cb = 1.0 * 50, 0.5 * 150
    assert np.allclose(out[0], (ca * np.array([1.0, 2.0]) + cb * np.array([4.0, 8.0])) / (ca + cb))
    assert out[0].dtype == np.float32 and out[1].dtype == np.int64
    assert int(out[1]) == round((ca * 10 + cb * 20) / (ca + cb))
    assert sum(r["normalised_weight"] for r in record) == pytest.approx(1.0)
    single, _ = async_w_fedavg([b], [0], 9, [7])
    assert np.allclose(single[0], b[0])                                   # one update: global = that model
    with pytest.raises(ValueError):
        async_w_fedavg([], [], 3, [])
    with pytest.raises(ValueError):
        async_w_fedavg([a], [4], 5, [0])


def test_11_equal_staleness_is_plain_weighting_by_size():
    small, large = [np.array([0.0])], [np.array([10.0])]
    out, _ = async_w_fedavg([small, large], [2, 2], 3, [155, 1716])
    assert out[0][0] == pytest.approx((155 * 0.0 + 1716 * 10.0) / (155 + 1716))


# ---- protocol harness ---------------------------------------------------------------------------------------------

class ScriptedEnv:
    """outcome(c, k): delay of client c's k-th job is c + 1 s and nothing fails, unless scripted otherwise.
    A failed job occupies its client for fail_time(c, k) seconds (default 2 s)."""

    def __init__(self, latency=None, fails=None, fail_time=None):
        self.latency = latency or (lambda c, k: float(c + 1))
        self.fails = fails or (lambda c, k: False)
        self.fail_time = fail_time or (lambda c, k: 2.0)
        self.stubs = [type("S", (), {"tier": "t"})() for _ in range(10)]
        self.calls = []

    def outcome(self, c, k):
        self.calls.append((c, k))
        if self.fails(c, k):
            return JobOutcome(failed=True, latency=float("inf"), busy_seconds=float(self.fail_time(c, k)))
        lat = float(self.latency(c, k))
        return JobOutcome(failed=False, latency=lat, busy_seconds=lat)


def fake_protocol(tmp_path, env=None, iterations=50):
    calls = []

    def train(client_id, arrays):
        calls.append((client_id, float(arrays[0][0])))
        return [arrays[0] + 1.0], 100 + client_id, {"train_loss": 0.5}, 0.01

    store = SnapshotStore(str(tmp_path / "snap"))
    protocol = FedStragProtocol([np.array([0.0])], 10, env or ScriptedEnv(), iterations, store, train)
    return protocol, calls, store


def run_all(protocol):
    infos = []
    while not protocol.done():
        infos.append(protocol.run_iteration())
    return infos


def check_invariants(protocol):
    t = protocol.totals()
    assert t["dispatched"] == t["completed_trainings"] == t["failed_attempts"] + t["updates_generated"]
    assert t["updates_generated"] == t["applied"] + t["pending_updates"]
    assert t["unaccounted_updates"] == 0 and t["arrived_not_aggregated"] == 0
    assert t["updates_received"] == t["applied"]
    assert t["in_flight_updates"] + t["failing_jobs_in_progress"] == len(protocol.busy)
    assert t["failing_jobs_in_progress"] <= t["failed_attempts"]
    return t


# ---- Test 12: broadcast, tags, polling ------------------------------------------------------------------------------

def test_12_first_iteration_by_hand(tmp_path):
    protocol, calls, _ = fake_protocol(tmp_path)
    info = protocol.run_iteration()
    assert info["iteration"] == 1 and info["tag_broadcast"] == 0 and info["latest_tag"] == 0
    assert info["receivers"] == list(range(10))                       # every hospital receives the broadcast
    assert sorted(c for c, _ in calls) == list(range(10)) and all(x == 0.0 for _, x in calls)   # all train tag 0
    # poll at 4 s: clients 0..3 (1, 2, 3, 4 s) have arrived
    assert info["poll_times"] == [4.0] and info["aggregated_at"] == 4.0
    assert [a["client_id"] for a in info["aggregated"]] == [0, 1, 2, 3]
    assert all(a["staleness"] == 0 and a["raw_weight"] == 1.0 and a["tag"] == 0 for a in info["aggregated"])
    assert np.allclose(protocol.x[0], [1.0]) and info["model_version_after"] == 1
    t = check_invariants(protocol)
    assert (t["dispatched"], t["failed_attempts"], t["applied"], t["in_flight_updates"]) == (10, 0, 4, 6)


def test_12_every_broadcast_reaches_every_client_with_the_right_tag(tmp_path):
    protocol, calls, _ = fake_protocol(tmp_path, iterations=12)
    infos = run_all(protocol)
    for i, info in enumerate(infos, start=1):
        assert info["iteration"] == i and info["tag_broadcast"] == i - 1 == info["latest_tag"]
        assert info["receivers"] == list(range(10))
    broadcasts = [e for e in protocol.events if e["event"] == "broadcast"]
    assert [e["tag"] for e in broadcasts] == list(range(12))
    assert protocol.store.tags()[-1] == 11                             # every tag was stored for the clients


def test_12_polling_is_exactly_every_4_seconds(tmp_path):
    assert WAIT_TIME_SECONDS == 4.0
    env = ScriptedEnv(latency=lambda c, k: 9.5 + c)                   # nobody before 9.5 s: polls at 4, 8, 12
    protocol, _, _ = fake_protocol(tmp_path, env, iterations=8)
    infos = run_all(protocol)
    assert infos[0]["poll_times"] == [4.0, 8.0, 12.0] and infos[0]["aggregated_at"] == 12.0
    for info in infos:
        expected = [info["broadcast_time"] + 4.0 * (m + 1) for m in range(info["polls"])]
        assert info["poll_times"] == pytest.approx(expected)
        assert len(info["aggregated"]) >= 1                           # it stops at the first poll with >= 1 update
        for p in info["poll_times"][:-1]:                             # earlier polls had nothing to aggregate
            assert all(a["arrival_time"] > p for a in info["aggregated"])


# ---- Test 13: FIFO backlog, stale updates, sample weighting ---------------------------------------------------------

def test_13_fifo_order_and_slow_clients_keep_every_update(tmp_path):
    env = ScriptedEnv(latency=lambda c, k: 30.0 if c >= 7 else 1.0)   # three slow clients
    protocol, calls, _ = fake_protocol(tmp_path, env, iterations=25)
    infos = run_all(protocol)
    for c in range(10):
        trained = [a["tag"] for i in infos for a in i["aggregated"] if a["client_id"] == c]
        assert trained == list(range(len(trained)))                  # strictly oldest first, none skipped/overwritten
        jobs = [a["job_index"] for i in infos for a in i["aggregated"] if a["client_id"] == c]
        assert jobs == list(range(len(jobs)))
    slow = [a for i in infos for a in i["aggregated"] if a["client_id"] == 8]
    assert len(slow) >= 3                                             # several updates from one slow client retained
    assert max(a["staleness"] for a in slow) > 3                      # and they grow stale, but still count
    assert all(len(protocol.queues[c]) > 0 for c in (7, 8, 9))       # the backlog is kept, not dropped
    check_invariants(protocol)


def test_13_staleness_and_weights_by_hand(tmp_path):
    env = ScriptedEnv(latency=lambda c, k: 30.0 if c == 9 else 1.0)
    protocol, calls, _ = fake_protocol(tmp_path, env, iterations=15)
    infos = run_all(protocol)
    stale = [(i, a) for i in infos for a in i["aggregated"] if a["staleness"] > 0]
    assert stale, "the slow client must deliver stale updates"
    for info, a in stale:
        assert a["staleness"] == info["latest_tag"] - a["tag"] == (info["iteration"] - 1) - a["tag"]
        assert a["raw_weight"] == pytest.approx(1.0 / (a["staleness"] + 1))
    for info in infos:
        for a in info["aggregated"]:
            assert a["num_examples_weight"] == 100 + a["client_id"]   # D = the client's sample count
        coeff = [a["raw_weight"] * a["num_examples_weight"] for a in info["aggregated"]]
        assert [a["normalised_weight"] for a in info["aggregated"]] == pytest.approx([c / sum(coeff) for c in coeff])
        assert sum(a["normalised_weight"] for a in info["aggregated"]) == pytest.approx(1.0)


def test_13_aggregate_value_by_hand(tmp_path):
    """Iteration 2 of a two-client scenario computed by hand from W*D."""
    env = ScriptedEnv(latency=lambda c, k: {0: 1.0, 1: 5.0}.get(c, 1e9))
    protocol, _, _ = fake_protocol(tmp_path, env, iterations=2)
    first = protocol.run_iteration()          # t=4: only client 0 (tag 0) -> x = 1.0
    assert [a["client_id"] for a in first["aggregated"]] == [0] and protocol.x[0][0] == 1.0
    second = protocol.run_iteration()         # broadcast tag 1 at t=4; client 1 (tag 0, D 101) arrives at 5;
    got = {a["client_id"]: a for a in second["aggregated"]}   # client 0 (tag 1, D 100) arrives at 5
    assert set(got) == {0, 1} and second["aggregated_at"] == 8.0
    c0, c1 = 1.0 * 100, 0.5 * 101             # staleness 0 and 1
    assert protocol.x[0][0] == pytest.approx((c0 * 2.0 + c1 * 1.0) / (c0 + c1))


def test_13_an_update_is_aggregated_exactly_once(tmp_path):
    protocol, calls, _ = fake_protocol(tmp_path, iterations=15)
    infos = run_all(protocol)
    keys = [(a["client_id"], a["tag"]) for i in infos for a in i["aggregated"]]
    assert len(keys) == len(set(keys))
    assert len(calls) == len(keys) + len(protocol.busy)


# ---- Test 14: failures ------------------------------------------------------------------------------------------------

def test_14_failed_updates_never_enter_aggregation(tmp_path):
    fails = lambda c, k: (c + k) % 3 == 0                             # a third of all jobs fail
    env = ScriptedEnv(latency=lambda c, k: 1.0 + (c % 4) * 3.0, fails=fails)
    protocol, calls, _ = fake_protocol(tmp_path, env, iterations=20)
    infos = run_all(protocol)
    applied = {(a["client_id"], a["job_index"]) for i in infos for a in i["aggregated"]}
    failed = {(c, k) for c, k in env.calls if fails(c, k)}
    assert failed and not (applied & failed)
    t = check_invariants(protocol)
    assert t["failed_attempts"] == len(failed) == t["updates_lost_to_failures"]
    assert t["completed_trainings"] == len(calls)                     # failed jobs were trained first (novelty timing)
    failed_events = [e for e in protocol.events if e["event"] == "client_failed"]
    assert len(failed_events) == len(failed) and all(e["status"] == "trained_update_will_be_lost" for e in failed_events)
    lost = [e for e in protocol.events if e["event"] == "update_lost"]
    assert len(lost) == len(failed) - t["failing_jobs_in_progress"]           # every finished failure is recorded


def test_14_failed_job_consumes_its_duration_on_the_virtual_clock():
    """Client 0: job 0 delivers after 30 s; jobs 1 and 2 fail after 5 s and 7 s of work; job 3 delivers after 1 s.
    The failures occupy the client: job 2 starts at 35, job 3 at 42 (not all at 30, as with zero-time failures)."""
    import tempfile
    env = ScriptedEnv(latency=lambda c, k: 30.0 if (c, k) == (0, 0) else 1.0,
                      fails=lambda c, k: c == 0 and k in (1, 2), fail_time=lambda c, k: {1: 5.0, 2: 7.0}.get(k, 2.0))
    protocol = FedStragProtocol([np.array([0.0])], 10, env, 15, SnapshotStore(tempfile.mkdtemp()),
                                lambda c, a: ([a[0] + 1.0], 100, {}, 0.0))
    run_all(protocol)
    jobs0 = {e["job_index"]: e for e in protocol.events if e.get("client_id") == 0 and e["event"] in ("client_start", "client_failed")}
    assert (jobs0[0]["start_time"], jobs0[0]["finish_time"]) == (0.0, 30.0)
    assert (jobs0[1]["event"], jobs0[1]["start_time"], jobs0[1]["finish_time"]) == ("client_failed", 30.0, 35.0)
    assert (jobs0[2]["event"], jobs0[2]["start_time"], jobs0[2]["finish_time"]) == ("client_failed", 35.0, 42.0)
    assert (jobs0[3]["event"], jobs0[3]["start_time"], jobs0[3]["finish_time"]) == ("client_start", 42.0, 43.0)
    lost = [(e["job_index"], e["failure_time"]) for e in protocol.events if e["event"] == "update_lost" and e["client_id"] == 0]
    assert lost == [(1, 35.0), (2, 42.0)]
    check_invariants(protocol)


def test_14_server_keeps_polling_while_a_failing_job_runs(tmp_path):
    """Every first job fails after 9 s: the server cannot know, polls at 4 and 8, and only at 12 (no client busy,
    nothing arrived) ends the iteration with the model unchanged."""
    env = ScriptedEnv(fails=lambda c, k: k == 0, fail_time=lambda c, k: 9.0)
    protocol, _, _ = fake_protocol(tmp_path, env, iterations=2)
    before = protocol.x[0].copy()
    info = protocol.run_iteration()
    assert info["empty"] and info["poll_times"] == [4.0, 8.0, 12.0]
    assert np.array_equal(protocol.x[0], before) and protocol.model_version == 0
    assert protocol.totals()["failing_jobs_in_progress"] == 0 and protocol.totals()["failed_attempts"] == 10


def test_14_a_failure_carries_no_penalty(tmp_path):
    """A client that failed keeps receiving every broadcast and its next job runs normally."""
    env = ScriptedEnv(fails=lambda c, k: c == 0 and k < 3)            # client 0's first three jobs fail
    protocol, calls, _ = fake_protocol(tmp_path, env, iterations=10)
    infos = run_all(protocol)
    assert all(0 in i["receivers"] for i in infos)
    applied0 = [a for i in infos for a in i["aggregated"] if a["client_id"] == 0]
    assert applied0 and applied0[0]["job_index"] == 3 and all(a["raw_weight"] == 1.0 for a in applied0)
    assert protocol.counts["failed_attempts"][0] == 3


def test_14_late_successful_updates_are_never_discarded(tmp_path):
    env = ScriptedEnv(latency=lambda c, k: 200.0 if c >= 7 else 1.0)  # far slower than any iteration
    protocol, _, _ = fake_protocol(tmp_path, env, iterations=60)
    infos = run_all(protocol)
    late = [a for i in infos for a in i["aggregated"] if a["client_id"] >= 7]
    assert late and max(a["staleness"] for a in late) >= 40          # very stale, still applied
    t = check_invariants(protocol)
    assert t["applied"] + t["in_flight_updates"] == t["updates_generated"]


# ---- Test 15: empty iterations and termination ------------------------------------------------------------------------

def test_15_global_unchanged_when_no_update_can_arrive(tmp_path):
    env = ScriptedEnv(fails=lambda c, k: k == 0)                      # every client's first job fails
    protocol, _, _ = fake_protocol(tmp_path, env, iterations=3)
    before = protocol.x[0].copy()
    info = protocol.run_iteration()
    assert info["empty"] and info["aggregated"] == [] and info["poll_times"] == [4.0]
    assert np.array_equal(protocol.x[0], before) and info["model_version_after"] == 0 == protocol.model_version
    assert protocol.iter == 1 and protocol.empty_iterations == 1
    nxt = protocol.run_iteration()                                    # the next broadcast proceeds normally
    assert not nxt["empty"] and nxt["aggregated"] and protocol.model_version == 1
    assert all(a["tag"] == 1 for a in nxt["aggregated"])


def test_15_termination_reports_failing_jobs_separately(tmp_path):
    env = ScriptedEnv(latency=lambda c, k: 1.0, fails=lambda c, k: c == 9, fail_time=lambda c, k: 100.0)
    protocol, _, _ = fake_protocol(tmp_path, env, iterations=5)
    run_all(protocol)
    t = check_invariants(protocol)
    assert t["failing_jobs_in_progress"] == 1 and protocol.busy[9].failed       # still running at the end
    assert t["failed_attempts"] == 1 and t["in_flight_updates"] == 0            # counted once, never as pending
    assert protocol.counts["applied"][9] == 0 and protocol.counts["updates_received"][9] == 0


def test_15_termination_excludes_pending_updates(tmp_path):
    env = ScriptedEnv(latency=lambda c, k: 50.0 if c >= 8 else 1.0)
    protocol, calls, _ = fake_protocol(tmp_path, env, iterations=10)
    infos = run_all(protocol)
    assert protocol.done() and protocol.iter == 10 and len(infos) == 10
    with pytest.raises(RuntimeError):
        protocol.run_iteration()                                      # nothing after the last iteration
    t = check_invariants(protocol)
    assert t["in_flight_updates"] == len(protocol.busy) > 0           # excluded, reported (no failures here)
    assert t["queued_broadcasts_not_started"] > 0
    assert t["applied"] == sum(len(i["aggregated"]) for i in infos)
    excluded = {(c, j.tag) for c, j in protocol.busy.items()}
    applied = {(a["client_id"], a["tag"]) for i in infos for a in i["aggregated"]}
    assert not (excluded & applied)


def test_15_snapshot_store_and_resume(tmp_path):
    env = ScriptedEnv(latency=lambda c, k: 1.0 + (c * 5 + k * 3) % 11, fails=lambda c, k: (c * 7 + k) % 5 == 0)
    protocol, calls, store = fake_protocol(tmp_path, env, iterations=20)
    for _ in range(10):
        protocol.run_iteration()
    referenced = protocol.referenced_tags()
    store.collect(referenced)
    assert store.tags() == referenced
    restored = pickle.loads(pickle.dumps(protocol))
    assert restored.train_fn is None and restored.store is None and restored.env is None
    restored.store, restored.train_fn, restored.env = store, protocol.train_fn, protocol.env
    a = [protocol.run_iteration()["aggregated"] for _ in range(10)]
    b = [restored.run_iteration()["aggregated"] for _ in range(10)]
    assert a == b and np.array_equal(protocol.x[0], restored.x[0]) and protocol.totals() == restored.totals()


# ---- Test 16: the full ResNet-50 state, all tensors -------------------------------------------------------------------

def test_16_real_model_all_tensors():
    import torch
    from collections import OrderedDict
    from models.model import FedMelanomaModel
    torch.manual_seed(0)
    model = FedMelanomaModel()
    base = [v.detach().cpu().numpy().copy() for v in model.state_dict().values()]
    rng = np.random.default_rng(0)
    models = [[a + rng.normal(0, 0.01, a.shape).astype(a.dtype) if a.dtype.kind == "f" else a + k for a in base] for k in (3, 9)]
    out, _ = async_w_fedavg(models, [6, 4], 7, [20, 30])              # staleness 0, 2 -> W 1, 1/3 -> W*D 20, 10
    assert len(out) == len(base)
    for i, got in enumerate(out):
        expected = (20 * models[0][i].astype(np.float64) + 10 * models[1][i].astype(np.float64)) / 30
        if got.dtype.kind == "f":
            assert np.allclose(got, expected, rtol=1e-6, atol=1e-7)
        else:
            assert int(got) == int(np.rint(expected))
    FedMelanomaModel().load_state_dict(OrderedDict((k, torch.tensor(v)) for k, v in zip(model.state_dict().keys(), out)), strict=True)
