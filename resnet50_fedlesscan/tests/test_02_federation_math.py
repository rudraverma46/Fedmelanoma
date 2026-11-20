"""Tests 11-15 and 20: FedLesScan selection, bookkeeping and aggregation.

Expected values are computed by hand or with plain numpy/sklearn, not with the code under test.
"""
import contextlib
import copy
import io
import pickle
import random

import numpy as np
import pytest
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy

import server.fedlesscan as fl
from monitoring.experiment_logging import ExperimentLogger
from server.strategy import FedMelanomaStrategy, HospitalConfig

SIZES = [564, 198, 200, 516, 1549, 1602, 1716, 223, 155, 1065]  # the real hospitals' training sizes


class FakeProxy(ClientProxy):
    def get_properties(self, ins, timeout, group_id): ...
    def get_parameters(self, ins, timeout, group_id): ...
    def fit(self, ins, timeout, group_id): ...
    def evaluate(self, ins, timeout, group_id): ...
    def reconnect(self, ins, timeout, group_id): ...


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


def make_strategy(tmp_path, sizes=SIZES, n_select=6, initial=None):
    configs = [HospitalConfig(n_samples=n, tier="reliable", latency_mu=1.39, latency_sigma=0.3) for n in sizes]
    initial = initial or [np.array([0.0, 1.0, 2.0], dtype=np.float32), np.array(5, dtype=np.int64)]
    strategy = FedMelanomaStrategy(hospital_configs=configs,
                                   n_select=n_select, initial_parameters=initial, rng=np.random.default_rng(0),
                                   experiment_logger=ExperimentLogger("unit", logs_root=str(tmp_path / "logs")), run_id="unit",
                                   checkpoint_dir=str(tmp_path / "ckpt"), results_dir=str(tmp_path / "res"), seed=42,
                                   max_rounds=50, round_deadline_seconds=30.0, allowed_stragglers=6, staleness_tolerance=2)
    manager = SimpleClientManager()
    proxies = {i: FakeProxy(f"{i:02d}") for i in range(len(sizes))}
    for proxy in proxies.values():
        manager.register(proxy)
    return strategy, manager, proxies, initial


def result(proxy, arrays, n, seconds=60.0):
    return proxy, FitRes(status=Status(code=Code.OK, message=""), parameters=ndarrays_to_parameters(arrays), num_examples=n,
                         metrics={"train_loss": 1.0, "fit_seconds": seconds})


# ---- Test 11: no fixed stragglers; every selected client trains -------------------------------

def force_env(strategy, outcome):
    """Replace the seeded trace by `outcome(client_id, job_index) -> (failed, latency_s)` for a deterministic test."""
    strategy._env.draw = lambda cid, k: outcome(cid, k)


def test_11_round_1_is_the_official_rookie_sample():
    f = fl.FedLesScan(range(10), 6, 50, 42, 6)
    assert f.select(0)[0] == sorted(random.Random(42).sample(list(range(10)), 6))   # no fixed-straggler draw before it
    assert not hasattr(f, "invocation_delay") and not hasattr(f, "fails")


def test_11_every_selected_client_is_sent_to_flower(tmp_path):
    strategy, manager, _, initial = make_strategy(tmp_path)
    with quiet():
        instructions = strategy.configure_fit(1, ndarrays_to_parameters(initial), manager)
    sent = {int(p.cid) for p, _ in instructions}
    assert len(strategy._last_round_selected_ids) == 6
    assert sent == set(strategy._last_round_selected_ids) == set(strategy._last_round_client_ids)
    assert strategy._last_round_failed_ids == []


# ---- Test 12: behaviour features, by hand ---------------------------------------------------

def test_12_training_time_ema_by_hand():
    assert fl.get_client_ema([]) == 0
    assert fl.get_client_ema([4.0]) == 4.0
    assert fl.get_client_ema([4.0, 2.0, 6.0]) == pytest.approx(((4.0 * 0.5 + 2.0 * 0.5) * 0.5) + 6.0 * 0.5)


def test_12_missed_round_penalty_by_hand():
    # missed rounds [1, 3] at round 5, max time 10: ratios (2/5, 4/5); EMA(0.8) = 0.4*0.2 + 0.8*0.8 = 0.72
    assert fl.get_missed_rounds_ema([1, 3], 5, 10.0) == pytest.approx(0.72 * 1.5 * 10.0)
    assert fl.get_missed_rounds_ema([], 5, 10.0) == 0


def test_12_cooldown_sequence_and_reset():
    f = fl.FedLesScan(range(10), 6, 50, 42, 6)
    for r, want in ((0, 1), (1, 2), (2, 4), (3, 8)):
        f.finish_round(r, [0], [])
        assert f.histories[0].client_backoff == want
    assert f.histories[0].missed_rounds == [0, 1, 2, 3]
    f.finish_round(4, [0], [0])
    assert f.histories[0].client_backoff == 0


def test_12_tiers():
    f = fl.FedLesScan(range(4), 2, 50, 42, 2)
    f.histories[1].training_times = [1.0]
    f.histories[2].missed_rounds, f.histories[2].client_backoff = [3], 2  # cooldown until round 5
    f.histories[3].missed_rounds, f.histories[3].client_backoff = [1], 1  # cooldown ended at round 2
    rookies, rest, stragglers = fl.filter_rookies([f.histories[c] for c in range(4)], 5)
    assert [c.client_id for c in rookies] == [0]
    assert [c.client_id for c in stragglers] == [2]
    assert [c.client_id for c in rest] == [1, 3]


def test_12_clustering_matches_independent_grid_search():
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import calinski_harabasz_score
    from sklearn.preprocessing import StandardScaler
    data = [[1.0, 0.0], [1.1, 0.0], [5.0, 0.0], [5.2, 0.0], [9.0, 3.0], [1.05, 0.0]]
    X = StandardScaler().fit_transform(data)
    best = (0, None)
    for eps in np.arange(0.01, 2, 0.1):
        labels = DBSCAN(eps=eps, min_samples=2).fit(X).labels_
        if 1 < len(set(labels)) <= len(X) - 1:
            score = calinski_harabasz_score(X, labels)
            if score > best[0]:
                best = (score, labels)
    labels, score, _ = fl.perform_clustering(data)
    assert score == pytest.approx(best[0]) and list(labels) == list(best[1])


def test_12_clusters_sorted_fastest_first_and_progress_start():
    f = fl.FedLesScan(range(6), 2, 10, 42, 6)
    for c, t in enumerate([1.0, 1.1, 1.05, 5.0, 5.2, 5.1]):
        f.histories[c].training_times = [t]
    selected, info = f.select(1)
    clusters = info["clustering"]["clusters"]
    assert [c["cluster_ema"] for c in clusters] == sorted(c["cluster_ema"] for c in clusters)
    assert set(clusters[0]["clients"]) == {0, 1, 2}
    # progress (1 - 1) / (10 - 1) = 0 -> start at the fastest cluster; the 2 least-used clients are taken
    assert info["clustering"]["start_cluster_idx"] == 0 and set(selected) <= {0, 1, 2}


# ---- Test 13: staleness-aware aggregation, by hand ------------------------------------------

def test_13_stall_aware_formula_by_hand_without_renormalisation():
    f = fl.FedLesScan(range(3), 3, 50, 42, 3)
    a, b, c = np.array([1.0, 2.0], np.float32), np.array([3.0, 5.0], np.float32), np.array([7.0, 7.0], np.float32)
    f.store_result(fl.ClientResult(0, 4, 100, [a]))   # this round (t = 4): score 1
    f.store_result(fl.ClientResult(1, 3, 300, [b]))   # one round late: score 4/5
    f.store_result(fl.ClientResult(2, 1, 600, [c]))   # three rounds late: beyond tolerance 2, discarded
    out, record = f.aggregate(4)
    expected = (100 * 1.0 * a + 300 * 0.8 * b) / 400
    assert np.allclose(out[0], expected)
    assert [r["client_id"] for r in record] == [0, 1]
    assert sum(r["effective_weight"] for r in record) == pytest.approx((100 + 240) / 400)  # < 1: not renormalised
    assert f.results == []  # every result is deleted after aggregation (PER_SESSION)


def test_13_tolerance_boundary():
    f = fl.FedLesScan(range(2), 2, 50, 42, 2)
    f.store_result(fl.ClientResult(0, 8, 10, [np.ones(1)]))
    f.store_result(fl.ClientResult(1, 7, 10, [np.ones(1)]))
    _, record = f.aggregate(10)
    assert [r["client_id"] for r in record] == [0]  # round_id >= 10 - 2


@pytest.mark.parametrize("staleness,accepted", [(0, True), (1, True), (2, True), (3, False)])
def test_13_staleness_boundary_official_rule(staleness, accepted):
    """Official code: round_id >= t - tolerance, i.e. staleness <= 2 is accepted, 3 is discarded."""
    f = fl.FedLesScan(range(2), 2, 50, 42, 2)
    t = 10
    f.store_result(fl.ClientResult(0, t - staleness, 10, [np.ones(1)]))
    out, record = f.aggregate(t)
    assert (out is not None) == accepted
    assert [r["client_id"] for r in record] == ([0] if accepted else [])


@pytest.mark.parametrize("t_k,t", [(0, 0), (0, 1), (3, 5), (7, 9), (47, 49)])
def test_13_zero_based_weight_is_tk_plus_1_over_t_plus_1(t_k, t):
    f = fl.FedLesScan(range(1), 1, 50, 42, 1)
    f.store_result(fl.ClientResult(0, t_k, 100, [np.array([2.0])]))
    out, record = f.aggregate(t)
    if t - t_k > 2:
        assert out is None
        return
    score = (t_k + 1) / (t + 1)
    assert record[0]["score"] == pytest.approx(score)
    assert np.allclose(out[0], 2.0 * score)  # 100 * score * 2.0 / 100: official, not renormalised


def test_13_discarded_update_does_not_change_the_global_model():
    a, c = np.array([1.0, 2.0]), np.array([100.0, -50.0])
    with_stale, without = fl.FedLesScan(range(2), 2, 50, 42, 2), fl.FedLesScan(range(2), 2, 50, 42, 2)
    for f in (with_stale, without):
        f.store_result(fl.ClientResult(0, 9, 200, [a]))
    with_stale.store_result(fl.ClientResult(1, 6, 900, [c]))  # staleness 3 at t = 9: discarded
    out1, rec1 = with_stale.aggregate(9)
    out2, rec2 = without.aggregate(9)
    assert np.array_equal(out1[0], out2[0]) and [r["client_id"] for r in rec1] == [r["client_id"] for r in rec2] == [0]
    assert with_stale.results == []  # and it is deleted, not kept for later


def test_13_no_candidates_keeps_the_model():
    f = fl.FedLesScan(range(1), 1, 50, 42, 1)
    assert f.aggregate(5) == (None, [])


def test_13_abort_rule():
    strict = fl.FedLesScan(range(3), 3, 50, 42, 0)
    with pytest.raises(RuntimeError):
        strict.finish_round(0, [0, 1, 2], [0, 1])
    fl.FedLesScan(range(3), 3, 50, 42, 6).finish_round(0, [0, 1, 2], [])  # allowed 6: never aborts


# ---- Test 14 and 15: the strategy round, end to end with fake clients ---------------------

def run_round(strategy, manager, proxies, parameters, t, rng):
    with quiet():
        strategy.configure_fit(t, parameters, manager)
    trained = strategy._last_round_client_ids
    arrays = {i: [rng.normal(size=3).astype(np.float32), np.array(10 + i, dtype=np.int64)] for i in trained}
    with quiet():
        new, metrics = strategy.aggregate_fit(t, [result(proxies[i], arrays[i], SIZES[i], 30.0 + i) for i in trained], [])
    return new, metrics, trained, arrays


def test_14_global_update_equals_sample_weighted_average_of_on_time_clients(tmp_path):
    strategy, manager, proxies, initial = make_strategy(tmp_path)
    force_env(strategy, lambda cid, k: (False, 10.0 + cid))   # all on time (< 30 s)
    rng = np.random.default_rng(2)
    parameters = ndarrays_to_parameters(initial)
    for t in range(1, 6):
        parameters, metrics, trained, arrays = run_round(strategy, manager, proxies, parameters, t, rng)
        out = parameters_to_ndarrays(parameters)
        total = sum(SIZES[i] for i in trained)
        assert np.allclose(out[0], sum(SIZES[i] / total * arrays[i][0].astype(np.float64) for i in trained), atol=1e-5)
        assert metrics["fedmelanoma_clients_aggregated"] == len(trained)
        assert strategy._pending_client_ndarrays == {} and strategy._fedlesscan.results == []
        for i in trained:
            assert strategy._fedlesscan.histories[i].training_times[-1] == pytest.approx((10.0 + i) / 60.0)


def test_14_failed_late_and_delayed_aggregation(tmp_path):
    """Failure: trained, then discarded, round missed. Late (45 s, deadline 30 s): arrives one round later and is
    aggregated asynchronously the next time aggregation runs, with score (t_k+1)/(t+1) = 1/2 (zero-based t_k = 0, t = 1)."""
    strategy, manager, proxies, initial = make_strategy(tmp_path)
    rng = np.random.default_rng(3)
    parameters = ndarrays_to_parameters(initial)
    roles = {}

    def outcome(cid, k):
        if roles.get(cid) == "fail" and k == 0:
            return True, float("inf")
        if roles.get(cid) == "late" and k == 0:
            return False, 45.0
        return False, 5.0

    force_env(strategy, outcome)
    with quiet():
        strategy.configure_fit(1, parameters, manager)
    sel = strategy._last_round_selected_ids
    roles[sel[0]], roles[sel[1]] = "fail", "late"
    failed, late = sel[0], sel[1]
    arrays = {i: [rng.normal(size=3).astype(np.float32), np.array(10 + i, dtype=np.int64)] for i in sel}
    with quiet():
        parameters, _ = strategy.aggregate_fit(1, [result(proxies[i], arrays[i], SIZES[i]) for i in sel], [])
    plan = strategy._round_plan[1]
    assert plan["failed"] == [failed] and plan["late"] == [late]
    assert set(plan["on_time"]) == set(sel) - {failed, late}
    assert [c for c, _ in plan["aggregated"]] == plan["on_time"]            # neither the failed nor the late update yet
    f = strategy._fedlesscan
    assert 0 in f.histories[failed].missed_rounds and 0 in f.histories[late].missed_rounds
    assert [(j["client_id"], j["arrival_round"]) for j in f.in_flight] == [(late, 1)]
    out1 = parameters_to_ndarrays(parameters)
    on_time = plan["on_time"]
    total = sum(SIZES[i] for i in on_time)
    assert np.allclose(out1[0], sum(SIZES[i] / total * arrays[i][0].astype(np.float64) for i in on_time), atol=1e-5)

    # round 2: the late update arrives and is aggregated now, with its own round_id
    arrays2 = {}
    with quiet():
        strategy.configure_fit(2, parameters, manager)
    sel2 = strategy._last_round_selected_ids
    arrays2 = {i: [rng.normal(size=3).astype(np.float32), np.array(10 + i, dtype=np.int64)] for i in sel2}
    with quiet():
        parameters2, _ = strategy.aggregate_fit(2, [result(proxies[i], arrays2[i], SIZES[i]) for i in sel2], [])
    plan2 = strategy._round_plan[2]
    assert [late, 1] in plan2["arrived_late"]                               # (client, 1-based round of its training)
    record = [e for e in plan2["aggregated"] if e[0] == late and e[1] == 1]
    assert record, plan2["aggregated"]
    assert 0 not in f.histories[late].missed_rounds                          # finishing clears the missed round
    out2 = parameters_to_ndarrays(parameters2)
    fresh = [i for i in sel2 if i in plan2["on_time"]]
    num = sum(SIZES[i] * arrays2[i][0].astype(np.float64) for i in fresh) + SIZES[late] * 0.5 * arrays[late][0].astype(np.float64)
    den = sum(SIZES[i] for i in fresh) + SIZES[late]
    assert np.allclose(out2[0], num / den, atol=1e-5)                        # official formula, no renormalisation


def test_15_evaluation_uses_the_clients_selected_this_round(tmp_path):
    strategy, manager, _, initial = make_strategy(tmp_path)
    for t in range(1, 4):
        with quiet():
            strategy.configure_fit(t, ndarrays_to_parameters(initial), manager)
            instructions = strategy.configure_evaluate(t, ndarrays_to_parameters(initial), manager)
        assert sorted(int(p.cid) for p, _ in instructions) == sorted(strategy._last_round_selected_ids)
        # keep the histories moving so the next selection differs from round 1
        strategy._fedlesscan.finish_round(t - 1, strategy._last_round_selected_ids, strategy._last_round_selected_ids)
        for c in strategy._last_round_selected_ids:
            strategy._fedlesscan.record_client_success(c, t - 1, 0.1 + 0.01 * c, SIZES[c])


def test_15_resume_state_continues_identically(tmp_path):
    strategy, manager, proxies, initial = make_strategy(tmp_path)
    force_env(strategy, lambda cid, k: (False, 5.0 + 20.0 * (cid % 3 == 2)))
    rng = np.random.default_rng(5)
    parameters = ndarrays_to_parameters(initial)
    for t in range(1, 4):
        parameters, *_ = run_round(strategy, manager, proxies, parameters, t, rng)
    saved = pickle.loads(pickle.dumps(strategy._fedlesscan))
    a = [strategy._fedlesscan.select(r)[0] for r in range(3, 8)]
    b = [saved.select(r)[0] for r in range(3, 8)]
    assert a == b


# ---- Test 20: the real model, all 329 tensors ------------------------------------------------

def test_20_real_model_all_329_tensors(tmp_path):
    import torch
    from collections import OrderedDict
    from models.model import FedMelanomaModel

    torch.manual_seed(0)
    model = FedMelanomaModel()
    keys = list(model.state_dict().keys())
    initial = [v.detach().cpu().numpy().copy() for v in model.state_dict().values()]
    rng = np.random.default_rng(0)

    def perturbed():
        return [a + rng.normal(0, 0.01, a.shape).astype(a.dtype) if a.dtype.kind == "f" else a + rng.integers(1, 50) for a in initial]

    strategy, manager, proxies, _ = make_strategy(tmp_path, initial=initial)
    force_env(strategy, lambda cid, k: (False, 5.0))
    with quiet():
        strategy.configure_fit(1, ndarrays_to_parameters(initial), manager)
    trained = strategy._last_round_selected_ids
    updates = {i: perturbed() for i in trained}
    with quiet():
        parameters, _ = strategy.aggregate_fit(1, [result(proxies[i], updates[i], SIZES[i]) for i in trained], [])
    out = parameters_to_ndarrays(parameters)
    assert len(out) == len(keys) == 329
    total = sum(SIZES[i] for i in trained)
    for index, (k, got) in enumerate(zip(keys, out)):
        expected = sum(SIZES[i] / total * updates[i][index].astype(np.float64) for i in trained)
        assert np.allclose(got, expected, rtol=1e-5, atol=2e-6), k
    FedMelanomaModel().load_state_dict(OrderedDict((k, torch.tensor(v)) for k, v in zip(keys, out)), strict=True)
