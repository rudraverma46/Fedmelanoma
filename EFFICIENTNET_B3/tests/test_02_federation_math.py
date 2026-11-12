"""Tests 11-15: selection, aggregation, straggler simulation, deadline, late updates.

Expected values are computed here with plain numpy from the formulas in the plan, not by calling
the module under test. test_13 replays all 50 rounds from scratch for the same reason.
"""
import contextlib
import io
import math

import numpy as np
import pytest
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy

from hospital_registry import build_hospital_registry
from monitoring.experiment_logging import ExperimentLogger
from server.aggregation import compute_aggregation_weights
from server.selector import Client, Selector
from server.straggler import BETA_PARAMS, BufferedUpdate, ClientResponse, StragglerManager
from server.strategy import FedMelanomaStrategy, HospitalConfig, compute_normalized_diversity_priors


class FakeProxy(ClientProxy):
    def get_properties(self, ins, timeout, group_id): ...
    def get_parameters(self, ins, timeout, group_id): ...
    def fit(self, ins, timeout, group_id): ...
    def evaluate(self, ins, timeout, group_id): ...
    def reconnect(self, ins, timeout, group_id): ...


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


def make_strategy(tmp_path, n_select, sizes=(100, 300, 600), rng=None):
    configs = [
        HospitalConfig(n_samples=n, tier="reliable", latency_mu=1.39, latency_sigma=0.3,
                       fitzpatrick_dist=[1 - i * 0.3, i * 0.3, 0, 0, 0, 0], modality_dist=[1 - i * 0.2, i * 0.2, 0],
                       confirmation_dist=[1 - i * 0.1, i * 0.1, 0, 0])
        for i, n in enumerate(sizes)
    ]
    initial = [np.array([0.0, 1.0, 2.0], dtype=np.float32), np.array(5, dtype=np.int64)]
    run_id = "unit"
    strategy = FedMelanomaStrategy(
        hospital_configs=configs, global_fitzpatrick=[0.6, 0.4, 0, 0, 0, 0], global_modality=[0.8, 0.2, 0],
        global_confirmation=[0.9, 0.1, 0, 0], n_select=n_select, initial_parameters=initial,
        rng=rng or np.random.default_rng(0), experiment_logger=ExperimentLogger(run_id, logs_root=str(tmp_path / "logs")),
        run_id=run_id, checkpoint_dir=str(tmp_path / "ckpt"), results_dir=str(tmp_path / "res"))
    manager = SimpleClientManager()
    proxies = {}
    for i in range(len(sizes)):
        proxies[i] = FakeProxy(f"{i:02d}")
        manager.register(proxies[i])
    return strategy, manager, proxies, initial


def scripted(outcomes):
    """Replace simulate_response with fixed outcomes: {client_id: 'on_time'|'late'|'failed'}."""
    def simulate(client, current_round):
        kind = outcomes[client.client_id]
        late, failed = kind == "late", kind == "failed"
        return ClientResponse(client.client_id, math.inf if failed else 1.0, late, failed,
                              math.inf if failed or late else 0.0, 0.0 if failed or late else 1.0, current_round)
    return simulate


def result(proxy, arrays, n):
    return proxy, FitRes(status=Status(code=Code.OK, message=""), parameters=ndarrays_to_parameters(arrays),
                         num_examples=n, metrics={"train_loss": 1.0})


# ------------------------------------------------------------------ test 11
def test_11_selection_score_and_probabilities_by_hand():
    selector = Selector()
    client = Client(client_id=0, n_samples=10, R=0.8, D=0.6, N_eff=4.0)
    expected = 0.8 * (1 + 1.0 * 0.6 / (1 + 0.05 * 4.0))
    assert selector.compute_score(client) == pytest.approx(expected)

    clients = [Client(client_id=i, n_samples=10, R=r, D=d, N_eff=n)
               for i, (r, d, n) in enumerate([(1.0, 1.0, 0), (0.9, 0.2, 3), (0.5, 0.9, 1), (0.2, 0.5, 8)])]
    scores = np.array([c.R * (1 + c.D / (1 + 0.05 * c.N_eff)) for c in clients])
    p = np.exp(scores / 0.2) / np.exp(scores / 0.2).sum()
    assert np.allclose(selector.boltzmann_probs(clients), p)

    rng = np.random.default_rng(7)
    first = np.zeros(4)
    with quiet():
        for _ in range(20000):
            picked = selector.boltzmann_selection(clients, 3, rng=rng)
            ids = [c.client_id for c in picked]
            assert len(set(ids)) == 3
            first[ids[0]] += 1
    assert np.allclose(first / 20000, p, atol=0.012)


def test_11_state_update_by_hand():
    selector = Selector()
    client = Client(client_id=0, n_samples=1, R=0.5, N_eff=2.0)
    selector.update_client_state(client, 1.0, 1)
    assert client.R == pytest.approx(0.85 * 0.5 + 0.15 * 0.9) and client.N_eff == pytest.approx(2.9)
    selector.update_client_state(client, math.inf, 2)
    assert client.R == pytest.approx(0.85 * (0.85 * 0.5 + 0.15 * 0.9)) and client.N_eff == pytest.approx(2.9)


def test_11_buffered_clients_excluded_from_selection(tmp_path):
    strategy, manager, proxies, initial = make_strategy(tmp_path, n_select=3)
    params = ndarrays_to_parameters(initial)
    with quiet():
        strategy.configure_fit(1, params, manager)
        strategy._straggler_manager.simulate_response = scripted({0: "late", 1: "on_time", 2: "on_time"})
        strategy.aggregate_fit(1, [result(proxies[i], initial, 1) for i in range(3)], [])
        instructions = strategy.configure_fit(2, params, manager)
    assert 0 not in strategy._last_round_client_ids and len(instructions) == 2


def test_11_diversity_prior_real_registry(cfg, planned):
    """D_i recomputed by a separate 1-D and LP calculation on the real distributions."""
    from scipy.optimize import linprog

    _, records, g_fitz, g_mod, g_conf = build_hospital_registry(cfg["data"]["partitions_root"])

    def transport(p, q, cost):
        n = len(p)
        rows = [np.kron(np.eye(n)[i], np.ones(n)) for i in range(n)] + [np.kron(np.ones(n), np.eye(n)[j]) for j in range(n)]
        return linprog(cost.flatten(), A_eq=np.array(rows), b_eq=np.r_[p, q], bounds=(0, None), method="highs").fun

    cost_mod = np.array([[0, .3, .7], [.3, 0, .45], [.7, .45, 0]])
    cost_conf = np.abs(np.subtract.outer(np.arange(4), np.arange(4))) / 3
    raw = []
    for r in records:
        fitz = np.abs(np.cumsum(g_fitz) - np.cumsum(r["fitzpatrick_dist"])).sum()
        raw.append(0.40 * fitz + 0.35 * transport(g_mod, r["modality_dist"], cost_mod)
                   + 0.25 * transport(g_conf, r["confirmation_dist"], cost_conf))
    expected = np.array(raw) / max(raw)
    configs = [HospitalConfig(**r) for r in records]
    assert np.allclose(compute_normalized_diversity_priors(configs, g_fitz, g_mod, g_conf), expected, atol=1e-6)
    assert np.allclose([h["D"] for h in planned["hospitals"]], expected, atol=1e-6)


# ------------------------------------------------------------------ test 12
def test_12_weights_by_hand():
    clients = {0: Client(client_id=0, n_samples=100, D=0.5), 1: Client(client_id=1, n_samples=300, D=0.0), 2: Client(client_id=2, n_samples=600, D=1.0)}
    responses = [ClientResponse(0, 1, False, False, 0.0, 1, 1), ClientResponse(1, 1, False, False, 1.0, .9, 0), ClientResponse(2, 1, False, False, 2.0, .81, 0)]
    raw = np.array([0.1 * 1.0 * 1.5, 0.3 * 0.9 * 1.0, 0.6 * 0.81 * 2.0])
    expected = raw / raw.sum()
    got = compute_aggregation_weights(responses, clients)
    assert np.allclose([got[0], got[1], got[2]], expected)
    assert sum(got.values()) == pytest.approx(1.0)


def test_12_global_update_matches_independent_calculation(tmp_path):
    strategy, manager, proxies, initial = make_strategy(tmp_path, n_select=3)
    client_arrays = {
        0: [np.array([1.0, 1.0, 1.0], dtype=np.float32), np.array(9, dtype=np.int64)],
        1: [np.array([2.0, 0.0, 4.0], dtype=np.float32), np.array(13, dtype=np.int64)],
        2: [np.array([-1.0, 3.0, 2.5], dtype=np.float32), np.array(21, dtype=np.int64)],
    }
    with quiet():
        strategy.configure_fit(1, ndarrays_to_parameters(initial), manager)
        strategy._straggler_manager.simulate_response = scripted({0: "on_time", 1: "on_time", 2: "on_time"})
        parameters, _ = strategy.aggregate_fit(1, [result(proxies[i], client_arrays[i], 1) for i in range(3)], [])
    out = parameters_to_ndarrays(parameters)
    D = [strategy._clients[i].D for i in range(3)]
    n = np.array([100.0, 300.0, 600.0])
    raw = n / n.sum() * (1 + np.array(D))
    c = raw / raw.sum()
    expected = initial[0].astype(np.float64) + sum(c[i] * (client_arrays[i][0].astype(np.float64) - initial[0]) for i in range(3))
    assert np.allclose(out[0], expected, atol=1e-5)
    assert out[0].dtype == np.float32
    assert float(out[1]) == pytest.approx(5 + sum(c[i] * (int(client_arrays[i][1]) - 5) for i in range(3)))
    assert out[1].dtype == np.float64  # the reference's behaviour for integer buffers


def test_12_empty_round_leaves_global_model_unchanged(tmp_path):
    strategy, manager, proxies, initial = make_strategy(tmp_path, n_select=3)
    other = [np.array([9.0, 9.0, 9.0], dtype=np.float32), np.array(50, dtype=np.int64)]
    with quiet():
        strategy.configure_fit(1, ndarrays_to_parameters(initial), manager)
        strategy._straggler_manager.simulate_response = scripted({0: "failed", 1: "failed", 2: "late"})
        parameters, metrics = strategy.aggregate_fit(1, [result(proxies[i], other, 1) for i in range(3)], [])
    out = parameters_to_ndarrays(parameters)
    assert np.array_equal(out[0], initial[0]) and int(out[1]) == 5
    assert metrics["fedmelanoma_clients_aggregated"] == 0
    assert strategy._clients[0].N_eff == 0 and strategy._clients[0].R == pytest.approx(0.85)
    assert strategy._clients[2].N_eff == pytest.approx(0.9)


# ------------------------------------------------------------------ test 13
def test_13_tier_distributions():
    rng = np.random.default_rng(3)
    manager = StragglerManager(rng=rng)
    manager.current_W_t = 1e9
    specs = {"reliable": (1.39, 0.30), "moderate": (2.77, 0.50), "high_straggler": (4.09, 0.80)}
    for tier, (mu, sigma) in specs.items():
        client = Client(client_id=0, n_samples=1, tier=tier, latency_mu=mu, latency_sigma=sigma)
        draws = [manager.simulate_response(client, 1) for _ in range(30000)]
        a, b = BETA_PARAMS[tier]
        assert np.mean([not d.is_failed for d in draws]) == pytest.approx(a / (a + b), abs=0.012)
        logs = np.log([d.response_time for d in draws if not d.is_failed])
        assert logs.mean() == pytest.approx(mu, abs=0.02)
        assert logs.std() == pytest.approx(sigma, abs=0.02)
        assert all(math.isinf(d.response_time) and math.isinf(d.staleness) and d.delta_weight == 0 for d in draws if d.is_failed)


def independent_replay(hospitals, seed=42, rounds=50):
    """The method written from the plan's equations, without importing any reference module."""
    rng = np.random.default_rng(seed)
    beta = {"reliable": (19, 1), "moderate": (7, 3), "high_straggler": (2, 3)}
    lat = {"reliable": (1.39, .30), "moderate": (2.77, .50), "high_straggler": (4.09, .80)}
    R = {h["client_id"]: 1.0 for h in hospitals}
    N = {h["client_id"]: 0.0 for h in hospitals}
    D = {h["client_id"]: h["D"] for h in hospitals}
    tier = {h["client_id"]: h["tier"] for h in hospitals}
    size = {h["client_id"]: h["n_samples"] for h in hospitals}
    history, W, buffered = [], 2.0, {}
    out = []
    for t in range(1, rounds + 1):
        pool = [c for c in R if c not in buffered]
        take = min(6, len(pool))
        selected = []
        while len(selected) < take:
            s = np.clip([R[c] * (1 + D[c] / (1 + 0.05 * N[c])) for c in pool], 0, 2)
            e = np.exp(s / 0.2 - (s / 0.2).max())
            k = int(rng.choice(len(pool), p=e / e.sum()))
            selected.append(pool.pop(k))
        W_used = W
        stal, times = {}, {}
        for c in selected:
            pi = rng.beta(*beta[tier[c]])
            if rng.random() > pi:
                stal[c] = math.inf
                continue
            times[c] = rng.lognormal(mean=lat[tier[c]][0], sigma=lat[tier[c]][1])
            stal[c] = 0.0 if times[c] <= W_used else 1.0
        history = (history + list(times.values()))[-10:]
        if len(history) >= 2:
            med = float(np.median(history))
            W = max(med + 2.5 * float(np.median(np.abs(np.array(history) - med))), 2.0)
        arrived = {c: 0.0 for c in selected if stal[c] == 0.0}
        drained = []
        for c in list(buffered):
            del buffered[c]
            arrived[c] = 1.0
            drained.append(c)
        for c in selected:
            if stal[c] == 1.0:
                buffered[c] = t
        n_a = sum(size[c] for c in arrived)
        raw = {c: size[c] / n_a * 0.9 ** s * (1 + D[c]) for c, s in arrived.items()} if arrived else {}
        total = sum(raw.values())
        for c in selected:
            q = 0.0 if math.isinf(stal[c]) else 0.9 ** stal[c]
            R[c] = 0.85 * R[c] + 0.15 * q
            N[c] += q
        out.append({"round": t, "selected": selected, "window": W_used,
                    "failed": [c for c in selected if math.isinf(stal[c])],
                    "late": [c for c in selected if stal[c] == 1.0], "drained": drained,
                    "weights": {c: v / total for c, v in raw.items()}})
    return out


def test_13_full_schedule_matches_independent_replay(planned):
    replay = independent_replay(planned["hospitals"])
    assert len(planned["schedule"]) == 50
    for mine, ref in zip(replay, planned["schedule"]):
        assert mine["selected"] == ref["selected"], f"round {mine['round']}"
        assert mine["failed"] == ref["failed"]
        assert mine["late"] == ref["new_stragglers"]
        assert sorted(mine["drained"]) == sorted(ref["drained"])
        assert mine["window"] == pytest.approx(ref["window_used_for_classification"])
        assert set(mine["weights"]) == {int(k) for k in ref["weights"]}
        for cid, value in ref["weights"].items():
            assert mine["weights"][int(cid)] == pytest.approx(value)


def test_13_strategy_reproduces_planned_schedule(tmp_path, cfg, planned):
    """The real FedMelanomaStrategy, driven for 50 rounds with tiny fake models."""
    names, records, g_fitz, g_mod, g_conf = build_hospital_registry(cfg["data"]["partitions_root"])
    initial = [np.zeros(4, dtype=np.float32)]
    strategy = FedMelanomaStrategy(
        hospital_configs=[HospitalConfig(**r) for r in records], global_fitzpatrick=g_fitz, global_modality=g_mod,
        global_confirmation=g_conf, n_select=6, initial_parameters=initial, local_epochs=5, rng=np.random.default_rng(42),
        experiment_logger=ExperimentLogger("sched", logs_root=str(tmp_path / "logs")), run_id="sched",
        checkpoint_dir=str(tmp_path / "ckpt"), results_dir=str(tmp_path / "res"))
    manager = SimpleClientManager()
    proxies = {}
    for i in range(10):
        proxies[i] = FakeProxy(f"{i:02d}")
        manager.register(proxies[i])
    parameters = ndarrays_to_parameters(initial)
    with quiet():
        for row in planned["schedule"]:
            t = row["round"]
            instructions = strategy.configure_fit(t, parameters, manager)
            selected = list(strategy._last_round_client_ids)
            assert selected == row["selected"], f"round {t}"
            assert [p.cid for p, _ in instructions] == [f"{c:02d}" for c in selected]
            arrays = [np.full(4, float(t), dtype=np.float32)]
            parameters, metrics = strategy.aggregate_fit(t, [result(proxies[c], arrays, 1) for c in selected], [])
            assert metrics["fedmelanoma_clients_aggregated"] == len(row["aggregated"])
            assert strategy._straggler_manager.current_W_t == pytest.approx(row["window_after_round"])


# ------------------------------------------------------------------ test 14
def test_14_window_by_hand():
    manager = StragglerManager(rng=np.random.default_rng(0))
    assert manager.current_W_t == 2.0
    manager._latency_history.extend([3.0, 9.0, 4.0, 100.0, 5.0])
    manager._update_window()
    # sorted 3,4,5,9,100: median 5; deviations 2,4,1,95,0 -> sorted 0,1,2,4,95: MAD 2
    assert manager.current_W_t == pytest.approx(5 + 2.5 * 2)

    manager._latency_history.clear()
    manager._latency_history.extend([0.5, 0.6, 0.7])
    manager._update_window()
    assert manager.current_W_t == 2.0  # W_min floor

    manager._latency_history.clear()
    manager._latency_history.extend([50.0])
    manager._update_window()
    assert manager.current_W_t == 2.0  # fewer than two points

    manager._latency_history.extend(range(100, 130))
    assert len(manager._latency_history) == 10
    assert list(manager._latency_history)[0] == 120


def test_14_classification_uses_window_from_before_the_round():
    manager = StragglerManager(rng=np.random.default_rng(1))
    manager.current_W_t = 1000.0
    client = Client(client_id=0, n_samples=1, tier="reliable", latency_mu=4.0, latency_sigma=0.01)
    response = None
    while response is None or response.is_failed:
        response = manager.simulate_response(client, 1)
    assert not response.is_straggler
    manager.current_W_t = 1.0
    late = None
    while late is None or late.is_failed:
        late = manager.simulate_response(client, 2)
    assert late.is_straggler


def test_14_quorum_value_and_it_never_blocks(tmp_path):
    with quiet():
        manager = StragglerManager(rng=np.random.default_rng(0))
        responses = {i: ClientResponse(i, 1.0, i > 0, False, 0.0, 1.0, 1) for i in range(6)}
        clients = [Client(client_id=i, n_samples=1) for i in range(6)]
        on_time, late = manager.manage_window(clients, responses, 1, 6)
    assert len(on_time) == 1 and len(late) == 5
    assert manager.quorum_met_history == [False]  # 1 < ceil(0.70 * 6) = 5
    assert max(1, math.ceil(0.70 * 6)) == 5

    strategy, cm, proxies, initial = make_strategy(tmp_path, n_select=3)
    other = [np.array([3.0, 3.0, 3.0], dtype=np.float32), np.array(7, dtype=np.int64)]
    with quiet():
        strategy.configure_fit(1, ndarrays_to_parameters(initial), cm)
        strategy._straggler_manager.simulate_response = scripted({0: "on_time", 1: "failed", 2: "failed"})
        parameters, metrics = strategy.aggregate_fit(1, [result(proxies[i], other, 1) for i in range(3)], [])
    assert metrics["fedmelanoma_clients_aggregated"] == 1  # quorum (3) missed, aggregation still ran
    assert np.allclose(parameters_to_ndarrays(parameters)[0], other[0])  # single client: c = 1


# ------------------------------------------------------------------ test 15
def test_15_late_update_lifecycle(tmp_path):
    strategy, manager, proxies, initial = make_strategy(tmp_path, n_select=3)
    D = [strategy._clients[i].D for i in range(3)]
    late_arrays = [np.array([10.0, 10.0, 10.0], dtype=np.float32), np.array(8, dtype=np.int64)]
    fresh_0 = [np.array([1.0, 1.0, 1.0], dtype=np.float32), np.array(6, dtype=np.int64)]
    fresh_1 = [np.array([2.0, 2.0, 2.0], dtype=np.float32), np.array(6, dtype=np.int64)]
    with quiet():
        strategy.configure_fit(1, ndarrays_to_parameters(initial), manager)
        strategy._straggler_manager.simulate_response = scripted({0: "on_time", 1: "on_time", 2: "late"})
        parameters, _ = strategy.aggregate_fit(1, [result(proxies[0], fresh_0, 1), result(proxies[1], fresh_1, 1), result(proxies[2], late_arrays, 1)], [])
    w2 = parameters_to_ndarrays(parameters)

    # Round 1: only clients 0 and 1 count. Client 2 is credited s = 1 immediately.
    n = np.array([100.0, 300.0])
    raw = n / n.sum() * (1 + np.array(D[:2]))
    c = raw / raw.sum()
    assert np.allclose(w2[0], initial[0] + c[0] * (fresh_0[0] - initial[0]) + c[1] * (fresh_1[0] - initial[0]), atol=1e-5)
    assert 2 in strategy._straggler_manager.get_buffer_state()
    assert strategy._clients[2].R == pytest.approx(0.85 + 0.15 * 0.9)
    assert strategy._clients[2].N_eff == pytest.approx(0.9)

    # Round 2: client 2 cannot be selected; its stored weights drain with s = 1 relative to the
    # CURRENT global (Q3), and its R/N_eff are not updated a second time.
    r_before, n_before = strategy._clients[2].R, strategy._clients[2].N_eff
    round2 = [np.array([4.0, 4.0, 4.0], dtype=np.float32), np.array(9, dtype=np.int64)]
    with quiet():
        strategy.configure_fit(2, parameters, manager)
        assert sorted(strategy._last_round_client_ids) == [0, 1]
        strategy._straggler_manager.simulate_response = scripted({0: "on_time", 1: "failed"})
        parameters, metrics = strategy.aggregate_fit(2, [result(proxies[0], round2, 1), result(proxies[1], round2, 1)], [])
    w3 = parameters_to_ndarrays(parameters)
    n = np.array([100.0, 600.0])
    raw = n / n.sum() * np.array([1.0, 0.9]) * (1 + np.array([D[0], D[2]]))
    c = raw / raw.sum()
    expected = w2[0] + c[0] * (round2[0] - w2[0]) + c[1] * (late_arrays[0] - w2[0])
    assert np.allclose(w3[0], expected, atol=1e-5)
    assert metrics["fedmelanoma_clients_aggregated"] == 2
    assert strategy._straggler_manager.get_buffer_state() == {}
    assert 2 not in strategy._pending_client_ndarrays
    assert (strategy._clients[2].R, strategy._clients[2].N_eff) == (r_before, n_before)


def test_15_expiry_at_s_max():
    with quiet():
        manager = StragglerManager(rng=np.random.default_rng(0))
        response = ClientResponse(4, 9.0, True, False, 1.0, 0.9, 1)
        manager._buffer[4] = BufferedUpdate(4, response, 1)
        drained, expired = manager._drain_buffer(4)  # age 3 = S_max
    assert drained == [] and [e.client_id for e in expired] == [4] and manager._buffer == {}

    with quiet():
        manager._buffer[4] = BufferedUpdate(4, response, 1)
        drained, expired = manager._drain_buffer(3)  # age 2: still eligible, staleness recomputed
    assert expired == [] and drained[0].staleness == 2.0 and drained[0].delta_weight == pytest.approx(0.81)
