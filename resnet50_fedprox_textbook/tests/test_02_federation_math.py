"""Tests 11-15 and 20: uniform dropout and sample-weighted averaging. Expected values use plain numpy, not the code under test."""
import contextlib
import io
import itertools
from collections import Counter

import numpy as np
import pytest
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy

from monitoring.experiment_logging import ExperimentLogger
from server.aggregation import fedavg_baseline_weights
from server.selector import Client
from server.straggler import ClientResponse
from server.strategy import FedMelanomaStrategy, HospitalConfig


class FakeProxy(ClientProxy):
    def get_properties(self, ins, timeout, group_id): ...
    def get_parameters(self, ins, timeout, group_id): ...
    def fit(self, ins, timeout, group_id): ...
    def evaluate(self, ins, timeout, group_id): ...
    def reconnect(self, ins, timeout, group_id): ...


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


def make_strategy(tmp_path, n_select, sizes, rng=None, skew=0.0):
    configs = [HospitalConfig(n_samples=n, tier="reliable", latency_mu=1.39, latency_sigma=0.3, fitzpatrick_dist=[1 - skew * i / len(sizes), skew * i / len(sizes), 0, 0, 0, 0],
                              modality_dist=[1, 0, 0], confirmation_dist=[1, 0, 0, 0]) for i, n in enumerate(sizes)]
    initial = [np.array([0.0, 1.0, 2.0], dtype=np.float32), np.array(5, dtype=np.int64)]
    strategy = FedMelanomaStrategy(hospital_configs=configs, global_fitzpatrick=[0.6, 0.4, 0, 0, 0, 0], global_modality=[1, 0, 0], global_confirmation=[1, 0, 0, 0], n_select=n_select,
                                   initial_parameters=initial, rng=rng or np.random.default_rng(0), experiment_logger=ExperimentLogger("unit", logs_root=str(tmp_path / "logs")),
                                   run_id="unit", checkpoint_dir=str(tmp_path / "ckpt"), results_dir=str(tmp_path / "res"))
    manager = SimpleClientManager()
    proxies = {i: FakeProxy(f"{i:02d}") for i in range(len(sizes))}
    for proxy in proxies.values():
        manager.register(proxy)
    return strategy, manager, proxies, initial


def result(proxy, arrays, n):
    return proxy, FitRes(status=Status(code=Code.OK, message=""), parameters=ndarrays_to_parameters(arrays), num_examples=n, metrics={"train_loss": 1.0})


def draw(strategy, manager, parameters, t):
    with quiet():
        return strategy.configure_fit(t, parameters, manager)


def test_11_four_dropped_six_train_every_round(tmp_path):
    strategy, manager, _, initial = make_strategy(tmp_path, 6, [100] * 10)
    parameters = ndarrays_to_parameters(initial)
    for t in range(1, 21):
        instructions = draw(strategy, manager, parameters, t)
        assert len(instructions) == 6 and len(strategy._last_round_dropped_ids) == 4
        assert {int(p.cid) for p, _ in instructions} == set(strategy._last_round_client_ids)
        assert not set(strategy._last_round_client_ids) & set(strategy._last_round_dropped_ids)


def test_12_every_client_drops_with_equal_probability():
    rng = np.random.default_rng(7)
    n = 200000
    counts = Counter(int(c) for _ in range(n) for c in rng.choice(list(range(10)), size=4, replace=False))
    assert all(abs(counts[c] / n - 0.4) < 0.005 for c in range(10))


def test_12_every_subset_of_four_is_equally_likely():
    rng = np.random.default_rng(8)
    n = 210000
    subsets = Counter(tuple(sorted(int(c) for c in rng.choice(list(range(10)), size=4, replace=False))) for _ in range(n))
    assert len(subsets) == len(list(itertools.combinations(range(10), 4))) == 210
    expected = n / 210
    assert sum((v - expected) ** 2 / expected for v in subsets.values()) < 280  # chi-square, 209 degrees of freedom, p about 0.001


def test_13_size_and_diversity_do_not_change_the_draw(tmp_path):
    runs = []
    for i, (sizes, skew) in enumerate((([100] * 10, 0.0), ([10, 5000, 7, 900, 3000, 20, 1, 40, 600, 2], 1.0))):
        strategy, manager, _, initial = make_strategy(tmp_path / str(i), 6, sizes, rng=np.random.default_rng(42), skew=skew)
        parameters = ndarrays_to_parameters(initial)
        drops = []
        for t in range(1, 51):
            draw(strategy, manager, parameters, t)
            drops.append(list(strategy._last_round_dropped_ids))
        runs.append(drops)
    assert runs[0] == runs[1]


def test_13_strategy_reproduces_planned_schedule(tmp_path, planned):
    sizes = [h["n_samples"] for h in planned["hospitals"]]
    strategy, manager, _, initial = make_strategy(tmp_path, 6, sizes, rng=np.random.default_rng(planned["seed"]))
    parameters = ndarrays_to_parameters(initial)
    assert len(planned["schedule"]) == 50
    for want in planned["schedule"]:
        draw(strategy, manager, parameters, want["round"])
        assert strategy._last_round_dropped_ids == want["dropped"] and strategy._last_round_client_ids == want["trained"]


def test_14_weights_by_hand():
    clients = {i: Client(client_id=i, n_samples=n, D=d) for i, (n, d) in enumerate(((100, 0.0), (300, 1.0), (600, 0.5)))}
    records = [ClientResponse(i, 0.0, False, False, 0.0, 1.0, 1) for i in clients]
    assert fedavg_baseline_weights(records, clients) == pytest.approx({0: 0.1, 1: 0.3, 2: 0.6})


def test_15_global_update_equals_textbook_average_and_dropped_clients_count_for_nothing(tmp_path):
    sizes = [100, 300, 600, 50]
    strategy, manager, proxies, initial = make_strategy(tmp_path, 3, sizes, rng=np.random.default_rng(1))
    draw(strategy, manager, ndarrays_to_parameters(initial), 1)
    trained, dropped = strategy._last_round_client_ids, strategy._last_round_dropped_ids
    assert len(trained) == 3 and len(dropped) == 1
    rng = np.random.default_rng(2)
    arrays = {i: [rng.normal(size=3).astype(np.float32), np.array(10 + i, dtype=np.int64)] for i in trained}
    with quiet():
        parameters, metrics = strategy.aggregate_fit(1, [result(proxies[i], arrays[i], sizes[i]) for i in trained], [])
    out = parameters_to_ndarrays(parameters)
    total = sum(sizes[i] for i in trained)
    expected = sum(sizes[i] / total * arrays[i][0].astype(np.float64) for i in trained)
    assert np.allclose(out[0], expected, atol=1e-5)
    assert metrics["fedmelanoma_clients_aggregated"] == 3 and metrics["fedmelanoma_new_stragglers_buffered"] == 1
    assert strategy._pending_client_ndarrays == {}


def test_15_client_state_is_never_updated(tmp_path):
    sizes = [100] * 10
    strategy, manager, proxies, initial = make_strategy(tmp_path, 6, sizes)
    draw(strategy, manager, ndarrays_to_parameters(initial), 1)
    with quiet():
        strategy.aggregate_fit(1, [result(proxies[i], initial, 100) for i in strategy._last_round_client_ids], [])
    assert all(c.R == 1.0 and c.N_eff == 0.0 for c in strategy._clients.values())


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

    sizes = [155, 1716, 1065]
    strategy, manager, proxies, _ = make_strategy(tmp_path, 2, sizes, rng=np.random.default_rng(3))
    draw(strategy, manager, ndarrays_to_parameters(initial), 1)
    trained = strategy._last_round_client_ids
    updates = {i: perturbed() for i in trained}
    with quiet():
        parameters, _ = strategy.aggregate_fit(1, [result(proxies[i], updates[i], sizes[i]) for i in trained], [])
    out = parameters_to_ndarrays(parameters)
    assert len(out) == len(keys) == 329
    total = sum(sizes[i] for i in trained)
    for index, (k, got) in enumerate(zip(keys, out)):
        expected = sum(sizes[i] / total * updates[i][index].astype(np.float64) for i in trained)
        assert np.allclose(got, expected, rtol=1e-5, atol=2e-6), k
    FedMelanomaModel().load_state_dict(OrderedDict((k, torch.tensor(v)) for k, v in zip(keys, out)), strict=True)
