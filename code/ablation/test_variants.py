"""CPU checks that every ablation arm changes exactly its own factor.

    python -m pytest ablation/test_variants.py -q
The schedule test needs the hospital partitions (FEDMEL_DATA); the rest do not.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import server.strategy as strategy_module  # noqa: E402
from ablation.common import build_run_config, data_paths, load_registry, variant_of  # noqa: E402
from ablation.variants import (CommonRandomEnvironment, UniformSelector, _aggregation_function,  # noqa: E402
                               _selector, _straggler_manager, total_variation_diversity_priors)
from server.aggregation import compute_aggregation_weights  # noqa: E402
from server.selector import Client, Selector  # noqa: E402
from server.straggler import ClientResponse  # noqa: E402

TIERS = ("reliable", "moderate", "high_straggler")
LATENCY = {"reliable": (1.39, 0.30), "moderate": (2.77, 0.50), "high_straggler": (4.09, 0.80)}


def clients(n=10):
    return {i: Client(client_id=i, n_samples=100 * (i + 1), D=i / (n - 1), tier=TIERS[i % 3],
                      latency_mu=LATENCY[TIERS[i % 3]][0], latency_sigma=LATENCY[TIERS[i % 3]][1])
            for i in range(n)}


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def test_crn_draw_depends_only_on_seed_round_client():
    pool = clients()
    a = CommonRandomEnvironment(env_seed=42, rng=np.random.default_rng(0))
    b = CommonRandomEnvironment(env_seed=42, rng=np.random.default_rng(999))
    b.rng.random(1000)   # different shared-stream position must not matter
    for t in (1, 7, 30):
        for k in (0, 4, 9):
            ra, rb = a.simulate_response(pool[k], t), b.simulate_response(pool[k], t)
            assert (ra.is_failed, ra.response_time) == (rb.is_failed, rb.response_time)
    c = CommonRandomEnvironment(env_seed=43, rng=np.random.default_rng(0))
    draws42 = [a.simulate_response(pool[k], 3).response_time for k in range(10)]
    draws43 = [c.simulate_response(pool[k], 3).response_time for k in range(10)]
    assert draws42 != draws43


def test_crn_does_not_consume_shared_rng():
    shared = np.random.default_rng(5)
    manager = CommonRandomEnvironment(env_seed=42, rng=shared)
    state = shared.bit_generator.state
    manager.simulate_response(clients()[2], 4)
    assert shared.bit_generator.state == state and manager.rng is shared


def test_uniform_selector_is_uniform_and_without_replacement():
    pool = list(clients().values())
    rng = np.random.default_rng(1)
    counts = np.zeros(10)
    for _ in range(3000):
        picked = UniformSelector().boltzmann_selection(pool, 6, rng=rng)
        ids = [c.client_id for c in picked]
        assert len(set(ids)) == 6
        counts[ids] += 1
    assert np.allclose(counts / 3000, 0.6, atol=0.04)


def test_no_diversity_selector_score_equals_reliability():
    c = clients()[9]
    c.R = 0.7
    assert Selector(gamma=0.0).compute_score(c) == pytest.approx(0.7)
    assert Selector().compute_score(c) > 0.7


def _responses(staleness):
    return [ClientResponse(client_id=k, response_time=1.0, is_straggler=False, is_failed=False, staleness=s,
                           delta_weight=0.9 ** s, round_generated=1) for k, s in staleness.items()]


def test_aggregation_modes():
    pool, on_time = clients(), _responses({1: 0.0, 4: 1.0, 8: 0.0})
    full = _aggregation_function("fedmelanoma")(on_time, pool, 0.9, 1.0)
    assert full == compute_aggregation_weights(on_time, pool, 0.9, 1.0)
    no_div = _aggregation_function("no_diversity")(on_time, pool, 0.9, 1.0)
    raw = {1: 200 * 1.0, 4: 500 * 0.9, 8: 900 * 1.0}
    assert no_div == pytest.approx({k: v / sum(raw.values()) for k, v in raw.items()})
    fedavg = _aggregation_function("fedavg")(on_time, pool, 0.9, 1.0)
    assert fedavg == pytest.approx({1: 200 / 1600, 4: 500 / 1600, 8: 900 / 1600})


def test_discard_late_keeps_buffer_empty_but_same_classification():
    pool = clients()
    selected = [pool[k] for k in range(10)]
    keep = _straggler_manager(np.random.default_rng(0), 42, buffer=True, crn=True)
    drop = _straggler_manager(np.random.default_rng(0), 42, buffer=False, crn=True)
    for t in range(1, 6):
        outcome = {}
        for name, manager in (("keep", keep), ("drop", drop)):
            responses = {c.client_id: manager.simulate_response(c, t) for c in selected}
            on_time, late = quiet(manager.manage_window, selected, responses, t, len(selected))
            outcome[name] = ({r.client_id for r in on_time if r.round_generated == t}, {r.client_id for r in late})
            if name == "drop":
                assert manager.get_buffer_state() == {}
                assert all(r.round_generated == t for r in on_time)   # nothing drained
        assert outcome["keep"] == outcome["drop"]


def test_greedy_selector_is_deterministic_top_k_and_uses_no_rng():
    pool = list(clients().values())
    rng = np.random.default_rng(3)
    state = rng.bit_generator.state
    picked = _selector("greedy").boltzmann_selection(pool, 6, rng=rng)
    assert [c.client_id for c in picked] == [9, 8, 7, 6, 5, 4]   # R = 1, N_eff = 0: score order = D order
    assert rng.bit_generator.state == state


def test_no_attenuation_keeps_full_diversity_bonus():
    c = clients()[9]
    c.N_eff = 20.0
    assert _selector("boltzmann", attenuation=False).compute_score(c) == pytest.approx(2.0)
    assert Selector().compute_score(c) == pytest.approx(1.5)   # 1 + 1 / (1 + 0.05 * 20)


@pytest.mark.parametrize("mode, expected", [("ema", (0.85 + 0.15, 0.85 + 0.15 * 0.9, 0.85)),
                                            ("binary", (1.0, 0.0, 0.0)), ("static", (1.0, 1.0, 1.0))])
def test_reliability_modes(mode, expected):
    selector = _selector("boltzmann", reliability=mode)
    for staleness, r_expected in zip((0.0, 1.0, float("inf")), expected):
        c = clients()[2]
        selector.update_client_state(c, staleness, round_idx=1, was_selected=True)
        assert c.R == pytest.approx(r_expected), (mode, staleness)
        assert c.N_eff == pytest.approx(0.0 if np.isinf(staleness) else 0.9 ** staleness)


def test_total_variation_priors():
    from server.strategy import HospitalConfig
    def hospital(f, m):
        return HospitalConfig(n_samples=10, tier="reliable", latency_mu=1.0, latency_sigma=0.3,
                              fitzpatrick_dist=f, modality_dist=m)
    g = ([0.5, 0.5], [1.0, 0.0])
    hs = [hospital(*g), hospital([1.0, 0.0], [1.0, 0.0]), hospital([0.0, 1.0], [0.0, 1.0])]
    raw = [0.0, 0.40 * 0.5, 0.40 * 0.5 + 0.35 * 1.0]
    assert total_variation_diversity_priors(hs, *g) == pytest.approx([r / raw[2] for r in raw])


def test_synchronous_aggregates_every_completed_client_now():
    pool = clients()
    selected = [pool[k] for k in range(10)]
    base = _straggler_manager(np.random.default_rng(0), 42, buffer=True, crn=True)
    sync = _straggler_manager(np.random.default_rng(0), 42, buffer=True, crn=True, synchronous=True)
    saw_late = False
    for t in range(1, 8):
        late_base = {r.client_id for r in quiet(base.manage_window, selected,
                                                {c.client_id: base.simulate_response(c, t) for c in selected},
                                                t, len(selected))[1]}
        responses = {c.client_id: sync.simulate_response(c, t) for c in selected}
        on_time, late = quiet(sync.manage_window, selected, responses, t, len(selected))
        saw_late = saw_late or bool(late_base)
        assert late == [] and sync.get_buffer_state() == {}
        assert {r.client_id for r in on_time} == {k for k, r in responses.items() if not r.is_failed}
        assert all(r.staleness == 0.0 for r in on_time)
    assert saw_late   # the base arm did meet stragglers in these rounds


ARM_CHANGES = {"E00_full": 0, "R01_fedavg_reference": 3, "A1_no_diversity": 2}   # A1 is gamma = 0 in both places


def test_registry_arms_differ_in_one_factor_only():
    registry = load_registry()
    base = variant_of(registry, "E00_full")
    full = lambda v: {**{"attenuation": True, "reliability": "ema", "diversity_metric": "ot", "synchronous": False}, **v}
    for exp in registry["experiments"]:
        changed = {k for k, v in full(variant_of(registry, exp)).items() if full(base)[k] != v}
        assert len(changed) == ARM_CHANGES.get(exp, 1), (exp, changed)


def test_run_ids_unique_and_seed_independent_hash():
    registry = load_registry()
    ids = {build_run_config(e, s, registry)["experiment"]["run_id"]
           for e in registry["experiments"] for s in registry["common"]["seeds"]}
    assert len(ids) == len(registry["experiments"]) * len(registry["common"]["seeds"])
    a, b = (build_run_config("E00_full", s, registry)["experiment"]["run_id"] for s in (42, 43))
    assert a.split("-")[-1] == b.split("-")[-1]


@pytest.mark.skipif(not os.path.isdir(data_paths()[0]), reason="hospital partitions not available")
def test_apply_variant_on_real_strategy(tmp_path, monkeypatch):
    from flwr.common import Context, RecordSet
    from ablation.variants import apply_variant, prepare_variant
    from server.server import server_fn
    monkeypatch.chdir(tmp_path)
    original = strategy_module.compute_aggregation_weights
    original_priors = strategy_module.compute_normalized_diversity_priors
    ot_priors = None
    try:
        for exp in ("E00_full", "E01_random_selection", "E04_fedavg_weights", "E05_no_straggler_buffer",
                    "A3_greedy_selection", "A4_binary_reliability", "A6_tv_diversity", "A8_synchronous"):
            cfg = build_run_config(exp, 42)
            ctx = Context(run_id=0, node_id=0, node_config={}, state=RecordSet(), run_config={
                "num-server-rounds": 1, "local-epochs": 1, "clients-per-round": 6,
                "partitions-root": cfg["data"]["partitions_root"], "seed": 42, "run-id": f"t-{exp}"})
            prepare_variant(cfg["ablation"]["variant"])
            strategy = quiet(server_fn, ctx).strategy
            record = apply_variant(strategy, cfg["ablation"]["variant"], cfg["ablation"]["harness"], 42)
            assert isinstance(strategy._straggler_manager, CommonRandomEnvironment)
            expected_selector = {"E01_random_selection": "UniformSelector", "A3_greedy_selection": "GreedySelector",
                                 "A4_binary_reliability": "BinaryReliabilitySelector"}.get(exp, "Selector")
            assert record["selector_class"] == expected_selector
            assert ("DiscardLate" in record["straggler_manager_class"]) == (exp == "E05_no_straggler_buffer")
            assert ("WaitForAll" in record["straggler_manager_class"]) == (exp == "A8_synchronous")
            if exp == "E00_full":
                ot_priors = record["diversity_priors_D"]
            if exp == "A6_tv_diversity":
                assert record["diversity_priors_D"] != ot_priors and max(record["diversity_priors_D"].values()) == 1.0
            strategy_module.compute_normalized_diversity_priors = original_priors
    finally:
        strategy_module.compute_aggregation_weights = original
        strategy_module.compute_normalized_diversity_priors = original_priors
