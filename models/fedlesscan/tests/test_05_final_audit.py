"""Test 5: final pre-training audit (failure handling, aggregation, experimental consistency). CPU only."""
import contextlib
import hashlib
import io
import os

import numpy as np
import pytest
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

import server.fedlesscan as fl
from test_02_federation_math import SIZES, FakeProxy, force_env, make_strategy, quiet, result  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXTBOOK = os.path.abspath(os.path.join(ROOT, "..", "resnet50_fedavg_textbook"))


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# ---- B. failure handling -------------------------------------------------------------------

def one_round(strategy, manager, proxies, parameters, t, rng):
    with quiet():
        strategy.configure_fit(t, parameters, manager)
    sel = strategy._last_round_selected_ids
    arrays = {i: [rng.normal(size=3).astype(np.float32) + 100.0 * i, np.array(10 + i, dtype=np.int64)] for i in sel}
    with quiet():
        out, metrics = strategy.aggregate_fit(t, [result(proxies[i], arrays[i], SIZES[i]) for i in sel], [])
    return out, sel, arrays


def test_failed_job_contributes_nothing_and_is_not_a_successful_training(tmp_path):
    strategy, manager, proxies, initial = make_strategy(tmp_path)
    parameters = ndarrays_to_parameters(initial)
    failed = {}
    force_env(strategy, lambda cid, k: (True, float("inf")) if cid in failed else (False, 5.0))
    with quiet():
        strategy.configure_fit(1, parameters, manager)
    sel = strategy._last_round_selected_ids
    failed[sel[0]] = True
    rng = np.random.default_rng(0)
    arrays = {i: [rng.normal(size=3).astype(np.float32) + 100.0 * i, np.array(10 + i, dtype=np.int64)] for i in sel}
    with quiet():
        out, _ = strategy.aggregate_fit(1, [result(proxies[i], arrays[i], SIZES[i]) for i in sel], [])
    f = strategy._fedlesscan
    ok = sel[1:]
    got = parameters_to_ndarrays(out)[0]
    total = sum(SIZES[i] for i in ok)
    assert np.allclose(got, sum(SIZES[i] / total * arrays[i][0].astype(np.float64) for i in ok), atol=1e-4)  # failed update absent
    h = f.histories[sel[0]]
    assert h.training_times == [] and h.missed_rounds == [0] and h.client_backoff == 1   # not a success; cooldown started
    assert strategy._round_plan[1]["failed"] == [sel[0]] and strategy._round_plan[1]["late"] == []
    assert all(r.client_id != sel[0] for r in f.results)                               # nothing stored for it
    assert f.in_flight == []                                                           # and it never arrives later


def test_failure_and_lateness_are_different_outcomes_with_the_same_cooldown_policy(tmp_path):
    strategy, manager, proxies, initial = make_strategy(tmp_path)
    roles = {}
    force_env(strategy, lambda cid, k: (True, float("inf")) if roles.get(cid) == "f" else ((False, 45.0) if roles.get(cid) == "l" else (False, 5.0)))
    with quiet():
        strategy.configure_fit(1, ndarrays_to_parameters(initial), manager)
    sel = strategy._last_round_selected_ids
    roles[sel[0]], roles[sel[1]] = "f", "l"
    rng = np.random.default_rng(1)
    arrays = {i: [rng.normal(size=3).astype(np.float32), np.array(1, dtype=np.int64)] for i in sel}
    with quiet():
        strategy.aggregate_fit(1, [result(proxies[i], arrays[i], SIZES[i]) for i in sel], [])
    plan, f = strategy._round_plan[1], strategy._fedlesscan
    assert plan["failed"] == [sel[0]] and plan["late"] == [sel[1]]
    assert len(f.in_flight) == 1 and f.in_flight[0]["client_id"] == sel[1]       # only the late one will arrive
    for c in (sel[0], sel[1]):                                                   # both missed the round: official cooldown
        assert f.histories[c].missed_rounds == [0] and f.histories[c].client_backoff == 1
    for c in sel[2:]:
        assert f.histories[c].client_backoff == 0 and len(f.histories[c].training_times) == 1


def test_cooldown_doubles_and_blocks_selection_in_the_official_way():
    f = fl.FedLesScan(range(10), 6, 50, 42, 6)
    for r, want in ((0, 1), (1, 2), (2, 4)):
        f.finish_round(r, [0], [])
        assert f.histories[0].client_backoff == want
    rookies, rest, stragglers = fl.filter_rookies([f.histories[c] for c in range(10)], 3)
    assert 0 in [c.client_id for c in stragglers]       # in cooldown: backoff 4 + last missed round 2 >= round 3


def test_late_update_within_tolerance_is_used_and_beyond_is_dropped(tmp_path):
    """A job 100 s late arrives 3 rounds later (staleness 3 > 2): stored, then ignored by the aggregation."""
    strategy, manager, proxies, initial = make_strategy(tmp_path)
    slow = {}
    force_env(strategy, lambda cid, k: (False, 100.0) if (cid in slow and k == 0) else (False, 5.0))
    with quiet():
        strategy.configure_fit(1, ndarrays_to_parameters(initial), manager)
    slow[strategy._last_round_selected_ids[0]] = True
    cid = strategy._last_round_selected_ids[0]
    rng = np.random.default_rng(2)
    parameters = ndarrays_to_parameters(initial)
    arrays = {i: [rng.normal(size=3).astype(np.float32), np.array(1, dtype=np.int64)] for i in strategy._last_round_selected_ids}
    with quiet():
        parameters, _ = strategy.aggregate_fit(1, [result(proxies[i], arrays[i], SIZES[i]) for i in strategy._last_round_selected_ids], [])
    assert [j["arrival_round"] for j in strategy._fedlesscan.in_flight] == [0 + 4 - 1]   # ceil(100/30) - 1 = 3 rounds later
    seen = []
    for t in range(2, 6):
        parameters, sel, _ = one_round(strategy, manager, proxies, parameters, t, rng)
        seen += [e for e in strategy._round_plan[t]["aggregated"] if e == [cid, 1]]
    assert seen == []                                                                  # arrived at round0 3: staleness 3, dropped
    assert 0 not in strategy._fedlesscan.histories[cid].missed_rounds                   # but it did finish: missed round cleared


# ---- C. aggregation ------------------------------------------------------------------------

def test_empty_aggregation_keeps_the_global_model(tmp_path):
    strategy, manager, proxies, initial = make_strategy(tmp_path)
    force_env(strategy, lambda cid, k: (True, float("inf")))   # every job fails
    rng = np.random.default_rng(3)
    parameters = ndarrays_to_parameters(initial)
    out, sel, _ = one_round(strategy, manager, proxies, parameters, 1, rng)
    kept = parameters_to_ndarrays(out)
    for a, b in zip(kept, initial):
        assert np.array_equal(a, b)
    assert strategy._round_plan[1]["aggregated"] == []


def test_no_novelty_weight_or_diversity_factor_in_the_effective_weights(tmp_path):
    strategy, manager, proxies, initial = make_strategy(tmp_path)
    force_env(strategy, lambda cid, k: (False, 5.0))
    parameters = ndarrays_to_parameters(initial)
    one_round(strategy, manager, proxies, parameters, 1, np.random.default_rng(4))
    # effective weight = n_k * (t_k+1)/(t+1) / sum n  ->  sample-count weights summing to 1 when all are on time
    sel = strategy._round_plan[1]["selected"]
    total = sum(SIZES[i] for i in sel)
    with open(os.path.join(str(tmp_path), "logs", "math", "unit.jsonl"), encoding="utf-8") as fh:
        lines = [l for l in fh if '"fedlesscan_aggregation"' in l]
    import json
    rec = json.loads(lines[0])
    for c in rec["candidates"]:
        assert c["effective_weight"] == pytest.approx(SIZES[c["client_id"]] / total)


# ---- D. experimental consistency -----------------------------------------------------------

def test_configuration_is_the_experimental_plan(cfg):
    assert cfg["federation"] == {"num_server_rounds": 50, "clients_per_round": 6, "local_epochs": 5, "num_clients": 10}
    assert cfg["fedlesscan"]["round_deadline_seconds"] == 30 and cfg["fedlesscan"]["staleness_tolerance"] == 2
    assert cfg["local_training"] == {"batch_size": 20, "learning_rate": 1.0e-4, "weight_decay": 1.0e-4, "focal_gamma": 2.0}
    assert cfg["model"]["backbone"] == "resnet50" and cfg["model"]["fusion_dim"] == 2080 and cfg["model"]["metadata_dim"] == 32


def test_same_partitions_and_three_tiers(hospitals, cfg):
    from hospital_registry import build_hospital_registry
    names, records = build_hospital_registry(cfg["data"]["partitions_root"])
    assert names == hospitals and len(names) == 10
    assert [r["n_samples"] for r in records] == SIZES                         # the 7,788 training images of the ten hospitals
    assert [r["tier"] for r in records] == ["reliable", "moderate", "high_straggler"] * 3 + ["reliable"]
    assert {r["tier"]: (r["latency_mu"], r["latency_sigma"]) for r in records} == {
        "reliable": (1.39, 0.30), "moderate": (2.77, 0.50), "high_straggler": (4.09, 0.80)}
    from server.environment import BETA_PARAMS
    assert BETA_PARAMS == {"reliable": (19.0, 1.0), "moderate": (7.0, 3.0), "high_straggler": (2.0, 3.0)}
    for r in records:
        assert set(r) == {"n_samples", "tier", "latency_mu", "latency_sigma"}


def test_model_loss_trainer_and_data_files_are_byte_identical_to_the_textbook_run():
    for rel in ("models/model.py", "client/dataset.py", "client/focal_loss.py", "client/local_trainer.py", "runtime/image_cache.py"):
        assert sha(os.path.join(ROOT, rel)) == sha(os.path.join(TEXTBOOK, rel)), rel


def test_fedlesscan_constants_are_the_official_ones():
    assert (fl.TRAINING_TIME_EMA_ALPHA, fl.MISSED_ROUND_EMA_ALPHA, fl.MISSED_ROUND_PENALTY_SCALE, fl.DBSCAN_MIN_SAMPLES) == (0.5, 0.8, 1.5, 2)
    assert (fl.EPS_START, fl.EPS_STOP, fl.EPS_STEP, fl.STALENESS_TOLERANCE) == (0.01, 2, 0.1, 2)
