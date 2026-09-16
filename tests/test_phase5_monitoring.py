"""
tests/test_phase5_monitoring.py
=================================
Phase 5 verification: checkpointing, parameter-broadcast proof, structured
logging, subgroup evaluation, and straggler monitoring.

Plain script (no pytest dependency), same style as
tests/test_phase4_runtime.py. Synthetic data (fake ClientProxy objects,
small random arrays, a handful of temp-directory CSV/image rows) is used
ONLY in this file -- no production module is altered to accommodate it,
and no fake dataset is written into the repository.

Run with:  python tests/test_phase5_monitoring.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from typing import Callable, List, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

for _stream in (sys.stdout, sys.stderr):
    if getattr(_stream, "encoding", "").lower() != "utf-8":
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

import numpy as np
import pandas as pd


# ===========================================================================
# Result bookkeeping (identical pattern to test_phase4_runtime.py)
# ===========================================================================

class CheckResult:
    def __init__(self, name: str, status: str, detail: str = "") -> None:
        self.name = name
        self.status = status
        self.detail = detail


RESULTS: List[CheckResult] = []


class _Skip(Exception):
    pass


def run_check(name: str, fn: Callable[[], str]) -> None:
    try:
        detail = fn() or ""
        RESULTS.append(CheckResult(name, "PASS", detail))
    except _Skip as exc:
        RESULTS.append(CheckResult(name, "SKIP", str(exc)))
    except Exception as exc:  # noqa: BLE001
        detail = f"{type(exc).__name__}: {exc}"
        RESULTS.append(CheckResult(name, "FAIL", detail))
        traceback.print_exc()


# ===========================================================================
# Shared fixture: a multi-round Strategy run driven with fake ClientProxy
# objects that echo REAL parameter hashes and REAL-shaped subgroup
# confusion-count JSON, exactly like client/client.py's real fit()/
# evaluate() would -- so this test exercises the genuine hash/JSON
# contract, not a hand-waved mock.
# ===========================================================================

_RUN_STATE: dict = {}


def _build_and_run_strategy(n_rounds: int = 3, n_hospitals: int = 6):
    from flwr.common import Code, EvaluateRes, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.server.client_manager import SimpleClientManager
    from flwr.server.client_proxy import ClientProxy

    from server.strategy import FedMelanomaStrategy, HospitalConfig
    from monitoring.experiment_logging import ExperimentLogger, new_run_id
    from monitoring.param_hash import hash_ndarrays
    from monitoring.subgroup_metrics import SUBGROUP_AXES

    rng = np.random.default_rng(123)
    checkpoint_dir = os.path.join(_RUN_STATE["tmp_root"], "checkpoints")
    results_dir = os.path.join(_RUN_STATE["tmp_root"], "results")
    logs_dir = os.path.join(_RUN_STATE["tmp_root"], "logs")
    run_id = new_run_id("phase5test")
    logger = ExperimentLogger(run_id, logs_root=logs_dir)

    class FakeClientProxy(ClientProxy):
        """
        In-memory ClientProxy stand-in. Computes REAL hash_ndarrays() and
        REAL-shaped subgroup-confusion-count JSON exactly as
        client/client.py's genuine fit()/evaluate() do, so the strategy's
        broadcast-verification and subgroup-aggregation code paths are
        exercised faithfully rather than mocked away.
        """

        def __init__(self, cid: str, n_examples: int) -> None:
            super().__init__(cid)
            self.node_id = int(cid)
            self.n_examples = n_examples

        def get_properties(self, ins, timeout, group_id):
            return None

        def get_parameters(self, ins, timeout, group_id):
            raise NotImplementedError

        def fit(self, ins, timeout, group_id):
            received = parameters_to_ndarrays(ins.parameters)
            received_hash = hash_ndarrays(received)  # real client-side hash, pre-training
            perturbed = [arr + rng.normal(scale=0.01, size=arr.shape).astype(arr.dtype) for arr in received]
            returned_hash = hash_ndarrays(perturbed)
            return FitRes(
                status=Status(Code.OK, ""),
                parameters=ndarrays_to_parameters(perturbed),
                num_examples=self.n_examples,
                metrics={
                    "train_loss": float(rng.random()),
                    "client_received_param_hash": received_hash,
                    "client_returned_param_hash": returned_hash,
                },
            )

        def evaluate(self, ins, timeout, group_id):
            received = parameters_to_ndarrays(ins.parameters)
            received_hash = hash_ndarrays(received)
            subgroup_counts = {}
            for axis, categories in SUBGROUP_AXES.items():
                subgroup_counts[axis] = {}
                for category in categories:
                    n = int(rng.integers(0, 5))
                    tp = int(rng.integers(0, max(n, 1)))
                    tn = max(n - tp, 0)
                    subgroup_counts[axis][category] = {"n": n, "tn": tn, "fp": 0, "fn": 0, "tp": tp}
            return EvaluateRes(
                status=Status(Code.OK, ""),
                loss=float(rng.random()),
                num_examples=self.n_examples,
                metrics={
                    "accuracy": float(rng.random()),
                    "precision": float(rng.random()),
                    "recall": float(rng.random()),
                    "f1": float(rng.random()),
                    "roc_auc": float(rng.random()),
                    "pr_auc": float(rng.random()),
                    "specificity": float(rng.random()),
                    "balanced_accuracy": float(rng.random()),
                    "client_received_param_hash": received_hash,
                    "subgroup_confusion_counts_json": json.dumps(subgroup_counts),
                },
            )

        def reconnect(self, ins, timeout, group_id):
            raise NotImplementedError

    configs = [
        HospitalConfig(
            n_samples=40 + 3 * i,
            tier=["reliable", "moderate", "high_straggler"][i % 3],
            latency_mu=1.39, latency_sigma=0.30,
            fitzpatrick_dist=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1],
            modality_dist=[0.8, 0.15, 0.05],
            confirmation_dist=[0.2, 0.2, 0.3, 0.3],
        )
        for i in range(n_hospitals)
    ]
    param_keys = [f"layer_{i}.weight" for i in range(3)]
    initial_parameters = [
        np.random.default_rng(0).normal(size=(4, 3)).astype(np.float32),
        np.zeros((4,), dtype=np.float32),
        np.random.default_rng(1).normal(size=(2, 4)).astype(np.float32),
    ]

    strategy = FedMelanomaStrategy(
        hospital_configs=configs,
        global_fitzpatrick=[1 / 6] * 6,
        global_modality=[0.8, 0.15, 0.05],
        global_confirmation=[0.25, 0.25, 0.25, 0.25],
        n_select=3,
        initial_parameters=initial_parameters,
        rng=np.random.default_rng(0),
        experiment_logger=logger,
        run_id=run_id,
        checkpoint_dir=checkpoint_dir,
        results_dir=results_dir,
        param_keys=param_keys,
        seed=123,
    )

    client_manager = SimpleClientManager()
    for cid in range(n_hospitals):
        client_manager.register(FakeClientProxy(str(cid), n_examples=configs[cid].n_samples))

    parameters = strategy.initialize_parameters(client_manager)
    for server_round in range(1, n_rounds + 1):
        fit_instructions = strategy.configure_fit(server_round, parameters, client_manager)
        fit_results = [(proxy, proxy.fit(ins, timeout=None, group_id=None)) for proxy, ins in fit_instructions]
        parameters, _ = strategy.aggregate_fit(server_round, fit_results, [])

        eval_instructions = strategy.configure_evaluate(server_round, parameters, client_manager)
        eval_results = [(proxy, proxy.evaluate(ins, timeout=None, group_id=None)) for proxy, ins in eval_instructions]
        strategy.aggregate_evaluate(server_round, eval_results, [])

    return {
        "run_id": run_id,
        "checkpoint_dir": checkpoint_dir,
        "results_dir": results_dir,
        "logs_dir": logs_dir,
        "n_rounds": n_rounds,
        "n_hospitals": n_hospitals,
    }


def _get_run() -> dict:
    if "run" not in _RUN_STATE:
        raise _Skip("run fixture not built (see check 1)")
    return _RUN_STATE["run"]


# ===========================================================================
# Check 1 -- build the fixture (multi-round Strategy run)
# ===========================================================================

def check_build_fixture() -> str:
    tmp_root = tempfile.mkdtemp(prefix="fedmelanoma_phase5_")
    _RUN_STATE["tmp_root"] = tmp_root
    run = _build_and_run_strategy(n_rounds=3, n_hospitals=6)
    _RUN_STATE["run"] = run
    return f"ran {run['n_rounds']} rounds with {run['n_hospitals']} hospitals; run_id={run['run_id']}"


# ===========================================================================
# Check 2 -- Requirement 1: checkpoints exist for every round + best exists
# ===========================================================================

def check_checkpoints_exist() -> str:
    run = _get_run()
    run_dir = os.path.join(run["checkpoint_dir"], run["run_id"])
    for r in range(1, run["n_rounds"] + 1):
        path = os.path.join(run_dir, f"global_round_{r:04d}.pth")
        assert os.path.isfile(path), f"missing checkpoint {path}"
    best_path = os.path.join(run_dir, "best_global_model.pth")
    assert os.path.isfile(best_path), f"missing {best_path}"
    return f"global_round_0001..{run['n_rounds']:04d}.pth and best_global_model.pth all present"


# ===========================================================================
# Check 3 -- TEST 3/4: checkpoint matches aggregated model & is loadable
# ===========================================================================

def check_checkpoint_matches_and_loads() -> str:
    import torch
    from collections import OrderedDict
    from monitoring.checkpointing import load_global_checkpoint
    from monitoring.experiment_logging import read_jsonl

    run = _get_run()
    run_dir = os.path.join(run["checkpoint_dir"], run["run_id"])
    math_log = list(read_jsonl(os.path.join(run["logs_dir"], "math", f"{run['run_id']}.jsonl")))
    aggregate_rows = {r["round"]: r for r in math_log if r.get("event") == "round_aggregate"}

    for r in range(1, run["n_rounds"] + 1):
        path = os.path.join(run_dir, f"global_round_{r:04d}.pth")
        ndarrays, ckpt = load_global_checkpoint(path)
        assert ckpt["round"] == r
        logged_hash = aggregate_rows[r]["global_output_hash"]
        assert ckpt["param_hash"] == logged_hash, f"round {r}: checkpoint hash != logged aggregate hash"

        # TEST 4: loadable into a fresh named state_dict (param_keys were
        # threaded through, so this checks the REAL state_dict path, not
        # just the raw-ndarrays fallback).
        assert ckpt["model_state_dict"] is not None, "expected a named state_dict (param_keys were provided)"
        state_dict = OrderedDict(ckpt["model_state_dict"])
        assert list(state_dict.keys()) == [f"layer_{i}.weight" for i in range(3)]
        for k, v in state_dict.items():
            assert isinstance(v, torch.Tensor)

    return f"checkpoint param_hash matches strategy's logged global_output_hash for all {run['n_rounds']} rounds; state_dict loadable"


# ===========================================================================
# Check 4 -- Requirement 2 / TEST 1 & 2: broadcast hash-chain proof
# ===========================================================================

def check_broadcast_hash_chain() -> str:
    from monitoring.experiment_logging import read_jsonl

    run = _get_run()
    math_log = list(read_jsonl(os.path.join(run["logs_dir"], "math", f"{run['run_id']}.jsonl")))
    broadcasts = {r["round"]: r for r in math_log if r.get("event") == "round_broadcast"}
    aggregates = {r["round"]: r for r in math_log if r.get("event") == "round_aggregate"}
    client_returns = [r for r in math_log if r.get("event") == "client_fit_returned"]

    assert broadcasts[1]["previous_round_global_output_hash"] is None, "round 1 has no previous round to compare"

    # TEST 1: aggregated_round_t == broadcast_round_(t+1)
    for r in range(1, run["n_rounds"]):
        assert broadcasts[r + 1]["global_input_hash"] == aggregates[r]["global_output_hash"], (
            f"round {r+1} broadcast hash != round {r} aggregated hash"
        )
        assert broadcasts[r + 1]["broadcast_matches_previous_round_output"] is True

    # TEST 2: every client that returned a fit result received exactly
    # that round's broadcast (client-computed hash, not the server's own
    # copy) before training.
    assert client_returns, "no client_fit_returned records logged"
    for rec in client_returns:
        assert rec["client_received_matches_broadcast"] is True, (
            f"round {rec['round']} client {rec['client_id']}: "
            f"client-reported received hash != server broadcast hash"
        )

    return (
        f"verified aggregated_hash(t) == broadcast_hash(t+1) for rounds 1..{run['n_rounds']-1}, "
        f"and client_received_hash == broadcast_hash for all {len(client_returns)} client fit returns"
    )


# ===========================================================================
# Check 5 -- Requirement 3: op/math/audit logs exist and are non-empty
# ===========================================================================

def check_structured_logs_exist() -> str:
    from monitoring.experiment_logging import read_jsonl

    run = _get_run()
    counts = {}
    for tier in ("op", "math", "audit"):
        path = os.path.join(run["logs_dir"], tier, f"{run['run_id']}.jsonl")
        rows = list(read_jsonl(path))
        assert rows, f"logs/{tier}/{run['run_id']}.jsonl is empty or missing"
        counts[tier] = len(rows)

    straggler_path = os.path.join(run["logs_dir"], "op", "stragglers", f"{run['run_id']}.jsonl")
    straggler_rows = list(read_jsonl(straggler_path))
    assert straggler_rows, "straggler log is empty"
    counts["straggler"] = len(straggler_rows)

    audit_events = {r.get("event") for r in read_jsonl(os.path.join(run["logs_dir"], "audit", f"{run['run_id']}.jsonl"))}
    assert "run_start" in audit_events, "audit log missing run_start reproducibility record"
    assert "checkpoint_saved" in audit_events, "audit log missing checkpoint_saved events"

    return f"op={counts['op']} math={counts['math']} audit={counts['audit']} straggler={counts['straggler']} records"


# ===========================================================================
# Check 6 -- TEST 5: every selected client has a straggler outcome recorded
# ===========================================================================

def check_every_selected_client_has_straggler_outcome() -> str:
    from monitoring.experiment_logging import read_jsonl

    run = _get_run()
    op_log = list(read_jsonl(os.path.join(run["logs_dir"], "op", f"{run['run_id']}.jsonl")))
    straggler_log = list(read_jsonl(os.path.join(run["logs_dir"], "op", "stragglers", f"{run['run_id']}.jsonl")))
    outcome_rows = [r for r in straggler_log if r.get("event") == "client_round_outcome"]

    selected_pairs = set()
    for r in op_log:
        if r.get("event") == "round_configure_fit":
            for cid in r["selected_clients"]:
                selected_pairs.add((r["round"], cid))

    recorded_pairs = {(r["round"], r["client_id"]) for r in outcome_rows}
    missing = selected_pairs - recorded_pairs
    assert not missing, f"selected clients with no recorded straggler outcome: {missing}"
    assert all(r["response_classification"] in ("on_time", "straggler", "failed") for r in outcome_rows)

    return f"all {len(selected_pairs)} (round, selected-client) pairs have a recorded straggler outcome"


# ===========================================================================
# Check 7 -- TEST 6: every aggregated update has a math log entry
# ===========================================================================

def check_aggregation_has_math_log() -> str:
    from monitoring.experiment_logging import read_jsonl

    run = _get_run()
    math_log = list(read_jsonl(os.path.join(run["logs_dir"], "math", f"{run['run_id']}.jsonl")))
    aggregate_rows = {r["round"]: r for r in math_log if r.get("event") == "round_aggregate"}
    assert len(aggregate_rows) == run["n_rounds"], "expected one round_aggregate record per round"
    for r, rec in aggregate_rows.items():
        assert "aggregation_c_i" in rec
        assert rec["n_aggregated"] == len(rec["aggregation_c_i"]), (
            f"round {r}: n_aggregated does not match number of logged c_i entries"
        )
    return f"round_aggregate log entry present for all {run['n_rounds']} rounds, c_i counts consistent"


# ===========================================================================
# Check 8 -- TEST 7: D_i is static across rounds
# ===========================================================================

def check_diversity_is_static() -> str:
    from monitoring.experiment_logging import read_jsonl

    run = _get_run()
    straggler_log = list(read_jsonl(os.path.join(run["logs_dir"], "op", "stragglers", f"{run['run_id']}.jsonl")))
    outcome_rows = [r for r in straggler_log if r.get("event") == "client_round_outcome"]
    by_client: dict = {}
    for r in outcome_rows:
        by_client.setdefault(r["client_id"], set()).add(r["diversity_D_static"])
    non_static = {cid: vals for cid, vals in by_client.items() if len(vals) > 1}
    assert not non_static, f"D_i changed across rounds for clients: {non_static}"
    return f"D_i constant across all rounds for every one of {len(by_client)} clients"


# ===========================================================================
# Check 9 -- TEST 9: no duplicate client contribution
# ===========================================================================

def check_no_duplicate_contribution() -> str:
    from monitoring.experiment_logging import read_jsonl

    run = _get_run()
    audit_log = list(read_jsonl(os.path.join(run["logs_dir"], "audit", f"{run['run_id']}.jsonl")))
    duplicates = [r for r in audit_log if r.get("event") == "DUPLICATE_CLIENT_CONTRIBUTION_DETECTED"]
    assert not duplicates, f"duplicate contributions detected: {duplicates}"

    math_log = list(read_jsonl(os.path.join(run["logs_dir"], "math", f"{run['run_id']}.jsonl")))
    for rec in math_log:
        if rec.get("event") == "round_aggregate":
            c_i = rec["aggregation_c_i"]
            assert len(c_i) == len(set(c_i.keys())), "duplicate client_id keys in aggregation_c_i"

    return "no DUPLICATE_CLIENT_CONTRIBUTION_DETECTED audit events; every round's aggregation_c_i keys are unique"


# ===========================================================================
# Check 10 -- Requirement 4/5, TEST 10: federated global subgroup snapshot
# ===========================================================================

def check_federated_subgroup_snapshot() -> str:
    run = _get_run()
    subgroup_dir = os.path.join(run["results_dir"], run["run_id"], "subgroup")
    axes = ("skin_type", "image_type", "confirm_type")
    found = []
    for r in range(1, run["n_rounds"] + 1):
        for axis in axes:
            path = os.path.join(subgroup_dir, f"round_{r:04d}_{axis}.csv")
            assert os.path.isfile(path), f"missing {path}"
            df = pd.read_csv(path)
            assert "subgroup" in df.columns and "sensitivity" in df.columns
            found.append(path)
    return f"{len(found)} per-round subgroup snapshot CSVs written across {run['n_rounds']} rounds x {len(axes)} axes"


# ===========================================================================
# Check 11 -- monitoring.subgroup_metrics: NaN-guarding + fairness gap
# (direct unit test, synthetic-only, of the reusable module)
# ===========================================================================

def check_subgroup_metrics_nan_guarding() -> str:
    from monitoring.subgroup_metrics import compute_subgroup_metrics, compute_fairness_gap

    rng = np.random.default_rng(5)
    n = 60
    y_true = rng.integers(0, 2, size=n)
    y_pred = rng.integers(0, 2, size=n)
    y_prob = rng.random(size=n)
    categories = ["I", "II", "III", "IV", "V", "VI"]
    group_labels = np.array(categories)[rng.integers(0, 4, size=n)]  # V, VI never appear -> 0 samples

    df = compute_subgroup_metrics(y_true, y_pred, y_prob, group_labels, categories)
    zero_sample_rows = df[df["subgroup"].isin(["V", "VI"])]
    assert (zero_sample_rows["n_samples"] == 0).all()
    assert zero_sample_rows["sensitivity"].isna().all(), "0-sample subgroup should report NaN, not a fabricated value"
    assert (zero_sample_rows["unavailable_reason"] != "").all()

    # Force a subgroup with samples but zero malignant cases -> sensitivity/roc_auc/etc must be NaN
    y_true_forced = y_true.copy()
    mask_I = group_labels == "I"
    y_true_forced[mask_I] = 0  # no malignant cases in group I
    df2 = compute_subgroup_metrics(y_true_forced, y_pred, y_prob, group_labels, categories)
    row_I = df2[df2["subgroup"] == "I"].iloc[0]
    assert row_I["n_malignant"] == 0
    assert pd.isna(row_I["sensitivity"]) and pd.isna(row_I["roc_auc"]), (
        "subgroup with 0 malignant samples must report NaN sensitivity/roc_auc, not a silently invalid 0.0"
    )

    gap_info = compute_fairness_gap(df, metric_col="sensitivity")
    assert set(gap_info["excluded_groups"]) >= {"V", "VI"}, "V/VI should be excluded from the fairness gap, not fabricated"

    return "0-sample and 0-malignant subgroups correctly report NaN (not fabricated 0.0); fairness gap excludes them explicitly"


# ===========================================================================
# Check 12 -- monitoring.subgroup_metrics: confusion-count aggregation is
# mathematically exact (sum-then-divide vs a hand-computed expectation)
# ===========================================================================

def check_confusion_count_aggregation_exact() -> str:
    from monitoring.subgroup_metrics import aggregate_client_confusion_reports, SUBGROUP_AXES

    axis = "skin_type"
    category = SUBGROUP_AXES[axis][0]
    reports = [
        {axis: {category: {"n": 10, "tn": 5, "fp": 1, "fn": 1, "tp": 3}}},
        {axis: {category: {"n": 6, "tn": 2, "fp": 0, "fn": 1, "tp": 3}}},
    ]
    result = aggregate_client_confusion_reports(reports)
    row = result[axis][result[axis]["subgroup"] == category].iloc[0]
    total_tp, total_fn = 3 + 3, 1 + 1
    expected_sensitivity = total_tp / (total_tp + total_fn)
    assert abs(row["sensitivity"] - expected_sensitivity) < 1e-9
    assert row["n_samples"] == 16
    return f"summed-confusion-count sensitivity == hand-computed expectation ({expected_sensitivity:.4f})"


# ===========================================================================
# Check 13 -- Requirement 7: graphs generated from saved logs
# ===========================================================================

def check_plots_generated() -> str:
    from monitoring.plotting import generate_all_plots

    run = _get_run()
    written = generate_all_plots(run["run_id"], logs_root=run["logs_dir"], results_root=run["results_dir"])
    assert written, "no plots were generated from the saved logs"

    plots_dir = os.path.join(run["results_dir"], run["run_id"], "plots")
    for name in written:
        png_path = os.path.join(plots_dir, f"{name}.png")
        assert os.path.isfile(png_path), f"plot listed as written but file missing: {png_path}"

    return f"{len(written)} plots generated and verified on disk: {sorted(written)}"


# ===========================================================================
# Check 14 -- reproducibility record (Requirement 11)
# ===========================================================================

def check_reproducibility_record() -> str:
    from monitoring.experiment_logging import read_jsonl

    run = _get_run()
    audit_log = list(read_jsonl(os.path.join(run["logs_dir"], "audit", f"{run['run_id']}.jsonl")))
    run_start = next(r for r in audit_log if r.get("event") == "run_start")
    for key in ("n_hospitals", "n_select", "local_epochs", "seed", "selector_hyperparameters", "straggler_hyperparameters"):
        assert key in run_start, f"run_start audit record missing '{key}'"
    return "run_start audit record contains seed, round/client config, and both research modules' hyperparameters"


# ===========================================================================
# Check 15 -- Requirement 4 (evaluation side): evaluate.py's
# build_test_loader() + subgroup CSV path, end-to-end with real (synthetic)
# images through the real model -- not just the isolated module unit tests
# above.
# ===========================================================================

def check_evaluate_subgroup_integration() -> str:
    import cv2
    import torch

    from client.dataset import MelanomaDataset
    from client.local_trainer import run_inference
    from models.model import FedMelanomaModel
    from evaluate import build_test_loader, compute_metrics
    from monitoring.subgroup_metrics import compute_all_subgroup_axes, compute_fairness_gap

    tmp_root = tempfile.mkdtemp(prefix="fedmelanoma_phase5_evalcheck_")
    rng = np.random.default_rng(9)
    hospital_names = ["SynthHospA", "SynthHospB"]
    for hospital in hospital_names:
        img_dir = os.path.join(tmp_root, "hospital_partitions", hospital, "images", "Test")
        os.makedirs(img_dir, exist_ok=True)
        rows = []
        for i in range(6):
            isic_id = f"{hospital}_{i:03d}"
            cv2.imwrite(os.path.join(img_dir, isic_id + ".jpg"), rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8))
            rows.append(
                {
                    "isic_id": isic_id,
                    "diagnosis_1": "malignant" if i == 0 else "benign",
                    "fitzpatrick_skin_type": rng.choice(MelanomaDataset._SKIN_TYPES),
                    "image_type": rng.choice(MelanomaDataset._IMAGE_TYPES),
                    "diagnosis_confirm_type": rng.choice(MelanomaDataset._CONFIRM_TYPES),
                }
            )
        pd.DataFrame(rows).to_csv(
            os.path.join(tmp_root, "hospital_partitions", hospital, "metadata_test.csv"), index=False
        )

    cwd = os.getcwd()
    try:
        os.chdir(tmp_root)  # build_test_loader() reads "hospital_partitions/..." relative to cwd
        test_loader, loaded, metadata_df, hospital_labels = build_test_loader(hospital_names, batch_size=4)
        assert len(loaded) == 2
        assert len(metadata_df) == len(test_loader.dataset) == 12
        assert len(hospital_labels) == 12

        try:
            model = FedMelanomaModel()
        except Exception as exc:
            raise _Skip(f"FedMelanomaModel() unavailable (no network for pretrained weights?): {exc}") from exc

        device = torch.device("cpu")
        y_true, y_pred, y_prob = run_inference(model, test_loader, device, positive_class_idx=1)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        subgroup_tables = compute_all_subgroup_axes(y_true, y_pred, y_prob, metadata_df, positive_class_idx=1)

        os.makedirs("results/metrics", exist_ok=True)
        for axis, filename in (
            ("skin_type", "skin_type_metrics.csv"),
            ("image_type", "image_type_metrics.csv"),
            ("confirm_type", "confirm_type_metrics.csv"),
        ):
            path = os.path.join("results", "metrics", filename)
            subgroup_tables[axis].to_csv(path, index=False)
            assert os.path.isfile(path)
            df = pd.read_csv(path)
            assert set(["subgroup", "n_samples", "sensitivity", "roc_auc"]).issubset(df.columns)

        fairness = compute_fairness_gap(subgroup_tables["skin_type"], metric_col="sensitivity")
        assert "gap" in fairness
    finally:
        os.chdir(cwd)

    return (
        f"build_test_loader()->compute_metrics()->compute_all_subgroup_axes() ran end-to-end on "
        f"{len(metadata_df)} synthetic samples across {len(loaded)} hospitals; 3 subgroup CSVs written and readable"
    )


# ===========================================================================
# Main
# ===========================================================================

CHECKS: List[Tuple[str, Callable[[], str]]] = [
    ("1. Build multi-round Strategy fixture", check_build_fixture),
    ("2. [Req 1] Checkpoints exist for every round + best", check_checkpoints_exist),
    ("3. [TEST 3/4] Checkpoint matches aggregated model & loads", check_checkpoint_matches_and_loads),
    ("4. [Req 2, TEST 1/2] Broadcast hash-chain proof", check_broadcast_hash_chain),
    ("5. [Req 3] op/math/audit logs exist", check_structured_logs_exist),
    ("6. [TEST 5] Every selected client has straggler outcome", check_every_selected_client_has_straggler_outcome),
    ("7. [TEST 6] Every aggregation has a math log entry", check_aggregation_has_math_log),
    ("8. [TEST 7] D_i static across rounds", check_diversity_is_static),
    ("9. [TEST 9] No duplicate client contribution", check_no_duplicate_contribution),
    ("10. [Req 4/5, TEST 10] Federated subgroup snapshot per round", check_federated_subgroup_snapshot),
    ("11. Subgroup NaN-guarding + fairness gap (unit)", check_subgroup_metrics_nan_guarding),
    ("12. Confusion-count aggregation is exact (unit)", check_confusion_count_aggregation_exact),
    ("13. [Req 7] Graphs generated from saved logs", check_plots_generated),
    ("14. [Req 11] Reproducibility record in audit log", check_reproducibility_record),
    ("15. [Req 4 eval-side] evaluate.py subgroup CSV integration", check_evaluate_subgroup_integration),
]


def main() -> int:
    # Defensive: every check above passes explicit temp paths for
    # checkpoints/logs/results, but guard against any accidental fallback
    # to cwd defaults polluting the repository, same as
    # test_phase4_runtime.py.
    import shutil
    pre_existing = {name for name in ("checkpoints", "logs", "results") if os.path.isdir(os.path.join(_REPO_ROOT, name))}

    for name, fn in CHECKS:
        run_check(name, fn)

    for name in ("checkpoints", "logs", "results"):
        path = os.path.join(_REPO_ROOT, name)
        if name not in pre_existing and os.path.isdir(path):
            shutil.rmtree(path)

    print("\n" + "=" * 78)
    print("PHASE 5 VERIFICATION RESULTS")
    print("=" * 78)
    for result in RESULTS:
        marker = {"PASS": "[PASS]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}[result.status]
        print(f"{marker} {result.name}")
        if result.detail:
            print(f"       {result.detail}")
    print("=" * 78)

    n_pass = sum(1 for r in RESULTS if r.status == "PASS")
    n_fail = sum(1 for r in RESULTS if r.status == "FAIL")
    n_skip = sum(1 for r in RESULTS if r.status == "SKIP")
    print(f"{n_pass} passed, {n_fail} failed, {n_skip} skipped")
    if "run" in _RUN_STATE:
        print(f"\nFixture artifacts left at: {_RUN_STATE['tmp_root']}")
    print()

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
