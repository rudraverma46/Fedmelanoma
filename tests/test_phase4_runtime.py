"""
tests/test_phase4_runtime.py
=============================
Phase 4 — Flower runtime wiring verification.

This script proves the Phase 4 deliverables listed in the task brief. It
is a plain, dependency-light script (no pytest requirement) that runs a
series of independent checks and prints a PASS/FAIL/SKIP summary.

IMPORTANT: synthetic data used here (a handful of temp-directory CSV
rows, tiny random tensors, in-memory fake Flower ClientProxy objects)
exists ONLY in this file. No fake dataset is written into the repository,
and no production module (hospital_registry.py, client/client.py,
server/server.py, server/strategy.py) is altered to accommodate it --
every check below calls the real, unmodified production code paths.

Run with:  python tests/test_phase4_runtime.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from typing import Callable, List, Tuple

# Make the repo root importable regardless of the invocation directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# server/straggler.py and server/selector.py (unmodified research modules)
# print Unicode status glyphs (e.g. U+27F3) as part of their documented,
# already-verified logging. On a Windows console whose default codepage
# is not UTF-8, those prints raise UnicodeEncodeError -- a harness/console
# environment limitation, not a defect in the research code. Reconfigure
# this test process's own stdout/stderr only; no production file is touched.
for _stream in (sys.stdout, sys.stderr):
    if getattr(_stream, "encoding", "").lower() != "utf-8":
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

import numpy as np
import pandas as pd


# ===========================================================================
# Result bookkeeping
# ===========================================================================

class CheckResult:
    def __init__(self, name: str, status: str, detail: str = "") -> None:
        self.name = name
        self.status = status  # "PASS" | "FAIL" | "SKIP"
        self.detail = detail


RESULTS: List[CheckResult] = []


def run_check(name: str, fn: Callable[[], str]) -> None:
    """Run one check; fn returns a short detail string on success, raises on failure."""
    try:
        detail = fn() or ""
        RESULTS.append(CheckResult(name, "PASS", detail))
    except _Skip as exc:
        RESULTS.append(CheckResult(name, "SKIP", str(exc)))
    except Exception as exc:  # noqa: BLE001 -- verification harness, want every failure
        detail = f"{type(exc).__name__}: {exc}"
        RESULTS.append(CheckResult(name, "FAIL", detail))
        traceback.print_exc()


class _Skip(Exception):
    """Raised by a check to mark itself SKIP (e.g. no network for pretrained weights)."""


# ===========================================================================
# Synthetic hospital_partitions/ builder (verification-only)
# ===========================================================================

from client.dataset import MelanomaDataset  # noqa: E402  (read-only: category lists)


def _make_synthetic_partitions(root: str, n_hospitals: int = 3, rows_per_hospital: int = 12) -> None:
    """
    Build a tiny, fake hospital_partitions/ tree under *root* purely for
    this test run. Column names match exactly what client/dataset.py and
    hospital_registry.py expect (fitzpatrick_skin_type, image_type,
    diagnosis_confirm_type, diagnosis_1) -- everything else about the real
    dataset (images, real distributions, real hospital names) is
    irrelevant to what Phase 4's runtime wiring needs to prove.
    """
    rng = np.random.default_rng(42)
    for h in range(n_hospitals):
        hosp_dir = os.path.join(root, f"SyntheticHospital_{h}")
        os.makedirs(hosp_dir, exist_ok=True)
        rows = {
            "isic_id": [f"ISIC_{h}_{i:04d}" for i in range(rows_per_hospital)],
            "diagnosis_1": rng.choice(["benign", "malignant"], size=rows_per_hospital, p=[0.9, 0.1]),
            "fitzpatrick_skin_type": rng.choice(MelanomaDataset._SKIN_TYPES, size=rows_per_hospital),
            "image_type": rng.choice(MelanomaDataset._IMAGE_TYPES, size=rows_per_hospital),
            "diagnosis_confirm_type": rng.choice(MelanomaDataset._CONFIRM_TYPES, size=rows_per_hospital),
        }
        pd.DataFrame(rows).to_csv(os.path.join(hosp_dir, "metadata_train.csv"), index=False)


# ===========================================================================
# Check 1 -- imports succeed with NO dataset present
# ===========================================================================

def check_imports_without_dataset() -> str:
    assert not os.path.isdir(os.path.join(_REPO_ROOT, "hospital_partitions")), (
        "This check assumes hospital_partitions/ is absent from the repo, "
        "per the task's stated dataset status."
    )
    import hospital_registry  # noqa: F401
    import client.client  # noqa: F401
    import server.server  # noqa: F401
    import server.strategy  # noqa: F401
    import models.model  # noqa: F401
    return "hospital_registry, client.client, server.server, server.strategy, models.model all imported cleanly"


# ===========================================================================
# Check 2 -- discover_hospitals() on an absent / empty directory
# ===========================================================================

def check_discover_hospitals_absent() -> str:
    from hospital_registry import discover_hospitals

    absent = discover_hospitals(os.path.join(_REPO_ROOT, "hospital_partitions"))
    assert absent == [], f"expected [] for absent directory, got {absent}"

    with tempfile.TemporaryDirectory() as tmp:
        empty = discover_hospitals(tmp)
        assert empty == [], f"expected [] for empty directory, got {empty}"

    return "discover_hospitals() returns [] for both absent and empty roots (no crash)"


# ===========================================================================
# Check 3 -- dynamic discovery + registry construction against synthetic data
# ===========================================================================

def check_hospital_registry_synthetic() -> str:
    from hospital_registry import build_hospital_registry, discover_hospitals

    with tempfile.TemporaryDirectory() as tmp:
        _make_synthetic_partitions(tmp, n_hospitals=4, rows_per_hospital=10)

        names = discover_hospitals(tmp)
        assert names == sorted(names), "discovery must be sorted/deterministic"
        assert len(names) == 4, f"expected 4 hospitals, found {names}"
        assert "hospital_partitions" not in tmp  # sanity: this is the synthetic tmp root, not the repo

        hosp_names, records, g_fitz, g_mod, g_conf = build_hospital_registry(tmp)
        assert hosp_names == names
        assert len(records) == 4

        for rec in records:
            assert rec["n_samples"] == 10
            assert rec["tier"] in ("reliable", "moderate", "high_straggler")
            for key in ("fitzpatrick_dist", "modality_dist", "confirmation_dist"):
                dist = rec[key]
                assert abs(sum(dist) - 1.0) < 1e-6, f"{key} does not sum to 1: {dist}"

        for g in (g_fitz, g_mod, g_conf):
            assert abs(sum(g) - 1.0) < 1e-6, f"global distribution does not sum to 1: {g}"

    return f"discovered {len(names)} synthetic hospitals; every per-hospital and global distribution sums to 1.0"


# ===========================================================================
# Check 4 -- pyproject.toml declares the real components/federation Flower's
#            own `flwr run` CLI reads (not a placeholder / not FedAvg)
# ===========================================================================

def check_pyproject_flwr_config() -> str:
    import tomllib

    pyproject_path = os.path.join(_REPO_ROOT, "pyproject.toml")
    assert os.path.isfile(pyproject_path), "pyproject.toml is missing from the repo root"

    with open(pyproject_path, "rb") as fh:
        config = tomllib.load(fh)

    flwr_app = config["tool"]["flwr"]["app"]
    components = flwr_app["components"]
    assert components["serverapp"] == "server.server:app", components
    assert components["clientapp"] == "client.client:app", components

    federations = config["tool"]["flwr"]["federations"]
    default_name = federations["default"]
    assert "options" in federations[default_name]
    assert "num-supernodes" in federations[default_name]["options"]

    run_config = flwr_app["config"]
    for key in ("num-server-rounds", "clients-per-round", "local-epochs", "partitions-root"):
        assert key in run_config, f"[tool.flwr.app.config] missing '{key}'"

    return (
        f"pyproject.toml wires serverapp={components['serverapp']!r}, "
        f"clientapp={components['clientapp']!r}, default federation "
        f"{default_name!r} with num-supernodes="
        f"{federations[default_name]['options']['num-supernodes']}"
    )


# ===========================================================================
# Check 5 -- ServerApp builds without invoking server_fn / needing dataset
# ===========================================================================

def check_serverapp_builds() -> str:
    from flwr.server import ServerApp
    from server.server import app as server_app

    assert isinstance(server_app, ServerApp), f"expected ServerApp, got {type(server_app)}"
    return "server.server.app is a flwr.server.ServerApp instance"


# ===========================================================================
# Check 6 -- ClientApp builds without invoking client_fn / needing dataset
# ===========================================================================

def check_clientapp_builds() -> str:
    from flwr.client import ClientApp
    from client.client import app as client_app

    assert isinstance(client_app, ClientApp), f"expected ClientApp, got {type(client_app)}"
    return "client.client.app is a flwr.client.ClientApp instance"


# ===========================================================================
# Check 7 -- FedMelanomaStrategy initializes from synthetic HospitalConfigs
# ===========================================================================

def check_strategy_initializes() -> str:
    from server.strategy import FedMelanomaStrategy, HospitalConfig

    configs = [
        HospitalConfig(
            n_samples=100 + 10 * i,
            tier=["reliable", "moderate", "high_straggler"][i % 3],
            latency_mu=1.39,
            latency_sigma=0.30,
            fitzpatrick_dist=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1],
            modality_dist=[0.8, 0.15, 0.05],
            confirmation_dist=[0.2, 0.2, 0.3, 0.3],
        )
        for i in range(5)
    ]
    global_fitz = [1 / 6] * 6
    global_mod = [0.8, 0.15, 0.05]
    global_conf = [0.25, 0.25, 0.25, 0.25]

    strategy = FedMelanomaStrategy(
        hospital_configs=configs,
        global_fitzpatrick=global_fitz,
        global_modality=global_mod,
        global_confirmation=global_conf,
        n_select=3,
    )

    assert len(strategy._clients) == 5
    for client in strategy._clients.values():
        assert client.R == 1.0, "R_i must initialise to 1.0 per Blueprint Section 3.5"
        assert client.N_eff == 0.0, "N_eff must initialise to 0.0 per Blueprint Section 3.6"
        assert 0.0 <= client.D <= 1.0, f"D_i out of [0,1]: {client.D}"

    return f"FedMelanomaStrategy initialised with {len(strategy._clients)} clients; R=1.0, N_eff=0.0, D in [0,1] for all"


# ===========================================================================
# Check 8 -- global model initializes (network-dependent: pretrained EfficientNet-B3)
# ===========================================================================

_MODEL_CACHE: dict = {}


def check_model_initializes() -> str:
    try:
        import torch  # noqa: F401
        from models.model import FedMelanomaModel
    except Exception as exc:  # pragma: no cover
        raise _Skip(f"torch/timm not importable in this environment: {exc}") from exc

    try:
        model = FedMelanomaModel()
    except Exception as exc:
        raise _Skip(
            f"FedMelanomaModel() could not be constructed (likely no network access "
            f"to download pretrained EfficientNet-B3 weights): {exc}"
        ) from exc

    _MODEL_CACHE["model"] = model
    n_params = sum(p.numel() for p in model.parameters())
    return f"FedMelanomaModel() constructed successfully ({n_params:,} parameters)"


# ===========================================================================
# Check 9 -- parameter serialization round-trip (Flower NDArrays <-> Parameters)
# ===========================================================================

def check_parameter_serialization() -> str:
    if "model" not in _MODEL_CACHE:
        raise _Skip("depends on check_model_initializes, which was skipped")

    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

    model = _MODEL_CACHE["model"]
    original = [v.detach().cpu().numpy().copy() for v in model.state_dict().values()]

    parameters = ndarrays_to_parameters(original)
    restored = parameters_to_ndarrays(parameters)

    assert len(restored) == len(original)
    for a, b in zip(original, restored):
        assert a.shape == b.shape
        assert np.allclose(a, b)

    return f"round-tripped {len(original)} parameter tensors through ndarrays_to_parameters/parameters_to_ndarrays byte-for-byte"


# ===========================================================================
# Check 10 -- FedMelanomaClient initializes and runs one real fit()/evaluate() round
# ===========================================================================

def check_client_initializes_and_runs() -> str:
    if "model" not in _MODEL_CACHE:
        raise _Skip("depends on check_model_initializes, which was skipped")

    import torch
    from torch.utils.data import DataLoader, Dataset

    from client.client import FedMelanomaClient

    class _FakeMelanomaDataset(Dataset):
        """
        Minimal stand-in exposing exactly the surface FedMelanomaClient /
        local_trainer touch: __len__, __getitem__ -> (image, metadata,
        label), and a `.df["diagnosis_1"]` column for local class-weight
        computation. Random tensors only -- no repo dataset.py code path
        is modified or bypassed; this simply avoids needing real images.
        """

        def __init__(self, n: int) -> None:
            labels = ["benign"] * (n - 1) + ["malignant"]
            self.df = pd.DataFrame({"diagnosis_1": labels})
            self._n = n

        def __len__(self) -> int:
            return self._n

        def __getitem__(self, idx: int):
            image = torch.randn(3, 224, 224, dtype=torch.float32)
            metadata = torch.zeros(13, dtype=torch.float32)
            label = torch.tensor(1 if idx == self._n - 1 else 0, dtype=torch.long)
            return image, metadata, label

    device = torch.device("cpu")
    train_loader = DataLoader(_FakeMelanomaDataset(4), batch_size=2, shuffle=True, num_workers=0)
    test_loader = DataLoader(_FakeMelanomaDataset(4), batch_size=2, shuffle=False, num_workers=0)

    client = FedMelanomaClient(
        model=_MODEL_CACHE["model"],
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        local_epochs=1,
    )

    initial_params = client.get_parameters(config={})
    assert isinstance(initial_params, list) and len(initial_params) > 0

    updated_params, num_examples, fit_metrics = client.fit(initial_params, config={"local_epochs": 1})
    assert num_examples == 4
    assert "train_loss" in fit_metrics

    loss, eval_examples, eval_metrics = client.evaluate(updated_params, config={})
    assert eval_examples == 4
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert key in eval_metrics

    return (
        "FedMelanomaClient initialised on synthetic tensors and completed one real "
        f"fit() + evaluate() round (train_loss={fit_metrics['train_loss']:.4f}, "
        f"eval_loss={loss:.4f})"
    )


# ===========================================================================
# Check 11 -- FedMelanomaStrategy round-trip via Flower's own lifecycle
#             (configure_fit -> aggregate_fit -> configure_evaluate ->
#              aggregate_evaluate), driven with in-memory fake ClientProxy
#              objects instead of a full Ray/gRPC simulation.
# ===========================================================================

def check_strategy_round_lifecycle() -> str:
    from flwr.common import (
        Code,
        EvaluateRes,
        FitRes,
        Status,
        ndarrays_to_parameters,
        parameters_to_ndarrays,
    )
    from flwr.server.client_manager import SimpleClientManager
    from flwr.server.client_proxy import ClientProxy

    from server.strategy import FedMelanomaStrategy, HospitalConfig
    from monitoring.experiment_logging import ExperimentLogger

    rng = np.random.default_rng(7)

    class FakeClientProxy(ClientProxy):
        """In-memory ClientProxy stand-in for Strategy-level verification only."""

        def __init__(self, cid: str, n_examples: int) -> None:
            super().__init__(cid)
            self.node_id = int(cid)
            self.n_examples = n_examples

        def get_properties(self, ins, timeout, group_id):
            return None  # not called by FedMelanomaStrategy

        def get_parameters(self, ins, timeout, group_id):
            raise NotImplementedError  # not called by FedMelanomaStrategy

        def fit(self, ins, timeout, group_id):
            arrays = parameters_to_ndarrays(ins.parameters)
            perturbed = [
                arr + rng.normal(scale=0.01, size=arr.shape).astype(arr.dtype) for arr in arrays
            ]
            return FitRes(
                status=Status(Code.OK, ""),
                parameters=ndarrays_to_parameters(perturbed),
                num_examples=self.n_examples,
                metrics={"train_loss": float(rng.random())},
            )

        def evaluate(self, ins, timeout, group_id):
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
                },
            )

        def reconnect(self, ins, timeout, group_id):
            raise NotImplementedError

    n_hospitals = 6
    configs = [
        HospitalConfig(
            n_samples=50 + 5 * i,
            tier=["reliable", "moderate", "high_straggler"][i % 3],
            latency_mu=1.39,
            latency_sigma=0.30,
            fitzpatrick_dist=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1],
            modality_dist=[0.8, 0.15, 0.05],
            confirmation_dist=[0.2, 0.2, 0.3, 0.3],
        )
        for i in range(n_hospitals)
    ]
    # Phase 5 added optional checkpoint_dir/results_dir/experiment_logger
    # params to FedMelanomaStrategy, defaulting to "checkpoints"/"results"/
    # "logs" under the CURRENT WORKING DIRECTORY when omitted -- correct
    # for a real run, but this is a verification harness, so point them at
    # a throwaway temp dir to keep this check side-effect-free against the
    # repo (never writes checkpoints/logs/results into the repository).
    tmp_dir = tempfile.mkdtemp(prefix="fedmelanoma_phase4_check11_")
    strategy = FedMelanomaStrategy(
        hospital_configs=configs,
        global_fitzpatrick=[1 / 6] * 6,
        global_modality=[0.8, 0.15, 0.05],
        global_confirmation=[0.25, 0.25, 0.25, 0.25],
        n_select=3,
        # Small, architecture-agnostic initial parameters -- FedMelanomaStrategy's
        # aggregate_fit only ever does elementwise ndarray arithmetic, so this
        # exercises the exact same code path as real model weights would,
        # without needing FedMelanomaModel() / network access.
        initial_parameters=[np.zeros((4, 3), dtype=np.float32), np.zeros((4,), dtype=np.float32)],
        rng=np.random.default_rng(0),
        checkpoint_dir=os.path.join(tmp_dir, "checkpoints"),
        results_dir=os.path.join(tmp_dir, "results"),
        experiment_logger=ExperimentLogger("phase4-check11", logs_root=os.path.join(tmp_dir, "logs")),
    )

    client_manager = SimpleClientManager()
    for cid in range(n_hospitals):
        client_manager.register(FakeClientProxy(str(cid), n_examples=configs[cid].n_samples))

    parameters = strategy.initialize_parameters(client_manager)
    assert parameters is not None

    aggregated_metrics_per_round = []
    for server_round in range(1, 3):  # 2 rounds: also exercises the straggler-buffer path
        fit_instructions = strategy.configure_fit(server_round, parameters, client_manager)
        assert 1 <= len(fit_instructions) <= 3, f"expected up to n_select=3 clients, got {len(fit_instructions)}"

        fit_results = [
            (proxy, proxy.fit(ins, timeout=None, group_id=None)) for proxy, ins in fit_instructions
        ]
        parameters, fit_metrics = strategy.aggregate_fit(server_round, fit_results, [])
        assert parameters is not None, "aggregate_fit must return updated parameters"
        assert "fedmelanoma_clients_aggregated" in fit_metrics
        aggregated_metrics_per_round.append(fit_metrics)

        eval_instructions = strategy.configure_evaluate(server_round, parameters, client_manager)
        eval_results = [
            (proxy, proxy.evaluate(ins, timeout=None, group_id=None)) for proxy, ins in eval_instructions
        ]
        loss, eval_metrics = strategy.aggregate_evaluate(server_round, eval_results, [])
        assert loss is not None or not eval_results

    return (
        f"ran {len(aggregated_metrics_per_round)} full rounds through "
        "configure_fit -> aggregate_fit -> configure_evaluate -> aggregate_evaluate; "
        f"round-1 clients_aggregated={aggregated_metrics_per_round[0]['fedmelanoma_clients_aggregated']}"
    )


# ===========================================================================
# Main
# ===========================================================================

CHECKS: List[Tuple[str, Callable[[], str]]] = [
    ("1. Imports succeed without dataset present", check_imports_without_dataset),
    ("2. discover_hospitals() handles absent/empty dirs", check_discover_hospitals_absent),
    ("3. Hospital registry builds from synthetic partitions", check_hospital_registry_synthetic),
    ("4. pyproject.toml wires FedMelanomaStrategy/Client (not FedAvg)", check_pyproject_flwr_config),
    ("5. ServerApp builds (no server_fn invocation)", check_serverapp_builds),
    ("6. ClientApp builds (no client_fn invocation)", check_clientapp_builds),
    ("7. FedMelanomaStrategy initializes", check_strategy_initializes),
    ("8. Global model initializes (FedMelanomaModel)", check_model_initializes),
    ("9. Parameter serialization round-trip", check_parameter_serialization),
    ("10. FedMelanomaClient initializes + runs fit/evaluate", check_client_initializes_and_runs),
    ("11. Strategy round-trip via Flower lifecycle callbacks", check_strategy_round_lifecycle),
]


def main() -> int:
    # Phase 5's FedMelanomaStrategy defaults to writing checkpoints/logs/
    # results under the current working directory when no explicit path is
    # passed (correct for a real run). Most checks below pass explicit temp
    # paths, but check 7 deliberately constructs a Strategy with NO Phase 5
    # args (to prove the defaults themselves work) -- clean up whatever
    # that creates in the repo root afterwards so this verification script
    # stays side-effect-free against the repository.
    import shutil
    pre_existing = {name for name in ("checkpoints", "logs", "results") if os.path.isdir(os.path.join(_REPO_ROOT, name))}

    for name, fn in CHECKS:
        run_check(name, fn)

    for name in ("checkpoints", "logs", "results"):
        path = os.path.join(_REPO_ROOT, name)
        if name not in pre_existing and os.path.isdir(path):
            shutil.rmtree(path)

    print("\n" + "=" * 78)
    print("PHASE 4 RUNTIME VERIFICATION RESULTS")
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
    print(f"{n_pass} passed, {n_fail} failed, {n_skip} skipped\n")

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
