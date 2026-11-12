"""Test 0: this experiment differs from resnet50_fedavg_textbook only in the client-side proximal term."""
import ast
import hashlib
import os

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.abspath(os.path.join(ROOT, "..", "resnet50_fedavg_textbook"))
CHANGED = {"client/client.py", "client/local_trainer.py", "client/proximal.py", "runtime/in_process.py", "tools/run_verification.py",
           "tests/test_00_reference_integrity.py", "tests/test_05_fedprox.py", "tests/test_06_fedprox_real_round.py"}


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def files():
    out = []
    for folder in ("client", "evaluation", "models", "monitoring", "runtime", "server", "tests", "tools"):
        for base, _, names in os.walk(os.path.join(ROOT, folder)):
            if "__pycache__" not in base:
                out += [os.path.relpath(os.path.join(base, n), ROOT).replace(os.sep, "/") for n in names if n.endswith(".py")]
    return sorted(set(out + ["emd.py", "evaluate.py", "hospital_registry.py"]))


@pytest.mark.parametrize("rel", [f for f in files() if f not in CHANGED])
def test_file_identical_to_source_experiment(rel):
    assert sha(os.path.join(ROOT, rel)) == sha(os.path.join(SOURCE, rel))


def calls(path):
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    return {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}


def test_strategy_calls_no_selection_or_straggler_logic():
    used = calls(os.path.join(ROOT, "server", "strategy.py"))
    for name in ("boltzmann_selection", "compute_score", "update_client_state", "simulate_response", "manage_window", "get_buffer_state", "score_breakdown"):
        assert name not in used, name


def test_config(cfg):
    assert cfg["experiment"] == {"run_id": "resnet50-fedprox-textbook-mu0.01-seed42", "seed": 42}
    assert cfg["fedprox"] == {"mu": 0.01}
    with open(os.path.join(SOURCE, "configs", "experiment_config.yaml"), encoding="utf-8") as fh:
        source = yaml.safe_load(fh)
    for key in ("partitions_root", "cache_dir"):  # load_config resolves these to absolute paths
        assert os.path.normpath(cfg["data"][key]) == os.path.normpath(os.path.join(os.path.dirname(SOURCE), os.path.basename(SOURCE), source["data"][key].split("#")[0].strip()))
    for section in ("federation", "local_training", "model", "runtime", "reference_constants"):
        assert cfg[section] == source[section], section
    assert os.path.isdir(cfg["data"]["cache_dir"])
