"""Test 0: this experiment differs from resnet 50_novelty_fedavg only in the intended files, and the source is unchanged."""
import ast
import hashlib
import os

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.abspath(os.path.join(ROOT, "..", "resnet 50_novelty_fedavg"))
CHANGED = {"server/strategy.py", "configs/experiment_config.yaml", "evaluation/build_report.py", "tools/dry_run_schedule.py", "README.md", ".gitignore",
           "tests/test_00_reference_integrity.py", "tests/test_02_federation_math.py", "tests/test_03_real_round.py"}


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
    assert cfg["experiment"] == {"run_id": "resnet50-fedavg-textbook-seed42", "seed": 42}
    with open(os.path.join(SOURCE, "configs", "experiment_config.yaml"), encoding="utf-8") as fh:
        source = yaml.safe_load(fh)
    for section in ("federation", "local_training", "model", "runtime"):
        assert cfg[section] == source[section], section
    assert (cfg["federation"]["num_clients"], cfg["federation"]["clients_per_round"], cfg["federation"]["num_server_rounds"], cfg["federation"]["local_epochs"]) == (10, 6, 50, 5)
    assert set(cfg["reference_constants"]) == {"evaluation_thresholds"}
    assert os.path.isdir(cfg["data"]["cache_dir"])
