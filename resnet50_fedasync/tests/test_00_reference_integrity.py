"""Test 0: identical experimental conditions and no Novelty / diversity code.

* Model, data, cache, loss, evaluation and monitoring files are byte-identical to resnet50_fedavg_textbook.
* The client files are byte-identical to resnet50_fedprox_textbook (textbook FedAvg's client plus the
  optional proximal term, which is FedAsync's worker regulariser).
* The config sections shared with textbook FedAvg have identical values.
* No selector, straggler window, diversity (EMD) or aggregation module from the Novelty code is present,
  and importing the runner imports none of them.
"""
import hashlib
import os
import re
import subprocess
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEDAVG = os.path.abspath(os.path.join(ROOT, "..", "resnet50_fedavg_textbook"))
FEDPROX = os.path.abspath(os.path.join(ROOT, "..", "resnet50_fedprox_textbook"))
SAME_AS_FEDAVG = ["models/model.py", "client/dataset.py", "client/focal_loss.py", "runtime/image_cache.py",
                  "runtime/config.py", "runtime/resources.py", "evaluate.py", "evaluation/evaluate_checkpoint.py",
                  "hospital_registry.py", "monitoring/checkpointing.py", "monitoring/experiment_logging.py",
                  "monitoring/param_hash.py", "monitoring/subgroup_metrics.py", "tests/conftest.py"]
SAME_AS_FEDPROX = ["client/client.py", "client/local_trainer.py", "client/proximal.py"]
FORBIDDEN_FILES = ["emd.py", "server/selector.py", "server/straggler.py", "server/aggregation.py", "server/strategy.py"]
FORBIDDEN_MODULES = ["emd", "server.selector", "server.straggler", "server.aggregation", "server.strategy"]


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


@pytest.mark.parametrize("rel", SAME_AS_FEDAVG)
def test_identical_to_textbook_fedavg(rel):
    assert sha(os.path.join(ROOT, rel)) == sha(os.path.join(FEDAVG, rel))


@pytest.mark.parametrize("rel", SAME_AS_FEDPROX)
def test_client_identical_to_textbook_fedprox(rel):
    assert sha(os.path.join(ROOT, rel)) == sha(os.path.join(FEDPROX, rel))


def test_fedprox_client_differs_from_fedavg_client_only_by_the_proximal_term():
    """local_trainer.py: with global_reference None the step is textbook FedAvg's (FedProx folder test)."""
    with open(os.path.join(ROOT, "client", "local_trainer.py"), encoding="utf-8") as fh:
        text = fh.read()
    assert "if global_reference is None:\n            objective = loss" in text
    assert "objective = loss + penalty" in text


def test_shared_config_values(cfg):
    with open(os.path.join(ROOT, "configs", "experiment_config.yaml"), encoding="utf-8") as fh:
        here = yaml.safe_load(fh)
    with open(os.path.join(FEDAVG, "configs", "experiment_config.yaml"), encoding="utf-8") as fh:
        base = yaml.safe_load(fh)
    for section in ("data", "local_training", "model", "runtime", "reference_constants"):
        assert here[section] == base[section], section
    assert here["experiment"]["seed"] == base["experiment"]["seed"] == 42
    assert here["federation"]["local_epochs"] == base["federation"]["local_epochs"] == 5
    assert here["federation"]["num_clients"] == base["federation"]["num_clients"] == 10
    assert here["fedasync"]["total_dispatches"] == base["federation"]["num_server_rounds"] * base["federation"]["clients_per_round"]
    assert here["fedasync"]["concurrency"] == base["federation"]["clients_per_round"]
    assert here["fedasync"] == {"total_dispatches": 300, "concurrency": 6, "scheduler": "uniform_idle", "alpha": 0.6,
                                "staleness_function": "hinge", "staleness_a": 10, "staleness_b": 4, "rho": 0.005,
                                "reject_stale": False, "eval_every_updates": 6, "arrival_time": "novelty_lognormal",
                                "drain_in_flight_at_budget": True}
    assert here["environment"] == {"kind": "novelty_tiers", "failure": "none", "trace_seed": 42}
    assert os.path.isdir(cfg["data"]["cache_dir"])


@pytest.mark.parametrize("rel", FORBIDDEN_FILES)
def test_no_novelty_or_diversity_files(rel):
    assert not os.path.exists(os.path.join(ROOT, rel))


def test_runner_imports_no_novelty_or_diversity_module():
    code = ("import sys, runtime.run_fedasync; bad = [m for m in %r if m in sys.modules]; "
            "print('BAD', bad); sys.exit(1 if bad else 0)") % (FORBIDDEN_MODULES,)
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


def test_server_code_has_no_diversity_terms():
    """Every identifier in the algorithm code (names, attributes, functions, arguments, imports)."""
    import ast
    pattern = re.compile(r"boltzmann|diversity|emd|wasserstein|novelty|reliab", re.IGNORECASE)
    for rel in ("server/fedasync.py", "server/scheduler.py", "runtime/run_fedasync.py", "runtime/clients.py"):
        with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        names = set()
        for node in ast.walk(tree):
            names.update(v for v in (getattr(node, "id", None), getattr(node, "attr", None), getattr(node, "name", None),
                                     getattr(node, "arg", None), getattr(node, "module", None)) if isinstance(v, str))
        assert not [n for n in names if pattern.search(n)], rel
