"""Test 0: this experiment differs from resnet50_fedlesscan only in the intended files (the straggler
environment and the delayed-update handling), and the architecture, data, local training and runtime are unchanged."""
import ast
import hashlib
import os

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.abspath(os.path.join(ROOT, "..", "resnet50_fedlesscan"))
TEXTBOOK = os.path.abspath(os.path.join(ROOT, "..", "resnet50_fedavg_textbook"))
NOVELTY = os.path.abspath(os.path.join(ROOT, "..", "RESNET 50_ fed novelty", "RESNET50_NOVELTY MODEL"))
NEW = {"server/environment.py", "tests/test_05_final_audit.py"}
CHANGED = {"client/client.py", "server/strategy.py", "server/server.py", "server/fedlesscan.py", "runtime/run_experiment.py", "runtime/state.py", "runtime/resources.py", "hospital_registry.py", "monitoring/experiment_logging.py", "monitoring/plotting.py", "evaluation/build_report.py", "tools/dry_run_schedule.py", "tools/train_fedlesscan.sh",
           "tests/test_00_reference_integrity.py", "tests/test_02_federation_math.py", "tests/test_03_real_round.py"}
COMMENT_MARKERS = ("server/aggregation.py)", "Strategy (FedAvg or the custom", "Client selection (server/selector.py", "from this file)",
                  "Straggler handling (server/straggler.py", "selector.py, aggregation.py, straggler.py, emd.py are not imported", "at all — this file has no client-selection",
                  "selector.Client(client_id=N)", "all server-side (server/fedlesscan.py", "server/environment.py); this file has none", "No server module is imported here", "client_id N on the server side always refer to", "per-hospital client list, so partition-id N here", "the same hospital.")
TIMING_MARKERS = ("fit_start", "fit_end", "fit_seconds", "resnet50_fedlesscan: FedLesScan clusters", "(fedless/client/client.py:201-205)")


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def files():
    out = []
    for folder in ("client", "evaluation", "models", "monitoring", "runtime", "server", "tests", "tools"):
        for base, _, names in os.walk(os.path.join(ROOT, folder)):
            if "__pycache__" not in base:
                out += [os.path.relpath(os.path.join(base, n), ROOT).replace(os.sep, "/") for n in names if n.endswith(".py")]
    return sorted(set(out + ["evaluate.py", "hospital_registry.py"]))


@pytest.mark.parametrize("rel", [f for f in files() if f not in CHANGED | NEW])
def test_file_identical_to_resnet50_fedlesscan(rel):
    assert sha(os.path.join(ROOT, rel)) == sha(os.path.join(SOURCE, rel))


def test_model_and_training_files_unchanged():
    for rel in ("models/model.py", "client/dataset.py", "client/focal_loss.py", "client/local_trainer.py", "runtime/image_cache.py"):
        assert sha(os.path.join(ROOT, rel)) == sha(os.path.join(TEXTBOOK, rel)), rel


def test_client_change_is_only_the_training_timer():
    def lines(path):
        with open(path, encoding="utf-8") as fh:
            return [l for l in fh.read().splitlines() if not any(m in l for m in TIMING_MARKERS + COMMENT_MARKERS)]
    assert lines(os.path.join(ROOT, "client", "client.py")) == lines(os.path.join(TEXTBOOK, "client", "client.py"))


def calls(path):
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    return {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None)
            for n in ast.walk(tree) if isinstance(n, ast.Call)}


def test_strategy_uses_no_novelty_or_textbook_logic():
    used = calls(os.path.join(ROOT, "server", "strategy.py"))
    for name in ("boltzmann_selection", "compute_score", "update_client_state", "simulate_response", "manage_window",
                 "get_buffer_state", "score_breakdown", "fedavg_baseline_weights", "compute_aggregation_weights", "choice"):
        assert name not in used, name
    for name in ("select", "finish_round", "aggregate", "record_client_success", "store_result", "launch_late", "arrivals",
                 "next_job_index", "draw"):
        assert name in used, name


def test_fedlesscan_module_imports_nothing_from_the_novelty_method():
    with open(os.path.join(ROOT, "server", "fedlesscan.py"), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
               {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m and (m.startswith("server") or m in ("emd",)) for m in imported), imported


def test_novelty_scheduler_modules_are_gone():
    for rel in ("server/selector.py", "server/aggregation.py", "server/straggler.py", "emd.py", "monitoring/straggler_logging.py"):
        assert not os.path.exists(os.path.join(ROOT, rel)), rel
    for rel in ("server/strategy.py", "server/fedlesscan.py", "server/environment.py", "server/server.py", "runtime/state.py",
                "runtime/run_experiment.py", "evaluation/build_report.py"):
        with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module} |                {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert not {m for m in mods if m in ("emd", "server.selector", "server.aggregation", "server.straggler", "monitoring.straggler_logging")}, (rel, mods)


def test_environment_equals_novelty_simulate_response_draw_for_draw():
    """The environment must reproduce the novelty simulator's Beta / Bernoulli / LogNormal draws (same generator, order, parameters)."""
    import importlib.util
    import types
    import numpy as np
    from server.environment import TraceEnvironment, TRACE_STREAM
    spec = importlib.util.spec_from_file_location("novelty_straggler", os.path.join(NOVELTY, "server", "straggler.py"))
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module   # dataclasses look the module up by name
    spec.loader.exec_module(module)
    records = [{"tier": "reliable", "latency_mu": 1.39, "latency_sigma": 0.30}, {"tier": "moderate", "latency_mu": 2.77, "latency_sigma": 0.50},
               {"tier": "high_straggler", "latency_mu": 4.09, "latency_sigma": 0.80}]
    env = TraceEnvironment(42, records)
    for i, rec in enumerate(records):
        manager = module.StragglerManager(rng=np.random.default_rng([42, TRACE_STREAM, i]))
        stub = types.SimpleNamespace(client_id=i, **rec)
        for k in range(200):
            r = manager.simulate_response(stub, k)
            failed, latency = env.draw(i, k)
            assert failed == r.is_failed
            assert (latency == r.response_time) or (failed and latency == float("inf") and r.response_time == float("inf"))


def test_config(cfg):
    assert cfg["experiment"] == {"run_id": "resnet50-fedlesscan-novelty-env-seed42", "seed": 42}
    raw = {}
    for name, folder in (("here", ROOT), ("source", SOURCE)):
        with open(os.path.join(folder, "configs", "experiment_config.yaml"), encoding="utf-8") as fh:
            raw[name] = yaml.safe_load(fh)
    for section in ("data", "federation", "local_training", "model", "runtime", "reference_constants"):
        assert raw["here"][section] == raw["source"][section], section
    assert cfg["fedlesscan"] == {"round_deadline_seconds": 30, "renormalise_aggregation": False,
                                 "allowed_stragglers": 6, "staleness_tolerance": 2}
    assert os.path.isdir(cfg["data"]["cache_dir"])
