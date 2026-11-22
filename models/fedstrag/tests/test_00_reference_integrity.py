"""Test 0: identical experimental pipeline to the ResNet-50 NOVELTY run, and no Novelty mechanism.

* Model, data, cache, loss, client, evaluation and monitoring files are byte-identical to the novelty folder.
* sim/straggler.py is byte-identical to the novelty server/straggler.py (the environment, see test_04).
* The config sections shared with the novelty run have identical values; the data paths resolve to the same folders.
* No selector, aggregation, strategy, EMD/diversity module is present, importing the runner imports none of them,
  and the algorithm code contains no diversity / reliability / exposure / quorum / window / Boltzmann identifier.
"""
import ast
import hashlib
import json
import os
import re
import subprocess
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOVELTY = os.path.abspath(os.path.join(ROOT, "..", "RESNET 50_ fed novelty", "RESNET50_NOVELTY MODEL"))
SAME_AS_NOVELTY = ["models/model.py", "client/dataset.py", "client/focal_loss.py", "client/client.py",
                   "client/local_trainer.py", "runtime/image_cache.py", "runtime/config.py",
                   "evaluate.py", "evaluation/evaluate_checkpoint.py", "hospital_registry.py",
                   "monitoring/checkpointing.py", "monitoring/experiment_logging.py", "monitoring/param_hash.py",
                   "monitoring/subgroup_metrics.py", "tests/conftest.py"]
FORBIDDEN_FILES = ["emd.py", "server/selector.py", "server/straggler.py", "server/aggregation.py",
                   "server/strategy.py", "server/server.py", "client/proximal.py"]
FORBIDDEN_MODULES = ["emd", "server.selector", "server.straggler", "server.aggregation", "server.strategy", "server.server"]
ALGORITHM_FILES = ("server/fedstrag.py", "server/protocol.py", "runtime/run_fedstrag.py", "runtime/clients.py",
                   "sim/environment.py")
FORBIDDEN_NAMES = re.compile(r"boltzmann|diversity|emd|wasserstein|novelty_score|reliab|exposure|n_eff|quorum|"
                             r"manage_window|update_window|drain|s_max|current_w_t|mad_k|apply_staleness_weight|select",
                             re.IGNORECASE)


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


with open(os.path.join(ROOT, "configs", "reference_manifest.json"), encoding="utf-8") as _fh:
    MANIFEST = json.load(_fh)   # written by tools/make_reference_manifest.py on the machine holding the novelty folder
HAVE_NOVELTY = os.path.isdir(NOVELTY)


@pytest.mark.parametrize("rel", SAME_AS_NOVELTY)
def test_identical_to_novelty(rel):
    assert sha(os.path.join(ROOT, rel)) == MANIFEST["pipeline_files"][rel]
    if HAVE_NOVELTY:
        assert MANIFEST["pipeline_files"][rel] == sha(os.path.join(NOVELTY, rel))


def test_environment_module_is_the_novelty_straggler_file():
    assert sha(os.path.join(ROOT, "sim", "straggler.py")) == MANIFEST["straggler_py"]
    if HAVE_NOVELTY:
        assert MANIFEST["straggler_py"] == sha(os.path.join(NOVELTY, "server", "straggler.py"))


def test_manifest_covers_every_pipeline_file():
    assert sorted(MANIFEST["pipeline_files"]) == sorted(SAME_AS_NOVELTY)


def test_same_cache_and_split_files_as_novelty(cfg):
    """Pixels, row order and every hospital split CSV: byte-identical to what the novelty run used."""
    cache = cfg["data"]["cache_dir"]
    assert sorted(os.listdir(cache)) == sorted(MANIFEST["cache_files"])
    for name, digest in MANIFEST["cache_files"].items():
        assert sha(os.path.join(cache, name)) == digest, name
    for rel, digest in MANIFEST["split_csvs"].items():
        assert sha(os.path.join(cfg["data"]["partitions_root"], *rel.split("/"))) == digest, rel


def test_shared_config_values():
    with open(os.path.join(ROOT, "configs", "experiment_config.yaml"), encoding="utf-8") as fh:
        here = yaml.safe_load(fh)
    base = MANIFEST["novelty_config"]
    if HAVE_NOVELTY:
        with open(os.path.join(NOVELTY, "configs", "experiment_config.yaml"), encoding="utf-8") as fh:
            live = yaml.safe_load(fh)
        assert live["local_training"] == base["local_training"] and live["model"] == base["model"]
    assert here["local_training"] == base["local_training"]
    assert here["model"] == base["model"]
    for key, value in base["runtime"].items():
        assert here["runtime"][key] == value, key
    assert here["reference_constants"]["evaluation_thresholds"] == base["evaluation_thresholds"]
    assert here["experiment"]["seed"] == base["seed"] == 42
    assert here["federation"]["local_epochs"] == base["local_epochs"] == 5
    assert here["federation"]["num_clients"] == base["num_clients"] == 10
    assert here["fedstrag"]["global_iterations"] == base["num_server_rounds"] == 50


def test_fedstrag_and_environment_settings():
    with open(os.path.join(ROOT, "configs", "experiment_config.yaml"), encoding="utf-8") as fh:
        here = yaml.safe_load(fh)
    assert here["fedstrag"] == {
        "global_iterations": 50, "broadcast_to": "all", "client_queue": "fifo", "wait_time_seconds": 4,
        "minimum_clients": 1, "staleness_weight": "inverse_staleness_plus_one", "sample_weighting": "paper_eq7_W_times_D",
        "empty_iteration": "keep_global_unchanged", "termination": "exclude_pending",
        "arrival_time": "novelty_lognormal", "eval_every_iterations": 1}
    assert here["environment"] == {"kind": "novelty_simulate_response", "failure": "novelty_beta_bernoulli",
                                   "failure_timing": "after_local_training",
                                   "failed_job_duration": "tier_lognormal_child_stream", "trace_seed": 42}


def test_initial_model_file_is_the_novelty_initial_model(cfg):
    """The run starts from the novelty run's exact initial weights (its logged hash), on any PyTorch version."""
    import pickle
    from monitoring.param_hash import hash_ndarrays
    novelty_hash = "330768fe201714e3f26b9973f70fd32cbec63e2bce8b8569194ef65641876f4a"
    assert cfg["initialisation"]["param_hash"] == novelty_hash
    with open(os.path.join(ROOT, cfg["initialisation"]["file"]), "rb") as fh:
        saved = pickle.load(fh)
    assert saved["param_hash"] == novelty_hash == hash_ndarrays(saved["arrays"])
    from models.model import FedMelanomaModel
    model = FedMelanomaModel()
    assert saved["keys"] == list(model.state_dict().keys())
    assert [a.shape for a in saved["arrays"]] == [tuple(v.shape) for v in model.state_dict().values()]


def test_poll_interval_is_exactly_4_seconds_everywhere(cfg):
    from server.fedstrag import MINIMUM_CLIENTS, WAIT_TIME_SECONDS
    assert WAIT_TIME_SECONDS == 4.0 and MINIMUM_CLIENTS == 1
    assert cfg["fedstrag"]["wait_time_seconds"] == 4
    with open(os.path.join(ROOT, "runtime", "run_fedstrag.py"), encoding="utf-8") as fh:
        code = fh.read()
    assert 'fs["wait_time_seconds"], fs["minimum_clients"])' in code
    assert '"wait_time_seconds": 4' in code          # the runner refuses any other value


@pytest.mark.parametrize("rel", FORBIDDEN_FILES)
def test_no_novelty_or_diversity_files(rel):
    assert not os.path.exists(os.path.join(ROOT, rel))


def test_runner_imports_no_novelty_or_diversity_module():
    code = ("import sys, runtime.run_fedstrag; bad = [m for m in %r if m in sys.modules]; "
            "print('BAD', bad); sys.exit(1 if bad else 0)") % (FORBIDDEN_MODULES,)
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


@pytest.mark.parametrize("rel", ALGORITHM_FILES)
def test_algorithm_code_has_no_novelty_identifiers(rel):
    """Every identifier (names, attributes, functions, arguments, imports) in the algorithm code."""
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    names = set()
    for node in ast.walk(tree):
        names.update(v for v in (getattr(node, "id", None), getattr(node, "attr", None), getattr(node, "name", None),
                                 getattr(node, "arg", None), getattr(node, "module", None)) if isinstance(v, str))
    assert not sorted(n for n in names if FORBIDDEN_NAMES.search(n))


def test_environment_calls_only_simulate_response():
    """The only StragglerManager method the code calls is simulate_response (the environment draw)."""
    from sim.straggler import StragglerManager
    methods = {n for n in dir(StragglerManager) if not n.startswith("__")}
    used = set()
    for rel in ALGORITHM_FILES:
        with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        used.update(n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr in methods)
    assert used == {"simulate_response"}
