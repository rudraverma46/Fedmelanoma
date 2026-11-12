"""Test 0: the reference code is unchanged and the configuration matches what it executes."""
import difflib
import hashlib
import json
import os
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLONE = os.path.abspath(os.path.join(ROOT, "..", "..", "FedMelanoma_repo"))
with open(os.path.join(ROOT, "REFERENCE_FILES.json"), encoding="utf-8") as fh:
    RECORD = json.load(fh)


def git_show(path: str) -> bytes:
    return subprocess.run(["git", "-C", CLONE, "show", f"{RECORD['commit']}:{path}"], check=True, capture_output=True).stdout


@pytest.mark.parametrize("rel", sorted(RECORD["files"]))
def test_file_byte_identical_to_reference(rel):
    with open(os.path.join(ROOT, rel), "rb") as fh:
        blob = fh.read()
    assert hashlib.sha256(blob).hexdigest() == RECORD["files"][rel]
    assert blob == git_show(rel)


def test_reference_clone_untouched():
    assert subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip() == RECORD["commit"]
    assert subprocess.run(["git", "-C", CLONE, "status", "--porcelain"], check=True, capture_output=True, text=True).stdout.strip() == ""


def test_model_file_is_byte_identical_to_reference():
    with open(os.path.join(ROOT, "models", "model.py"), "rb") as fh:
        assert fh.read() == git_show("models/model.py")


def test_config_matches_reference_constants(cfg):
    import tomllib
    from client import client as ref_client
    from hospital_registry import _TIER_CYCLE, _TIER_LATENCY
    from server import aggregation, selector, straggler

    pyproject = tomllib.loads(git_show("pyproject.toml").decode())
    app = pyproject["tool"]["flwr"]["app"]["config"]
    assert cfg["federation"]["num_server_rounds"] == app["num-server-rounds"] == 50
    assert cfg["federation"]["clients_per_round"] == app["clients-per-round"] == 6
    assert cfg["federation"]["local_epochs"] == app["local-epochs"] == 5
    assert cfg["federation"]["num_clients"] == pyproject["tool"]["flwr"]["federations"]["local-simulation"]["options"]["num-supernodes"]

    local = cfg["local_training"]
    assert local["batch_size"] == ref_client.DEFAULT_BATCH_SIZE
    assert local["learning_rate"] == ref_client.DEFAULT_LEARNING_RATE
    assert local["weight_decay"] == ref_client.DEFAULT_WEIGHT_DECAY
    assert local["focal_gamma"] == ref_client.DEFAULT_FOCAL_GAMMA

    const = cfg["reference_constants"]
    assert const["selector"] == {"rho": selector.RHO_DEFAULT, "delta": selector.DELTA_DEFAULT, "lambda": selector.LAMBDA_DEFAULT,
                                 "gamma": selector.GAMMA_DEFAULT, "tau": selector.TAU_DEFAULT}
    assert const["straggler"] == {"s_max": straggler.S_MAX_DEFAULT, "quorum_frac": straggler.QUORUM_FRAC_DEFAULT,
                                  "mad_k": straggler.MAD_K_DEFAULT, "w_min": straggler.W_MIN_DEFAULT,
                                  "delta": straggler.DELTA_DEFAULT, "history_len": straggler.WINDOW_HISTORY_LEN}
    assert const["aggregation"] == {"delta": aggregation.DELTA, "gamma": aggregation.GAMMA}
    assert straggler.BETA_PARAMS == {"reliable": (19.0, 1.0), "moderate": (7.0, 3.0), "high_straggler": (2.0, 3.0)}
    assert _TIER_CYCLE == ("reliable", "moderate", "high_straggler")
    assert _TIER_LATENCY == {"reliable": (1.39, 0.30), "moderate": (2.77, 0.50), "high_straggler": (4.09, 0.80)}
