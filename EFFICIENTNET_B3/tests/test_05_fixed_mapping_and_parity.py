"""Test 20-23: the fixed mapping, the hospital properties, the pretrained model, and parity with ResNet-50.

Run before any training. test_06_real_run_mapping.py checks the same mapping from a real run's logs.
"""
import hashlib
import json
import os

import pandas as pd
import pytest
import timm
import torch
import yaml

from hospital_registry import build_hospital_registry
from models.model import FedMelanomaModel
from runtime.in_process import build_federation
from tests_expected import EXPECTED

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESNET = os.path.abspath(os.path.join(ROOT, "..", "RESNET 50_ fed novelty", "RESNET50_EFFICIENTNET_B3_EQUIVALENT"))

# Files allowed to differ from the ResNet-50 experiment, and why.
ALLOWED_CODE_DIFFERENCES = {
    "models/model.py",            # the backbone: the intended difference
    "tools/run_supervised.py",    # default run id only
    "tools/run_verification.py",  # runs the extra mapping test
    "tools/audit_implementations.py",  # new here: the audit itself
}


def resnet_plan():
    with open(os.path.join(RESNET, "results", "schedule", "planned_schedule.json"), encoding="utf-8") as fh:
        return json.load(fh)


def digest(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def test_20_hospital_order_is_the_fixed_table(hospitals):
    assert list(hospitals) == EXPECTED


def test_20_each_client_loads_its_own_hospital(cfg, hospitals):
    manager = build_federation(cfg, hospitals, torch.nn.Linear(1, 1), torch.device("cpu"), lambda record: None)
    proxies = manager.all()
    assert sorted(proxies) == [f"{i:02d}" for i in range(10)]
    root = cfg["data"]["partitions_root"]
    for client_id, hospital in enumerate(EXPECTED):
        proxy = proxies[f"{client_id:02d}"]
        assert proxy.client_id == client_id and proxy.hospital == hospital
        numpy_client = proxy.client.numpy_client
        for split, loader in (("train", numpy_client.train_loader), ("test", numpy_client.test_loader)):
            on_disk = pd.read_csv(os.path.join(root, hospital, f"metadata_{split}.csv"), usecols=["isic_id"], low_memory=False)
            assert list(loader.dataset.df["isic_id"]) == list(on_disk["isic_id"])


def test_21_sample_counts_match_csv_and_resnet_plan(cfg):
    root = cfg["data"]["partitions_root"]
    registry = build_hospital_registry(root)
    plan = {h["client_id"]: h for h in resnet_plan()["hospitals"]}
    assert list(registry[0]) == EXPECTED
    total = 0
    for client_id, hospital in enumerate(EXPECTED):
        rows = len(pd.read_csv(os.path.join(root, hospital, "metadata_train.csv"), usecols=["isic_id"], low_memory=False))
        assert plan[client_id]["hospital"] == hospital and plan[client_id]["n_samples"] == rows
        total += rows
    assert total == 7788


def test_22_tier_diversity_and_schedule_match_resnet(planned):
    resnet = resnet_plan()
    assert planned["hospitals"] == resnet["hospitals"]       # tier, n and D for every client
    assert planned["schedule"] == resnet["schedule"]         # all 50 rounds
    assert planned["seed"] == resnet["seed"] == 42


def test_23_pretrained_weights_are_the_timm_reference():
    model = FedMelanomaModel()
    reference = timm.create_model("efficientnet_b3", pretrained=True, num_classes=0)
    assert model.backbone.pretrained_cfg["hf_hub_id"] == "timm/efficientnet_b3.ra2_in1k"
    ours, theirs = model.backbone.state_dict(), reference.state_dict()
    assert list(ours) == list(theirs)
    assert all(torch.equal(ours[k], theirs[k]) for k in ours)
    assert model.backbone.num_features == 1536 and model.classifier.in_features == 1568


def test_23_code_matches_resnet_experiment():
    different = []
    for top in ("client", "server", "monitoring", "models", "runtime", "evaluation", "tools"):
        for folder, _, files in os.walk(os.path.join(ROOT, top)):
            for name in files:
                if name.endswith(".py"):
                    rel = os.path.relpath(os.path.join(folder, name), ROOT).replace(os.sep, "/")
                    other = os.path.join(RESNET, rel)
                    if not os.path.exists(other) or digest(os.path.join(ROOT, rel)) != digest(other):
                        different.append(rel)
    for rel in ("emd.py", "hospital_registry.py", "evaluate.py"):
        if digest(os.path.join(ROOT, rel)) != digest(os.path.join(RESNET, rel)):
            different.append(rel)
    assert set(different) <= ALLOWED_CODE_DIFFERENCES and "models/model.py" in different


def test_23_config_differs_only_in_labels_and_paths():
    with open(os.path.join(ROOT, "configs", "experiment_config.yaml"), encoding="utf-8") as fh:
        ours = yaml.safe_load(fh)
    with open(os.path.join(RESNET, "configs", "experiment_config.yaml"), encoding="utf-8") as fh:
        theirs = yaml.safe_load(fh)
    assert os.path.normpath(os.path.join(ROOT, ours["data"]["partitions_root"])) == os.path.normpath(os.path.join(RESNET, theirs["data"]["partitions_root"]))
    for section in ("federation", "local_training", "runtime", "reference_constants"):
        assert ours[section] == theirs[section], section
    assert ours["experiment"]["seed"] == theirs["experiment"]["seed"] == 42
    assert ours["model"]["dropout"] == theirs["model"]["dropout"] and ours["model"]["metadata_dim"] == theirs["model"]["metadata_dim"]
    assert (ours["model"]["backbone"], ours["model"]["feature_dim"], ours["model"]["fusion_dim"]) == ("efficientnet_b3", 1536, 1568)


@pytest.mark.parametrize("name", ["requirements-lock.txt"])
def test_23_same_locked_environment(name):
    assert digest(os.path.join(ROOT, name)) == digest(os.path.join(RESNET, name))
