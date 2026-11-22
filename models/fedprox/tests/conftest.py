import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")


@pytest.fixture(scope="session")
def cfg():
    from runtime.config import load_config
    return load_config()


@pytest.fixture(scope="session")
def hospitals(cfg):
    from hospital_registry import discover_hospitals
    return discover_hospitals(cfg["data"]["partitions_root"])


@pytest.fixture(scope="session")
def planned():
    import json
    with open(os.path.join(ROOT, "results", "schedule", "planned_schedule.json"), encoding="utf-8") as fh:
        return json.load(fh)
