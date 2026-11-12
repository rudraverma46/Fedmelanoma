"""Runs the whole verification sequence and saves the evidence to results/verification/.

  1. tests 0-15 (pytest)
  2. a real run `verify-seed42`: round 1, process exits, a new process resumes round 2
  3. evaluate.py on its round-2 checkpoint
  4. tests 16-18 on those artefacts
  5. test 19: transport equivalence

Exits non-zero if any step fails. `--skip-run` reuses an existing verify-seed42.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "verification")
RUN = "verify-seed42"
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}


def step(name: str, args: list) -> None:
    print(f"== {name}", flush=True)
    start = time.time()
    with open(os.path.join(OUT, f"{name}.log"), "w", encoding="utf-8") as log:
        code = subprocess.call([sys.executable, "-W", "ignore", *args], cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT)
    print(f"   exit {code} in {time.time() - start:.0f} s", flush=True)
    if code != 0:
        raise SystemExit(f"{name} failed; see results/verification/{name}.log")


def pytest_args(files: list, name: str) -> list:
    return ["-m", "pytest", *files, "-v", "-p", "no:cacheprovider", "--tb=short", f"--junitxml={os.path.join(OUT, name + '.xml')}"]


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    step("tests_00_to_15", pytest_args(["tests/test_00_reference_integrity.py", "tests/test_01_model_and_data.py", "tests/test_02_federation_math.py", "tests/test_05_fixed_mapping_and_parity.py"], "tests_00_to_15"))
    if "--skip-run" not in sys.argv:
        for folder in (f"checkpoints/{RUN}", f"results/{RUN}"):
            shutil.rmtree(os.path.join(ROOT, folder), ignore_errors=True)
        for rel in (f"logs/op/{RUN}.jsonl", f"logs/math/{RUN}.jsonl", f"logs/audit/{RUN}.jsonl", f"logs/op/stragglers/{RUN}.jsonl",
                    f"logs/clients/{RUN}.jsonl", f"logs/rounds/{RUN}.jsonl", f"logs/resources/{RUN}.csv"):
            if os.path.exists(os.path.join(ROOT, rel)):
                os.remove(os.path.join(ROOT, rel))
        step("real_round_1", ["-m", "runtime.run_experiment", "--run-id", RUN, "--rounds", "2", "--stop-after", "1"])
        step("real_round_2_after_restart", ["-m", "runtime.run_experiment", "--run-id", RUN, "--rounds", "2", "--resume"])
    step("evaluate_round_2", ["evaluation/evaluate_checkpoint.py", f"checkpoints/{RUN}/global_round_0002.pth", f"results/{RUN}/final_round2"])
    step("tests_16_to_18", pytest_args(["tests/test_03_real_round.py", "tests/test_06_real_run_mapping.py"], "tests_16_to_18"))
    step("test_19_transport", pytest_args(["tests/test_04_transport_equivalence.py"], "test_19_transport"))
    print("== verification passed", flush=True)


if __name__ == "__main__":
    main()
