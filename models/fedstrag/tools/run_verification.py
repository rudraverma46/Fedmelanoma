"""Runs the whole FedStrag verification sequence and saves the evidence to results/verification/.

  1. tests 0-16: integrity, model and data (GPU), FedStrag weights, protocol and environment (pytest)
  2. GPU smoke run `smoke-seed42`: 10% of training rows, 1 local epoch, 6 global iterations;
     the process stops after iteration 2 and a NEW process resumes it
  3. evaluation/evaluate_checkpoint.py on the final smoke checkpoint (the full pooled test set)
  4. tests 17-22 on those artefacts

Exits non-zero if any step fails. Smoke results verify behaviour only; they are not experimental results.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "verification")
RUN = "smoke-seed42"
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
SMOKE = ["--run-id", RUN, "--smoke-fraction", "0.10", "--global-iterations", "6", "--local-epochs", "1"]


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
    step("tests_00_to_16", pytest_args(["tests/test_00_reference_integrity.py", "tests/test_01_model_and_data.py",
                                        "tests/test_02_fedstrag_math.py", "tests/test_04_environment.py"], "tests_00_to_16"))
    for folder in (f"checkpoints/{RUN}", f"results/{RUN}"):
        shutil.rmtree(os.path.join(ROOT, folder), ignore_errors=True)
    for sub in ("iterations", "clients", "updates", "rounds", "op", "math", "audit"):
        path = os.path.join(ROOT, "logs", sub, f"{RUN}.jsonl")
        if os.path.exists(path):
            os.remove(path)
    if os.path.exists(os.path.join(ROOT, "logs", "resources", f"{RUN}.csv")):
        os.remove(os.path.join(ROOT, "logs", "resources", f"{RUN}.csv"))
    step("smoke_until_iteration_2", ["-m", "runtime.run_fedstrag", *SMOKE, "--stop-after-iteration", "2"])
    step("smoke_resume_to_end", ["-m", "runtime.run_fedstrag", *SMOKE, "--resume"])
    step("smoke_evaluate_final", ["evaluation/evaluate_checkpoint.py", f"checkpoints/{RUN}/final_global_model.pth", f"results/{RUN}/final"])
    step("tests_17_to_22", pytest_args(["tests/test_03_smoke_run.py"], "tests_17_to_22"))
    print("== verification passed", flush=True)


if __name__ == "__main__":
    main()
