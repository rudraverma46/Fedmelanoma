"""Runs the experiment and resumes it after a crash, as long as crashes are not repeating.

Stops on success, on a safe stop (exit 10: memory limit or logs/STOP, reported rather than
retried), or after MAX_STALLS consecutive crashes without a new completed round. It never
changes any setting between attempts.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_STALLS = 3


def completed_rounds(run_id: str) -> int:
    path = os.path.join(ROOT, "logs", "rounds", f"{run_id}.jsonl")
    if not os.path.isfile(path):
        return 0
    with open(path, encoding="utf-8") as fh:
        return max((json.loads(line)["round"] for line in fh if line.strip()), default=0)


def main() -> int:
    run_id = sys.argv[1] if len(sys.argv) > 1 else "resnet50-seed42"
    log_dir = os.path.join(ROOT, "logs", "console")
    os.makedirs(log_dir, exist_ok=True)
    stalls, attempt = 0, 0
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
    while True:
        attempt += 1
        before = completed_rounds(run_id)
        with open(os.path.join(log_dir, f"{run_id}_attempt{attempt:02d}.log"), "w", encoding="utf-8") as log:
            code = subprocess.call([sys.executable, "-W", "ignore", "-m", "runtime.run_experiment", "--run-id", run_id, "--resume"],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        after = completed_rounds(run_id)
        print(f"attempt {attempt}: exit {code}, rounds {before} -> {after}", flush=True)
        if code in (0, 10):
            return code
        stalls = 0 if after > before else stalls + 1
        if stalls >= MAX_STALLS:
            print("stopping: crashes without progress", flush=True)
            return code
        time.sleep(30)


if __name__ == "__main__":
    sys.exit(main())
