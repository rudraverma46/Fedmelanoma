"""Copy reference files byte-for-byte from the EfficientNet-B3 commit and record their hashes.

Reads with `git show <commit>:<path>` from the reference clone (read-only: nothing in the
reference working tree is touched) and writes REFERENCE_FILES.json, which
tests/test_00_reference_integrity.py checks on every test run.

Usage: python tools/vendor_reference_files.py [path-to-reference-clone]
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CLONE = os.path.abspath(os.path.join(ROOT, "..", "..", "..", "FedMelanoma_repo"))
COMMIT = "1351f63b37c022a252c18960098eb388bb2f723a"
BRANCH = "federated_efficientnetb3"

VERBATIM = [
    "client/__init__.py", "client/client.py", "client/dataset.py", "client/focal_loss.py",
    "client/local_trainer.py",
    "server/__init__.py", "server/aggregation.py", "server/selector.py", "server/server.py",
    "server/straggler.py", "server/strategy.py",
    "monitoring/__init__.py", "monitoring/checkpointing.py", "monitoring/experiment_logging.py",
    "monitoring/param_hash.py", "monitoring/plotting.py", "monitoring/straggler_logging.py",
    "monitoring/subgroup_metrics.py",
    "models/__init__.py", "emd.py", "hospital_registry.py", "evaluate.py",
]


def main() -> None:
    clone = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CLONE
    record = {"repository": "https://github.com/rudraverma46/Fedmelanoma", "branch": BRANCH,
              "commit": COMMIT, "files": {}}
    for rel in VERBATIM:
        blob = subprocess.run(["git", "-C", clone, "show", f"{COMMIT}:{rel}"],
                              check=True, capture_output=True).stdout
        dest = os.path.join(ROOT, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as fh:
            fh.write(blob)
        record["files"][rel] = hashlib.sha256(blob).hexdigest()
    with open(os.path.join(ROOT, "REFERENCE_FILES.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    print(f"vendored {len(VERBATIM)} files from {COMMIT[:7]}")


if __name__ == "__main__":
    main()
