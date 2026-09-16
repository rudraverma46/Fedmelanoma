"""
monitoring/experiment_logging.py
==================================
Structured, machine-readable experiment logging, separated by purpose per
the blueprint (Section 4, Phase 8 "Three-tier logging"):

    logs/op/<run_id>.jsonl     operational events (round lifecycle, client
                                participation, checkpointing, broadcast
                                verification)
    logs/math/<run_id>.jsonl   research-algorithm quantities (R_i, N_eff,
                                D_i, c_i, W_t, staleness, ...) -- values
                                only, never formulas; the formulas
                                themselves stay in selector.py/
                                aggregation.py/straggler.py, untouched.
    logs/audit/<run_id>.jsonl  reproducibility record: config, seeds,
                                parameter hashes, checkpoint events,
                                run start/end, exceptions.

One JSON object per line (JSONL) so logs are trivially appendable,
streamable, and analyzable with pandas.read_json(lines=True) without
holding the whole run in memory.

No raw images, raw patient metadata, or raw per-sample data are ever
written here -- every record is either a scalar/aggregate quantity or a
small id/label already used elsewhere in this codebase (client_id,
hospital name, round number).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, Optional


def new_run_id(prefix: str = "run") -> str:
    """
    Deterministic-looking, collision-resistant run identifier:
    <prefix>-<UTC timestamp>-<8 hex chars>. Used to namespace
    checkpoints/, logs/, and results/plots/ so re-running an experiment
    never silently clobbers a previous run's outputs.
    """
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"


class ExperimentLogger:
    """
    Thin, dependency-free JSONL writer for the three log tiers plus the
    dedicated straggler log (monitoring/straggler_logging.py writes
    through this same class so all logs for one run share one run_id and
    one root directory).

    Each `log_*` call appends one line immediately (no buffering), so
    logs remain complete even if the process is killed mid-run.
    """

    def __init__(self, run_id: str, logs_root: str = "logs") -> None:
        self.run_id = run_id
        self.logs_root = logs_root
        self.op_path = os.path.join(logs_root, "op", f"{run_id}.jsonl")
        self.math_path = os.path.join(logs_root, "math", f"{run_id}.jsonl")
        self.audit_path = os.path.join(logs_root, "audit", f"{run_id}.jsonl")
        self.straggler_path = os.path.join(logs_root, "op", "stragglers", f"{run_id}.jsonl")
        for path in (self.op_path, self.math_path, self.audit_path, self.straggler_path):
            os.makedirs(os.path.dirname(path), exist_ok=True)

    def _append(self, path: str, record: Dict[str, Any]) -> None:
        record = dict(record)
        record.setdefault("run_id", self.run_id)
        record.setdefault("logged_at", time.time())
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def log_op(self, record: Dict[str, Any]) -> None:
        self._append(self.op_path, record)

    def log_math(self, record: Dict[str, Any]) -> None:
        self._append(self.math_path, record)

    def log_audit(self, record: Dict[str, Any]) -> None:
        self._append(self.audit_path, record)

    def log_straggler(self, record: Dict[str, Any]) -> None:
        self._append(self.straggler_path, record)

    # ------------------------------------------------------------------
    # Convenience: run-start reproducibility snapshot (Requirement 11)
    # ------------------------------------------------------------------
    def log_run_start(self, config: Dict[str, Any]) -> None:
        """
        Record everything needed to reproduce this run: seed, round/client
        counts, local epochs, batch size, learning rate, strategy
        hyperparameters (read, not altered), timestamp, initial model
        hash. Call once, right after FedMelanomaStrategy is constructed.
        """
        self.log_audit({"event": "run_start", **config})

    def log_run_end(self, summary: Dict[str, Any]) -> None:
        self.log_audit({"event": "run_end", **summary})

    def log_exception(self, where: str, exc: BaseException) -> None:
        self.log_audit(
            {
                "event": "exception",
                "where": where,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            }
        )


def read_jsonl(path: str):
    """Yield dict records from a JSONL file; returns nothing if the file doesn't exist yet."""
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
