"""Runs a queue of (experiment, seed) on one GPU, resuming, stopping before a deadline, finalising.

    python -m ablation.worker --gpu 0 --runs E00_full:42 E01_random_selection:42 [--deadline EPOCH]

Per run, in results/raw/<exp>/seed_<s>/:
  status.json    QUEUED / RUNNING / COMPLETED / FAILED / INTERRUPTED / RETRY_PENDING, rounds done, reason
  training.log   console output of every attempt (appended)
  DONE.json      written last, only when the run and its evaluations are complete
A run with DONE.json is never started again. A run with a resume_state.pkl always continues from it.
Crashes are retried from the last completed round, at most MAX_STALLS times without progress.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ablation.common import atomic_write_json, build_run_config, run_dir  # noqa: E402

MAX_STALLS = 3
POLL_SECONDS = 30
DEFAULT_ROUND_SECONDS = 900      # used for the deadline margin until a round has been timed


def rounds_log(rid: str) -> List[dict]:
    path = os.path.join(ROOT, "logs", "rounds", f"{rid}.jsonl")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def set_status(out: str, **fields) -> None:
    path = os.path.join(out, "status.json")
    status = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            status = json.load(fh)
    status.update(fields, updated_at=time.time())
    atomic_write_json(path, status)


def failure_reason(log_path: str) -> str:
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        tail = fh.readlines()[-60:]
    text = "".join(tail)
    if "CUDA out of memory" in text or "OutOfMemoryError" in text:
        return "GPU out of memory"
    if "MemoryError" in text:
        return "host out of memory"
    errors = [line.strip() for line in tail if "Error" in line or "Exception" in line]
    return errors[-1][:300] if errors else "process exited without a traceback"


def write_metrics_csv(rid: str, out: str) -> int:
    rows = []
    for rec in rounds_log(rid):
        fit, ev = rec.get("fit") or {}, rec.get("eval") or {}
        rows.append({
            "round": rec["round"], "round_seconds": rec.get("seconds"),
            "train_loss": fit.get("train_loss"),
            "clients_aggregated": fit.get("fedmelanoma_clients_aggregated"),
            "new_stragglers_buffered": fit.get("fedmelanoma_new_stragglers_buffered"),
            "eval_loss_1_minus_auc": rec.get("eval_loss"), "eval_focal_loss": ev.get("eval_focal_loss"),
            **{f"eval_{k}": ev.get(k) for k in ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1",
                                                "sensitivity", "specificity", "balanced_accuracy", "mcc")},
        })
    if rows:
        tmp = os.path.join(out, "metrics.csv.tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, os.path.join(out, "metrics.csv"))
    return len(rows)


def write_test_rows(partitions_root: str, out: str) -> None:
    """Patient / lesion ids of the pooled test rows and whether either also occurs in any training split,
    so the analysis can cluster by patient and report unseen-patient scores without the dataset."""
    import pandas as pd
    cols = ["isic_id", "patient_id", "lesion_id", "diagnosis_confirm_type"]
    from hospital_registry import discover_hospitals
    hospitals = discover_hospitals(partitions_root)
    read = lambda split: pd.concat(  # noqa: E731
        [pd.read_csv(os.path.join(partitions_root, h, f"metadata_{split}.csv"), usecols=cols, low_memory=False)
         .assign(hospital=h) for h in hospitals], ignore_index=True)
    train, test = read("train"), read("test")
    test = test.assign(seen_patient=test.patient_id.isin(set(train.patient_id.dropna())),
                       seen_lesion=test.lesion_id.isin(set(train.lesion_id.dropna())))
    test.to_csv(os.path.join(out, "test_rows.csv"), index=False)


def finalise(exp: str, seed: int, rid: str, rounds: int, out: str, env: Dict[str, str]) -> None:
    """Evaluate round-N and best checkpoints on the pooled test set, copy logs, prune checkpoints."""
    ckpt_dir = os.path.join(ROOT, "checkpoints", rid)
    write_test_rows(build_run_config(exp, seed)["data"]["partitions_root"], out)
    done_rounds = write_metrics_csv(rid, out)
    if done_rounds != rounds:
        raise RuntimeError(f"incomplete run: {done_rounds}/{rounds} rounds in the round log")
    targets = {"final_round": os.path.join(ckpt_dir, f"global_round_{rounds:04d}.pth"),
               "final_best": os.path.join(ckpt_dir, "best_global_model.pth")}
    with open(os.path.join(out, "training.log"), "a", encoding="utf-8") as log:
        for name, ckpt in targets.items():
            if os.path.isfile(os.path.join(out, name, "metrics.json")):
                continue
            code = subprocess.call([sys.executable, "-W", "ignore", "-m", "ablation.evaluate_run", "--exp", exp,
                                    "--seed", str(seed), ckpt, os.path.join(out, name)],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if code != 0:
                raise RuntimeError(f"evaluation of {name} failed (exit {code})")
    for kind in ("rounds", "clients", "op", "math", "audit", "straggler", "resources"):
        for src in glob.glob(os.path.join(ROOT, "logs", kind, f"{rid}.*")):
            os.makedirs(os.path.join(out, "logs"), exist_ok=True)
            shutil.copy2(src, os.path.join(out, "logs", f"{kind}{os.path.splitext(src)[1]}"))
    keep = {os.path.basename(p) for p in targets.values()}
    os.makedirs(os.path.join(out, "checkpoint"), exist_ok=True)
    for name in keep:
        if not os.path.isfile(os.path.join(out, "checkpoint", name)):
            shutil.copy2(os.path.join(ckpt_dir, name), os.path.join(out, "checkpoint", name))
    if os.path.isdir(ckpt_dir):
        shutil.rmtree(ckpt_dir)   # per-round checkpoints and resume state; final and best are kept above
    with open(os.path.join(out, "final_round", "metrics.json"), encoding="utf-8") as fh:
        final = json.load(fh)
    atomic_write_json(os.path.join(out, "DONE.json"), {"run_id": rid, "rounds": rounds, "finished_at": time.time(),
                                                       "final_round_roc_auc": final.get("roc_auc")})


def run(exp: str, seed: int, gpu: str, deadline: Optional[float], env_base: Dict[str, str]) -> str:
    cfg = build_run_config(exp, seed)
    rid, rounds = cfg["experiment"]["run_id"], cfg["federation"]["num_server_rounds"]
    out = run_dir(exp, seed)
    os.makedirs(out, exist_ok=True)
    if os.path.isfile(os.path.join(out, "DONE.json")):
        return "COMPLETED"
    env = {**env_base, "CUDA_VISIBLE_DEVICES": gpu}
    stop_file = os.path.join(ROOT, "logs", f"STOP_{rid}")
    if os.path.exists(stop_file):
        os.remove(stop_file)

    stalls, attempt = 0, 0
    while True:
        before = len(rounds_log(rid))
        if before < rounds:
            if deadline and time.time() + 2 * DEFAULT_ROUND_SECONDS > deadline:
                set_status(out, state="INTERRUPTED", run_id=rid, rounds_done=before, reason="deadline before start")
                return "INTERRUPTED"
            attempt += 1
            set_status(out, state="RUNNING", run_id=rid, rounds_done=before, rounds_total=rounds, gpu=gpu,
                       attempt=attempt, started_at=time.time())
            log_path = os.path.join(out, "training.log")
            with open(log_path, "a", encoding="utf-8") as log:
                log.write(f"\n===== attempt {attempt} at {time.strftime('%Y-%m-%d %H:%M:%S')} gpu {gpu} =====\n")
                log.flush()
                proc = subprocess.Popen([sys.executable, "-W", "ignore", "-m", "ablation.run_one", "--exp", exp,
                                         "--seed", str(seed), "--resume"],
                                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                while proc.poll() is None:
                    time.sleep(POLL_SECONDS)
                    recs = rounds_log(rid)
                    longest = max((r.get("seconds") or 0 for r in recs), default=DEFAULT_ROUND_SECONDS)
                    set_status(out, rounds_done=len(recs), last_round_seconds=recs[-1].get("seconds") if recs else None)
                    if deadline and not os.path.exists(stop_file) and time.time() + 1.5 * longest > deadline:
                        open(stop_file, "w").close()   # run_experiment stops cleanly after this round
                code = proc.returncode
            after = len(rounds_log(rid))
            if code == 10:
                set_status(out, state="INTERRUPTED", rounds_done=after, reason="deadline (safe stop after round)")
                return "INTERRUPTED"
            if code != 0:
                stalls = 0 if after > before else stalls + 1
                reason = failure_reason(log_path)
                if stalls >= MAX_STALLS:
                    set_status(out, state="FAILED", rounds_done=after, reason=reason, exit_code=code)
                    return "FAILED"
                set_status(out, state="RETRY_PENDING", rounds_done=after, reason=reason, exit_code=code)
                time.sleep(POLL_SECONDS)
                continue
        try:
            finalise(exp, seed, rid, rounds, out, env)
        except Exception as exc:  # keep the run's files; report instead of hiding the failure
            set_status(out, state="FAILED", reason=f"finalise: {exc}")
            return "FAILED"
        set_status(out, state="COMPLETED", rounds_done=rounds, finished_at=time.time())
        return "COMPLETED"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--runs", nargs="+", required=True, help="EXP:SEED ...")
    parser.add_argument("--deadline", type=float, help="unix time by which the worker must have stopped")
    args = parser.parse_args()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
    queue = [(exp, int(seed)) for exp, seed in (item.rsplit(":", 1) for item in args.runs)]
    for exp, seed in queue:   # mark the queue so the monitor can see what is waiting
        if not os.path.isfile(os.path.join(run_dir(exp, seed), "status.json")):
            set_status(run_dir(exp, seed), state="QUEUED", gpu=args.gpu)
    for exp, seed in queue:
        state = run(exp, seed, args.gpu, args.deadline, env)
        print(f"[worker gpu{args.gpu}] {exp} seed {seed}: {state}", flush=True)
        if state == "INTERRUPTED":
            return 10
    return 0


if __name__ == "__main__":
    sys.exit(main())
