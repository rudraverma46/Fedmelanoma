"""Laptop-side controller for Kaggle background jobs. State lives in manifests/, so any session can
pick up where the last one stopped.

    python -m ablation.kaggle_launcher plan    --batch batch1 [--chains 1]
    python -m ablation.kaggle_launcher submit  --batch batch1 [--chains 1]   first job of each chain
    python -m ablation.kaggle_launcher submit  --batch blueprint --seeds 44 --chain b    a teammate's share
    python -m ablation.kaggle_launcher submit  --chain d --runs A2_no_attenuation:43 ...   an explicit run list
    python -m ablation.kaggle_launcher status                               job state + per-run progress
    python -m ablation.kaggle_launcher advance                              continue chains whose job ended
    python -m ablation.kaggle_launcher collect                              download results (no checkpoints)
    python -m ablation.kaggle_launcher requeue --chain batch1-a --batch blueprint
                                            after a job ended: the chain's next jobs run this batch instead

A chain is a sequence of jobs over the same run queues; job k+1 reads job k's output, so runs
resume from their last completed round. Completed runs (DONE.json) are skipped by the worker, so a
continuation never repeats work. Each job pins the commit it clones; uncommitted changes are refused.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ablation.common import atomic_write_json, batch_runs, build_run_config, config_hash, load_registry  # noqa: E402

REPO_URL = "https://github.com/rudraverma46/Fedmelanoma"
BRANCH = "resnet50-novelty-ablation"
DATASET = "visheshkumarbansal/fedmel01"
CACHE_DATASET = os.environ.get("FEDMEL_CACHE_DATASET", "")   # e.g. visheshkumarbansal/fedmel01-cache224
JOB_HOURS = 11.0          # Kaggle stops sessions at 12 h; the worker stops after the round that would cross this
SMOKE_AFTER_REQUEUE = ["A6_tv_diversity:42", "A8_synchronous:42", "A3_greedy_selection:42", "A4_binary_reliability:42"]
MAX_FAILED_JOBS = 2       # consecutive jobs of a chain that end without a STATUS "end" line
MANIFESTS = os.path.join(ROOT, "manifests")
JOBS_DIR = os.path.join(ROOT, "ablation", "kaggle_jobs")
JOBS_PATH = os.path.join(MANIFESTS, "jobs.json")
STATUS_PATH = os.path.join(MANIFESTS, "experiment_status.json")
EXPERIMENTS_PATH = os.path.join(MANIFESTS, "experiments.json")


def kaggle(*args: str, check: bool = True) -> str:
    proc = subprocess.run([sys.executable, "-c", "from kaggle.cli import main; main()", *args],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (proc.stdout + proc.stderr).strip()
    if check and proc.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)} failed: {out[-500:]}")
    return out


def username() -> str:
    """Owner of the jobs. The access_token file (API tokens) carries no username, so it is the dataset owner."""
    home = os.path.join(os.path.expanduser("~"), ".kaggle")
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    if os.path.isfile(os.path.join(home, "kaggle.json")):
        with open(os.path.join(home, "kaggle.json"), encoding="utf-8") as fh:
            return json.load(fh)["username"]
    if os.path.isfile(os.path.join(home, "access_token")) or os.environ.get("KAGGLE_API_TOKEN"):
        return DATASET.split("/")[0]
    raise SystemExit(f"No Kaggle API token: create one at kaggle.com -> Settings -> API Tokens, save to {home}")


def pinned_commit() -> str:
    git = ["git", "-C", ROOT]
    dirty = subprocess.run(git + ["status", "--porcelain", "--", "ablation", "configs", "runtime", "server", "client",
                                  "models", "monitoring", "evaluation", "*.py"], capture_output=True, text=True).stdout
    dirty = "\n".join(l for l in dirty.splitlines() if "kaggle_jobs" not in l)
    if dirty.strip():
        raise SystemExit(f"Commit and push first; uncommitted changes:\n{dirty}")
    head = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    remote = subprocess.run(git + ["ls-remote", REPO_URL, f"refs/heads/{BRANCH}"], capture_output=True, text=True,
                            check=True).stdout.split()
    if not remote or remote[0] != head:
        raise SystemExit(f"Local HEAD {head[:8]} is not the pushed head of {BRANCH}; push first.")
    return head


def load_jobs() -> Dict:
    if os.path.isfile(JOBS_PATH):
        with open(JOBS_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    return {"chains": {}}


def assignment(batch: str, chains: int, seeds: Optional[List[int]] = None,
               first: str = "a", runs: Optional[List[str]] = None) -> Dict[str, List[List[str]]]:
    """runs, if given, is an explicit exp:seed list (e.g. a chunk handed to a collaborator's own account)
    and is used as-is instead of being computed from batch + seeds."""
    all_runs = runs if runs else [f"{exp}:{seed}" for exp, seed in batch_runs(batch) if not seeds or seed in seeds]
    plan = {}
    for c in range(chains):
        mine = all_runs[c::chains]
        plan[chr(ord(first) + c)] = [mine[0::2], mine[1::2]]   # one queue per T4
    return plan


def write_experiments_manifest(batch: str) -> None:
    rows = []
    for exp, seed in batch_runs(batch):
        cfg = build_run_config(exp, seed)
        rows.append({"experiment": exp, "seed": seed, "run_id": cfg["experiment"]["run_id"],
                     "config_hash": config_hash(cfg), "variant": cfg["ablation"]["variant"],
                     "rounds": cfg["federation"]["num_server_rounds"],
                     "result_location": f"results/raw/{exp}/seed_{seed}"})
    atomic_write_json(EXPERIMENTS_PATH, {"batch": batch, "runs": rows})


def push_job(user: str, batch: str, chain: str, part: int, queues: List[List[str]], commit: str,
             previous: Optional[str], smoke: Optional[List[str]] = None) -> Dict:
    slug = f"fedmel-{batch}-{chain}{part}"
    folder = os.path.join(JOBS_DIR, slug)
    os.makedirs(folder, exist_ok=True)
    params = {"slug": slug, "batch": batch, "chain": chain, "part": part, "queues": queues, "repo": REPO_URL,
              "branch": BRANCH, "commit": commit, "previous": previous, "hours": JOB_HOURS,
              "cache_dataset": CACHE_DATASET, "smoke": smoke or part == 1}
    with open(os.path.join(ROOT, "ablation", "kaggle_job_template.py"), encoding="utf-8") as fh:
        body = fh.read().replace("PARAMS = {}  # filled in by kaggle_launcher.py", f"PARAMS = {params!r}")
    with open(os.path.join(folder, "job.py"), "w", encoding="utf-8") as fh:
        fh.write(body)
    meta = {"id": f"{user}/{slug}", "title": slug, "code_file": "job.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [DATASET] + ([CACHE_DATASET] if CACHE_DATASET else []),
            "kernel_sources": [previous] if previous else [], "competition_sources": [], "model_sources": []}
    with open(os.path.join(folder, "kernel-metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    out = kaggle("kernels", "push", "-p", folder, "--accelerator", "NvidiaTeslaT4")
    print(out)
    return {"slug": f"{user}/{slug}", "part": part, "previous": previous, "commit": commit,
            "submitted_at": time.time(), "push_output": out[-300:]}


def last_status(slug: str) -> Optional[Dict]:
    """Last STATUS line of a job. Kaggle returns the log of an ended job as a JSON list of stream records."""
    logs = kaggle("kernels", "logs", slug, check=False)
    try:
        lines = [rec.get("data", "") for rec in json.loads(logs)]
    except ValueError:
        lines = logs.splitlines()
    found = None
    for line in "".join(lines).splitlines():
        m = re.search(r"STATUS (\{.*\})", line)
        if m:
            try:
                found = json.loads(m.group(1))
            except ValueError:
                pass
    return found


def job_state(slug: str) -> str:
    out = kaggle("kernels", "status", slug, check=False)
    m = re.search(r'status "?([A-Za-z_.]+)"?', out)
    return (m.group(1) if m else out[-80:]).lower().replace("kernelworkerstatus.", "")


def cmd_plan(args) -> None:
    for chain, queues in assignment(args.batch, args.chains, args.seeds, args.chain, args.runs).items():
        for gpu, queue in enumerate(queues):
            print(f"chain {chain} gpu{gpu}: {len(queue)} runs: {' '.join(queue)}")


def cmd_submit(args) -> None:
    user, commit = username(), pinned_commit()
    jobs = load_jobs()
    write_experiments_manifest(args.batch)
    for chain, queues in assignment(args.batch, args.chains, args.seeds, args.chain, args.runs).items():
        key = f"{args.batch}-{chain}"
        if key in jobs["chains"]:
            print(f"chain {key} already submitted; use advance")
            continue
        try:
            job = push_job(user, args.batch, chain, 1, queues, commit, None,
                           None if args.batch == "batch1" else SMOKE_AFTER_REQUEUE)
        except RuntimeError as exc:
            print(f"chain {key}: push refused, will retry on advance: {exc}")
            job = None
        jobs["chains"][key] = {"batch": args.batch, "chain": chain, "queues": queues,
                               "jobs": [job] if job else [], "failed_jobs": 0, "finished": False}
        atomic_write_json(JOBS_PATH, jobs)


def cmd_status(args) -> Dict:
    jobs = load_jobs()
    runs: Dict[str, Dict] = {}
    for key, chain in jobs["chains"].items():
        if not chain["jobs"]:
            print(f"{key}: not started (waiting for a free GPU session)")
            continue
        slug = chain["jobs"][-1]["slug"]
        state, st = job_state(slug), last_status(slug)
        chain["jobs"][-1]["state"] = state
        print(f"{key}: job {slug} is {state}; last STATUS stage: {st and st.get('stage')}"
              f"{' (chain finished)' if chain['finished'] else ''}")
        for run, info in ((st or {}).get("runs") or {}).items():
            runs[run] = {**info, "job": slug}
            print(f"   {run:32s} {info.get('state')!s:13s} {info.get('rounds_done')}/{info.get('rounds_total')}"
                  f"  {('last round %.0f s' % info['last_round_seconds']) if info.get('last_round_seconds') else ''}"
                  f"  {info.get('reason') or ''}")
    atomic_write_json(JOBS_PATH, jobs)
    atomic_write_json(STATUS_PATH, {"updated_at": time.time(), "runs": runs})
    return jobs


def cmd_advance(args) -> None:
    user = username()
    jobs = cmd_status(args)
    for key, chain in jobs["chains"].items():
        if chain["finished"]:
            continue
        previous = chain["jobs"][-1]["slug"] if chain["jobs"] else None
        if previous:
            state = chain["jobs"][-1].get("state", "")
            if state in ("running", "queued", "new", "pending"):
                continue
            st = last_status(previous)
            if st and st.get("stage") == "end" and st.get("all_done"):
                chain["finished"] = True
                print(f"{key}: all runs completed")
                continue
            if not (st and st.get("stage") == "end"):
                chain["failed_jobs"] += 1
                print(f"{key}: job {previous} ended as {state} without finishing cleanly "
                      f"({chain['failed_jobs']}/{MAX_FAILED_JOBS})")
                if chain["failed_jobs"] >= MAX_FAILED_JOBS:
                    print(f"{key}: stopped; inspect `kaggle kernels logs {previous}` before continuing")
                    continue
            else:
                chain["failed_jobs"] = 0
        try:
            part = len(chain["jobs"]) + 1
            chain["jobs"].append(push_job(user, chain["batch"], chain["chain"], part, chain["queues"],
                                          pinned_commit(), previous, chain.get("smoke_next")))
            chain.pop("smoke_next", None)
            print(f"{key}: submitted part {part}")
        except RuntimeError as exc:
            print(f"{key}: push refused, will retry: {exc}")
        atomic_write_json(JOBS_PATH, jobs)


def cmd_requeue(args) -> None:
    """Point a chain at another batch from its next job on. Runs the ended job finished are dropped, runs it left
    part-way go first (they resume from their checkpoint), and the rest are split over the two GPUs by rounds left."""
    jobs = cmd_status(args)
    chain = jobs["chains"].get(args.chain)
    if not chain or not chain["jobs"]:
        raise SystemExit(f"no started chain {args.chain!r}")
    last = chain["jobs"][-1]
    st = last_status(last["slug"])
    if last.get("state") in ("running", "queued", "new", "pending") or not (st and st.get("stage") == "end"):
        raise SystemExit(f"{last['slug']} has not ended cleanly yet ({last.get('state')}); requeue after it ends")
    states = st.get("runs") or {}
    rounds = int(load_registry()["common"]["num_server_rounds"])
    todo = [f"{exp}:{seed}" for exp, seed in batch_runs(args.batch)
            if (states.get(f"{exp}:{seed}") or {}).get("state") != "COMPLETED"]
    done_rounds = {r: (states.get(r) or {}).get("rounds_done") or 0 for r in todo}
    todo.sort(key=lambda r: done_rounds[r] == 0)   # stable: started runs first, then batch order
    queues, load = [[], []], [0, 0]
    for run in todo:
        gpu = load.index(min(load))
        queues[gpu].append(run)
        load[gpu] += rounds - done_rounds[run]
    dropped = sorted(r for q in chain["queues"] for r in q
                     if r not in todo and (states.get(r) or {}).get("state") != "COMPLETED")
    chain.setdefault("history", []).append({"at": time.time(), "after_job": last["slug"], "from_batch": chain["batch"],
                                            "to_batch": args.batch, "old_queues": chain["queues"],
                                            "dropped_unfinished": dropped})
    chain.update(batch=args.batch, queues=queues, finished=False, failed_jobs=0)
    chain["smoke_next"] = SMOKE_AFTER_REQUEUE   # the next job first runs these for 2 rounds (new code paths on GPU)
    write_experiments_manifest(args.batch)
    atomic_write_json(JOBS_PATH, jobs)
    for gpu, queue in enumerate(queues):
        print(f"{args.chain} gpu{gpu}: {len(queue)} runs, {load[gpu]} rounds: {' '.join(queue)}")
    print(f"unfinished runs left behind: {', '.join(dropped) or 'none'}")


def cmd_collect(args) -> None:
    jobs = load_jobs()
    pattern = r".*(results/raw/.*\.(json|csv|yaml|log|txt|png|jsonl)|job_summary\.json|worker_gpu\d\.log)$"
    for key, chain in jobs["chains"].items():
        if not chain["jobs"]:
            continue
        slug = chain["jobs"][-1]["slug"]
        dest = os.path.join(ROOT, "results", "kaggle", slug.split("/")[-1])
        kaggle("kernels", "output", slug, "-p", dest, "--file-pattern", pattern, "-o", check=False)
        copied = 0
        for folder, _, files in os.walk(dest):
            if "DONE.json" not in files or f"{os.sep}results{os.sep}raw{os.sep}" not in folder + os.sep:
                continue
            rel = folder.split(f"{os.sep}results{os.sep}raw{os.sep}", 1)[1]
            target = os.path.join(ROOT, "results", "raw", rel)
            if os.path.isfile(os.path.join(target, "DONE.json")):
                continue   # never overwrite a collected run
            shutil.copytree(folder, target, dirs_exist_ok=True)
            copied += 1
        print(f"{key}: downloaded {slug} -> {dest}; {copied} newly completed runs copied to results/raw")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "submit"):
        p = sub.add_parser(name)
        p.add_argument("--batch", default="batch1")
        p.add_argument("--chains", type=int, default=1)
        p.add_argument("--seeds", type=int, nargs="+", help="only these seeds (e.g. 44 for a second account's share)")
        p.add_argument("--chain", default="a", help="letter of the first chain (b for a second account)")
        p.add_argument("--runs", nargs="+", help="explicit exp:seed list handed to this account; "
                       "overrides --batch/--seeds selection")
    for name in ("status", "advance", "collect"):
        sub.add_parser(name)
    p = sub.add_parser("requeue")
    p.add_argument("--chain", required=True)
    p.add_argument("--batch", required=True)
    args = parser.parse_args()
    {"plan": cmd_plan, "submit": cmd_submit, "status": cmd_status, "advance": cmd_advance,
     "collect": cmd_collect, "requeue": cmd_requeue}[args.cmd](args)


if __name__ == "__main__":
    main()
