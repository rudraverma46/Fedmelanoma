"""Body of one Kaggle background job. ablation/kaggle_launcher.py fills in PARAMS and pushes it.

The job: checks the GPUs, installs Flower 1.13.1, clones the pinned commit into /kaggle/working,
copies results / checkpoints / logs from the previous job of the same chain (kernel_sources), gets
the image cache, runs one ablation/worker.py queue per GPU with a deadline, prunes per-round
checkpoints of unfinished runs (the resume state is kept), and prints STATUS lines that
`kaggle kernels logs` shows while it runs. Everything under /kaggle/working is the job's output.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

PARAMS = {}  # filled in by kaggle_launcher.py

T0 = time.time()
DEADLINE = T0 + PARAMS["hours"] * 3600
WORK = "/kaggle/working"
REPO = os.path.join(WORK, "repo")
ROOT = os.path.join(REPO, "RESNET50_NOVELTY MODEL")
CACHE = "/tmp/fedmel_cache"
os.environ["PATH"] += ":/opt/bin"


def sh(cmd, check=True):
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=check)


def status(**fields):
    print("STATUS " + json.dumps({"t": round(time.time() - T0), "job": PARAMS["slug"], **fields}), flush=True)


def find_dir(name, top="/kaggle/input"):
    hits = [p for p in glob.glob(os.path.join(top, "**", name), recursive=True) if os.path.isdir(p)]
    return min(hits, key=len) if hits else None


def run_states():
    states = {}
    for path in glob.glob(os.path.join(ROOT, "results", "raw", "*", "seed_*", "status.json")):
        with open(path, encoding="utf-8") as fh:
            st = json.load(fh)
        key = f"{path.split(os.sep)[-3]}:{path.split(os.sep)[-2][5:]}"
        states[key] = {k: st.get(k) for k in ("state", "rounds_done", "rounds_total", "reason", "last_round_seconds")}
    return states


def main():
    import torch
    gpus = torch.cuda.device_count()
    status(stage="start", gpus=gpus, gpu_names=[torch.cuda.get_device_name(i) for i in range(gpus)])
    if gpus < len(PARAMS["queues"]):
        raise SystemExit(f"need {len(PARAMS['queues'])} GPUs, found {gpus}: refusing to run on fewer or on the CPU")

    sh("pip install -q --no-deps --target /tmp/fmdeps flwr==1.13.1 'protobuf>=4.25.2,<5' 'cryptography>=42.0.4,<43' "
       "'iterators>=0.0.2,<0.0.3' 'pathspec>=0.12.1,<0.13' 'pycryptodome>=3.18,<4' 'rich>=13.5,<14' "
       "'tomli>=2.0.1,<3' 'tomli-w>=1,<2' 'typer>=0.12.5,<0.13'")
    sh(f"git clone -q -b {PARAMS['branch']} {PARAMS['repo']} {REPO} && git -C {REPO} checkout -q {PARAMS['commit']}")

    if PARAMS.get("previous"):
        prev = find_dir(PARAMS["previous"].split("/")[-1])
        prev_root = prev and find_dir("RESNET50_NOVELTY MODEL", prev)
        if not prev_root:
            raise SystemExit(f"previous job output {PARAMS['previous']} not found under /kaggle/input")
        for sub in ("results", "checkpoints", "logs"):
            if os.path.isdir(os.path.join(prev_root, sub)):
                shutil.copytree(os.path.join(prev_root, sub), os.path.join(ROOT, sub), dirs_exist_ok=True)
        status(stage="restored", previous=PARAMS["previous"], runs=run_states())

    data = find_dir("hospital_partitions")
    cached = find_dir(PARAMS["cache_dataset"].split("/")[-1]) if PARAMS.get("cache_dataset") else None
    env = {**os.environ, "PYTHONPATH": "/tmp/fmdeps", "FEDMEL_DATA": data, "FEDMEL_CACHE": CACHE,
           "FEDMEL_EXECUTION_ID": PARAMS["slug"], "PYTHONIOENCODING": "utf-8"}
    if cached:
        src = find_dir("fedmel_cache", cached) or cached
        shutil.copytree(src, CACHE, dirs_exist_ok=True)
    subprocess.run([sys.executable, "-m", "ablation.build_cache"], cwd=ROOT, env=env, check=True)
    status(stage="cache_ready", seconds=round(time.time() - T0))

    workers = []
    for gpu, queue in enumerate(PARAMS["queues"]):
        log = open(os.path.join(WORK, f"worker_gpu{gpu}.log"), "a", encoding="utf-8")
        workers.append(subprocess.Popen([sys.executable, "-m", "ablation.worker", "--gpu", str(gpu), "--runs", *queue,
                                         "--deadline", str(DEADLINE)], cwd=ROOT, env=env, stdout=log,
                                        stderr=subprocess.STDOUT))
    while any(w.poll() is None for w in workers):
        time.sleep(300)
        status(stage="running", runs=run_states())
    codes = [w.returncode for w in workers]

    for ckpt_dir in glob.glob(os.path.join(ROOT, "checkpoints", "*")):   # unfinished runs only
        rounds = sorted(glob.glob(os.path.join(ckpt_dir, "global_round_*.pth")))
        for path in rounds[:-1]:
            os.remove(path)
    states = run_states()
    all_done = all(states.get(r, {}).get("state") == "COMPLETED" for q in PARAMS["queues"] for r in q)
    status(stage="end", worker_exit_codes=codes, all_done=all_done, runs=states)
    with open(os.path.join(WORK, "job_summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"params": PARAMS, "worker_exit_codes": codes, "all_done": all_done, "runs": states,
                   "seconds": time.time() - T0}, fh, indent=2)


if __name__ == "__main__":
    main()
