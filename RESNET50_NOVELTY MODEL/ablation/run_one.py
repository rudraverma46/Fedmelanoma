"""Runs one (experiment, seed) through the unmodified runtime/run_experiment.py round loop.

    python -m ablation.run_one --exp E00_full --seed 42 [--resume] [--stop-after N]

What is replaced, all from outside the reference files:
  - load_config     -> this run's config (ablation/common.py)
  - server_fn       -> the reference server_fn, then ablation/variants.apply_variant
  - STOP_FILE       -> logs/STOP_<run_id>, so stopping one run never stops another
  - Windows-only resource probes -> Linux equivalents (psutil, nvidia-smi for this run's GPU only)
Exit codes are run_experiment's: 0 finished, 10 safe stop, anything else a crash.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import psutil  # noqa: E402

import runtime.resources as resources  # noqa: E402
import runtime.run_experiment as rx  # noqa: E402  (chdir to ROOT, utf-8 streams)
from ablation.common import atomic_write_json, build_run_config, code_hash, config_hash, run_dir, split_id  # noqa: E402
from ablation.variants import apply_variant  # noqa: E402


def linux_memory_snapshot():
    vm, gb = psutil.virtual_memory(), 2**30
    return {"ram_free_gb": vm.available / gb, "commit_limit_gb": vm.total / gb, "commit_free_gb": vm.available / gb}


def visible_gpu_snapshot():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    out = subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip()
    util, mem, temp, power = [x.strip() for x in out.splitlines()[0].split(",")]
    return {"gpu_util_pct": float(util), "gpu_mem_mb": float(mem), "gpu_temp_c": float(temp), "gpu_power_w": float(power)}


def write_run_info(cfg, exp_id: str, seed: int) -> None:
    import flwr
    import torch
    info = {
        "experiment": exp_id, "seed": seed, "run_id": cfg["experiment"]["run_id"],
        "config_hash": config_hash(cfg), "split_id": split_id(cfg["data"]["partitions_root"]),
        "code_hash": code_hash(), "execution_id": os.environ.get("FEDMEL_EXECUTION_ID", "manual"),
        "python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
        "flwr": flwr.__version__, "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    out = run_dir(exp_id, seed)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.yaml"), "w", encoding="utf-8") as fh:
        import yaml
        yaml.safe_dump(cfg, fh, sort_keys=False)
    attempts_path = os.path.join(out, "run_info.json")
    atomic_write_json(attempts_path, info)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()

    cfg = build_run_config(args.exp, args.seed)
    rid = cfg["experiment"]["run_id"]
    ablation = cfg["ablation"]

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; refusing to train on the CPU.")
    write_run_info(cfg, args.exp, args.seed)

    reference_server_fn = rx.server_fn

    def server_fn(context):
        components = reference_server_fn(context)
        apply_variant(components.strategy, ablation["variant"], ablation["harness"], args.seed)
        return components

    rx.load_config = lambda: cfg
    rx.server_fn = server_fn
    rx.STOP_FILE = os.path.join("logs", f"STOP_{rid}")
    rx.keep_awake = lambda enable: None
    resources.memory_snapshot = linux_memory_snapshot
    resources.gpu_snapshot = visible_gpu_snapshot

    sys.argv = ["run_experiment", "--run-id", rid]
    if args.resume:
        sys.argv.append("--resume")
    if args.stop_after:
        sys.argv += ["--stop-after", str(args.stop_after)]
    return rx.main()


if __name__ == "__main__":
    sys.exit(main())
