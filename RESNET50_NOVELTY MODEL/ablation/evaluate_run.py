"""Runs evaluation/evaluate_checkpoint.py with this run's data paths instead of the base config's.

    python -m ablation.evaluate_run --exp E00_full --seed 42 <checkpoint.pth> <output_dir>
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ablation.common import build_run_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("checkpoint")
    parser.add_argument("out_dir")
    args = parser.parse_args()
    cfg = build_run_config(args.exp, args.seed)

    import matplotlib
    matplotlib.use("Agg")
    from evaluation import evaluate_checkpoint
    evaluate_checkpoint.load_config = lambda: cfg
    sys.argv = ["evaluate_checkpoint", args.checkpoint, args.out_dir]
    evaluate_checkpoint.main()


if __name__ == "__main__":
    main()
