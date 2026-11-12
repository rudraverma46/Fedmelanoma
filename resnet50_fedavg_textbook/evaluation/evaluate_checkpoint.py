"""Runs the reference's evaluate.py main() on one checkpoint, then adds the requested extras.

evaluate.py is unmodified. It hard-codes a checkpoint path, a loader with 4 workers and output
paths relative to the working directory, so this wrapper sets the first two and runs from the
output folder (difference D11). Its loader is replaced by the same pooled, unshuffled test set
read from the pixel-identical cache.

Added outputs: thresholds 0.1/0.3/0.5/0.7/0.9 with confusion counts, per-image predictions,
and metrics.json.

Usage: python evaluation/evaluate_checkpoint.py <checkpoint.pth> <output_dir>
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding="utf-8", line_buffering=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402
from torch.utils.data import ConcatDataset  # noqa: E402

import evaluate  # noqa: E402
from runtime.config import load_config  # noqa: E402
from runtime.image_cache import build_cached_dataset, build_loader  # noqa: E402

REQUESTED_THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]


def main() -> None:
    checkpoint, out_dir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    cfg = load_config()
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(out_dir)
    captured = {}

    def build_test_loader(hospital_names, batch_size):
        evaluate.configure_multiprocessing_env()
        datasets = [build_cached_dataset(cfg["data"]["partitions_root"], cfg["data"]["cache_dir"], h, "test")
                    for h in hospital_names]
        loader = build_loader(ConcatDataset(datasets), batch_size, False, cfg["runtime"]["dataloader_workers"],
                              cfg["runtime"]["pin_memory"])
        columns = ["fitzpatrick_skin_type", "image_type", "diagnosis_confirm_type"]
        metadata_df = pd.concat([ds.df[columns] for ds in datasets], ignore_index=True)
        labels = np.concatenate([np.full(len(ds), h, dtype=object) for h, ds in zip(hospital_names, datasets)])
        captured["ids"] = [row["isic_id"] for ds in datasets for row in ds.rows]
        captured["hospital"] = labels
        return loader, list(hospital_names), metadata_df, labels

    reference_inference = evaluate.run_inference

    def run_inference(*args, **kwargs):
        captured["out"] = reference_inference(*args, **kwargs)
        return captured["out"]

    def compute_metrics(y_true, y_pred, y_prob):
        metrics = reference_compute(y_true, y_pred, y_prob)
        captured.setdefault("metrics", metrics)  # first call is the overall one
        return metrics

    reference_compute = evaluate.compute_metrics
    evaluate.CHECKPOINT_PATH = checkpoint
    evaluate.build_test_loader = build_test_loader
    evaluate.run_inference = run_inference
    evaluate.compute_metrics = compute_metrics
    evaluate.main()

    y_true, y_pred, y_prob = captured["out"]
    evaluate.run_threshold_analysis(y_true, y_prob, REQUESTED_THRESHOLDS, "threshold_analysis_requested.csv")
    rows = []
    for threshold in REQUESTED_THRESHOLDS:
        tn, fp, fn, tp = confusion_matrix(y_true, (y_prob >= threshold).astype(int), labels=[0, 1]).ravel()
        accuracy = (tp + tn) / len(y_true)
        rows.append({"threshold": threshold, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "accuracy": accuracy})
    pd.DataFrame(rows).to_csv("threshold_confusion_requested.csv", index=False)
    pd.DataFrame({"isic_id": captured["ids"], "hospital": captured["hospital"], "y_true": y_true,
                  "y_pred": y_pred, "y_prob": y_prob}).to_csv("test_predictions.csv", index=False)
    with open("metrics.json", "w", encoding="utf-8") as fh:
        json.dump({"checkpoint": checkpoint, **{k: float(v) for k, v in captured["metrics"].items()}}, fh, indent=2)


if __name__ == "__main__":
    main()
