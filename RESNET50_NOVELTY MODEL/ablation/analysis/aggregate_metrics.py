"""Collects every completed run in results/raw into tidy tables and checks they are comparable.

Writes results/processed/{combined_metrics, per_round, per_hospital, subsets}.csv and validation.json.
Only runs with DONE.json are used; incomplete or failed runs are listed in validation.json.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from ablation.common import RAW_DIR as RAW  # noqa: E402
PROCESSED = os.environ.get("FEDMEL_PROCESSED_DIR", os.path.join(ROOT, "results", "processed"))
BYTES_PER_VALUE = 4   # float32 state_dict (BN counters are int64 but negligible)


def _json(path: str) -> Dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _jsonl(path: str) -> List[Dict]:
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _auc(y, p) -> float:
    return float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan")


def model_values(run: str) -> int:
    for rec in _jsonl(os.path.join(run, "logs", "audit.jsonl")):
        if rec.get("event") == "checkpoint_saved":
            return int(rec["n_parameters"])
    return 0


def load_run(run: str) -> Dict:
    exp, seed = run.split(os.sep)[-2], int(run.split(os.sep)[-1].split("_")[1])
    info, done = _json(os.path.join(run, "run_info.json")), _json(os.path.join(run, "DONE.json"))
    rounds = pd.read_csv(os.path.join(run, "metrics.csv"))
    clients = pd.DataFrame(_jsonl(os.path.join(run, "logs", "clients.jsonl")))
    fits = clients[clients.event == "client_fit"] if len(clients) else clients
    mb = model_values(run) * BYTES_PER_VALUE
    selected = fits.groupby("round").size() if len(fits) else pd.Series(dtype=float)
    row = {"experiment": exp, "seed": seed, "run_id": info["run_id"], "config_hash": info["config_hash"],
           "split_id": info["split_id"], "code_hash": info["code_hash"], "gpu": info["gpu"],
           "torch": info["torch"], "flwr": info["flwr"], "rounds": len(rounds),
           "train_hours": rounds.round_seconds.sum() / 3600,
           "client_fit_hours": fits.wall_seconds.sum() / 3600 if len(fits) else np.nan,
           "gpu_peak_mb": fits.gpu_peak_mb.max() if len(fits) else np.nan,
           "clients_aggregated_mean": rounds.clients_aggregated.mean(),
           "updates_aggregated": rounds.clients_aggregated.sum(),
           "gb_transmitted": 2 * selected.sum() * mb / 1e9,             # broadcast + return, every selected client
           "gb_aggregated_uplink": rounds.clients_aggregated.sum() * mb / 1e9,
           "best_round": None}
    for tag in ("final_round", "final_best"):
        metrics = _json(os.path.join(run, tag, "metrics.json"))
        for key in ("roc_auc", "pr_auc", "f1", "sensitivity", "specificity", "balanced_accuracy", "accuracy", "mcc"):
            row[f"{tag}_{key}"] = metrics.get(key)
    best = [r for r in _jsonl(os.path.join(run, "logs", "audit.jsonl")) if r.get("became_best_checkpoint")]
    row["best_round"] = best[-1]["round"] if best else None
    return row


def predictions(run: str, tag: str = "final_round") -> pd.DataFrame:
    pred = pd.read_csv(os.path.join(run, tag, "test_predictions.csv"))
    rows = pd.read_csv(os.path.join(run, "test_rows.csv"))
    return pred.merge(rows.drop(columns=["hospital"]), on="isic_id", how="left", validate="one_to_one")


def main() -> None:
    os.makedirs(PROCESSED, exist_ok=True)
    runs = sorted(glob.glob(os.path.join(RAW, "*", "seed_*")))
    done = [r for r in runs if os.path.isfile(os.path.join(r, "DONE.json"))]
    not_done = {os.path.relpath(r, RAW): (_json(os.path.join(r, "status.json")) if os.path.isfile(
        os.path.join(r, "status.json")) else {"state": "UNKNOWN"}) for r in runs if r not in done}

    combined = pd.DataFrame([load_run(r) for r in done])
    per_round, per_hospital, subsets = [], [], []
    for r in done:
        exp, seed = r.split(os.sep)[-2], int(r.split(os.sep)[-1].split("_")[1])
        per_round.append(pd.read_csv(os.path.join(r, "metrics.csv")).assign(experiment=exp, seed=seed))
        pred = predictions(r)
        for hospital, g in pred.groupby("hospital"):
            per_hospital.append({"experiment": exp, "seed": seed, "hospital": hospital, "rows": len(g),
                                 "malignant": int(g.y_true.sum()), "roc_auc": _auc(g.y_true, g.y_prob)})
        for name, mask in (("all", pred.index == pred.index), ("unseen_patient", ~pred.seen_patient.astype(bool)),
                           ("unseen_lesion", ~pred.seen_lesion.astype(bool)),
                           ("histopathology", pred.diagnosis_confirm_type.eq("histopathology"))):
            g = pred[mask]
            subsets.append({"experiment": exp, "seed": seed, "subset": name, "rows": len(g),
                            "malignant": int(g.y_true.sum()), "roc_auc": _auc(g.y_true, g.y_prob),
                            "pr_auc": float(average_precision_score(g.y_true, g.y_prob)) if g.y_true.any() else np.nan})

    checks = {"completed_runs": len(done), "incomplete_runs": not_done}
    if len(combined):
        checks["one_config_hash_per_experiment"] = bool((combined.groupby("experiment").config_hash.nunique() == 1).all())
        checks["single_split_id"] = combined.split_id.nunique() == 1
        checks["single_code_hash"] = combined.code_hash.nunique() == 1
        checks["single_software"] = combined[["torch", "flwr"]].drop_duplicates().shape[0] == 1
        checks["gpus"] = sorted(combined.gpu.unique().tolist())
        with open(os.path.join(ROOT, "ablation", "experiments.yaml"), encoding="utf-8") as fh:
            registry = yaml.safe_load(fh)
        expected_rounds = int(os.environ.get("FEDMEL_ROUNDS", registry["common"]["num_server_rounds"]))
        checks["all_runs_full_rounds"] = bool((combined.rounds == expected_rounds).all())
        checks["seeds_per_experiment"] = combined.groupby("experiment").seed.apply(sorted).to_dict()
        bad = combined[(combined.final_round_roc_auc < 0.5) | combined.final_round_roc_auc.isna()]
        checks["unexpected_metric_values"] = bad[["experiment", "seed", "final_round_roc_auc"]].to_dict("records")
        checks["duplicate_runs"] = int(combined.duplicated(["experiment", "seed"]).sum())

    combined.to_csv(os.path.join(PROCESSED, "combined_metrics.csv"), index=False)
    (pd.concat(per_round) if per_round else pd.DataFrame()).to_csv(os.path.join(PROCESSED, "per_round.csv"), index=False)
    pd.DataFrame(per_hospital).to_csv(os.path.join(PROCESSED, "per_hospital.csv"), index=False)
    pd.DataFrame(subsets).to_csv(os.path.join(PROCESSED, "subsets.csv"), index=False)
    with open(os.path.join(PROCESSED, "validation.json"), "w", encoding="utf-8") as fh:
        json.dump(checks, fh, indent=2, default=str)
    print(json.dumps({k: v for k, v in checks.items() if k != "incomplete_runs"}, indent=2, default=str))
    print(f"incomplete runs: {len(not_done)}")


if __name__ == "__main__":
    main()
