"""Per-run quantities that blueprint v4 Section 7 makes predictions about, plus A7 (calibration).

    python -m ablation.analysis.blueprint_metrics

Writes results/processed/blueprint_metrics.csv (one row per completed run):
  fairness_gap          Fitzpatrick sensitivity max - min over skin types with malignant cases (A1)
  late_round_gain       mean per-round eval ROC-AUC, rounds 21-30 minus rounds 11-20 (A2 plateau)
  min_selections, never_selected, selection_gini
                        how often each of the 10 clients was selected over the run (A3 / A6 starvation)
  sim_hours_*           simulated wall time of the run from the straggler log (A8); "window" = server waits
                        for the adaptive window W_t when anyone is late or failed, otherwise for the slowest
                        on-time client; "sync" = waits for every client that completes, and for
                        SYNC_TIMEOUT_S when one fails (blueprint W_max)
  ece_raw, ece_ts, temperature
                        A7: expected calibration error of the final-round model without (T = 1) and with
                        post-hoc temperature scaling. No validation split exists, so T is fitted by 2-fold
                        cross-fitting on the pooled test rows, folds split by patient.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from ablation.analysis.aggregate_metrics import PROCESSED, RAW, predictions  # noqa: E402

N_CLIENTS = 10
SYNC_TIMEOUT_S = 600.0
ECE_BINS = 15
EPS = 1e-6


def ece(y: np.ndarray, p: np.ndarray, bins: int = ECE_BINS) -> float:
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def scale(p: np.ndarray, t: float) -> np.ndarray:
    z = np.log(np.clip(p, EPS, 1 - EPS) / np.clip(1 - p, EPS, 1 - EPS))
    return 1.0 / (1.0 + np.exp(-z / t))


def fit_temperature(y: np.ndarray, p: np.ndarray) -> float:
    def nll(t):
        q = np.clip(scale(p, t), EPS, 1 - EPS)
        return -float(np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
    return float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)


def calibration(pred: pd.DataFrame, seed: int = 0) -> dict:
    y, p = pred.y_true.to_numpy(dtype=float), pred.y_prob.to_numpy(dtype=float)
    groups = pred.patient_id.fillna(pred.isic_id).astype(str).to_numpy()
    patients = np.unique(groups)
    fold_of = dict(zip(patients, np.random.default_rng(seed).permutation(len(patients)) % 2))
    fold = np.array([fold_of[g] for g in groups])
    calibrated, temps = np.empty_like(p), []
    for k in (0, 1):
        t = fit_temperature(y[fold != k], p[fold != k])
        calibrated[fold == k] = scale(p[fold == k], t)
        temps.append(t)
    return {"ece_raw": ece(y, p), "ece_ts": ece(y, calibrated), "temperature": float(np.mean(temps))}


def fairness_gap(run: str) -> float:
    path = os.path.join(run, "final_round", "results", "metrics", "skin_type_metrics.csv")
    if not os.path.isfile(path):
        return np.nan
    sens = pd.read_csv(path).sensitivity.dropna()
    return float(sens.max() - sens.min()) if len(sens) > 1 else np.nan


def late_round_gain(run: str) -> float:
    m = pd.read_csv(os.path.join(run, "metrics.csv"))
    if "eval_roc_auc" not in m or m["round"].max() < 30:
        return np.nan
    late, mid = m[m["round"].between(21, 30)].eval_roc_auc, m[m["round"].between(11, 20)].eval_roc_auc
    return float(late.mean() - mid.mean())


def gini(counts: np.ndarray) -> float:
    c = np.sort(counts.astype(float))
    n = len(c)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(c) / (n * c.sum())) if c.sum() else np.nan


def _selection_counts(selected_sets) -> dict:
    counts = np.zeros(N_CLIENTS)
    for ids in selected_sets:
        for cid in ids or []:
            counts[int(cid)] += 1
    return {"min_selections": float(counts.min()), "never_selected": int((counts == 0).sum()),
            "selection_gini": gini(counts)}


def _op_selected_sets(run: str) -> list:
    """Selected clients per round from logs/op.jsonl, for runs finished before stragglers.jsonl was kept."""
    path = os.path.join(run, "logs", "op.jsonl")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as fh:
        recs = [json.loads(line) for line in fh if line.strip()]
    last = {r["round"]: r["selected_clients"] for r in recs if r.get("event") == "round_configure_fit"}
    return list(last.values())


def selection_and_time(run: str) -> dict:
    path = os.path.join(run, "logs", "stragglers.jsonl")
    out = {"min_selections": np.nan, "never_selected": np.nan, "selection_gini": np.nan,
           "sim_hours_window": np.nan, "sim_hours_sync": np.nan}
    if not os.path.isfile(path):
        op_sets = _op_selected_sets(run)
        if op_sets:
            out.update(_selection_counts(op_sets))
        return out
    with open(path, encoding="utf-8") as fh:
        recs = [json.loads(line) for line in fh if line.strip()]
    # A round re-run after a resume is logged twice; keep the last attempt (last summary, last outcome per client).
    summaries = {r["round"]: r for r in recs if r.get("event") == "round_summary"}
    last_outcome = {(r["round"], int(r["client_id"])): r for r in recs
                    if r.get("event") == "client_round_outcome" and r.get("selected")}
    window = sync = 0.0
    for rnd, s in summaries.items():
        rows = [last_outcome[(rnd, int(c))] for c in s.get("selected_set") or [] if (rnd, int(c)) in last_outcome]
        arrived = [o["latency_seconds"] for o in rows if o["response_classification"] != "failed"
                   and o.get("latency_seconds") is not None]
        on_time = [o["latency_seconds"] for o in rows if o["response_classification"] == "on_time"
                   and o.get("latency_seconds") is not None]
        waited = s.get("straggler_set") or s.get("failed_set")
        window += float(s["adaptive_window_w_t"]) if waited else max(on_time, default=0.0)
        sync += SYNC_TIMEOUT_S if s.get("failed_set") else max(arrived, default=0.0)
    out.update(_selection_counts(s.get("selected_set") for s in summaries.values()),
               sim_hours_window=window / 3600, sim_hours_sync=sync / 3600)
    return out


def main() -> None:
    os.makedirs(PROCESSED, exist_ok=True)
    rows = []
    for run in sorted(glob.glob(os.path.join(RAW, "*", "seed_*"))):
        if not os.path.isfile(os.path.join(run, "DONE.json")):
            continue
        exp, seed = run.split(os.sep)[-2], int(run.split(os.sep)[-1].split("_")[1])
        rows.append({"experiment": exp, "seed": seed, "fairness_gap": fairness_gap(run),
                     "late_round_gain": late_round_gain(run), **selection_and_time(run),
                     **calibration(predictions(run))})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(PROCESSED, "blueprint_metrics.csv"), index=False)
    print(f"blueprint metrics for {len(df)} runs -> {os.path.join(PROCESSED, 'blueprint_metrics.csv')}")


if __name__ == "__main__":
    main()
