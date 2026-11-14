"""Seed-level summaries and paired comparisons of every arm against E00.

The unit of replication is the seed (training rounds are never treated as repetitions).
  experiment_summary.csv   mean ± SD over seeds per arm
  statistical_summary.csv  per arm and metric: paired difference vs E00 over shared seeds, 95 % t CI,
                           paired t-test p, Holm-adjusted p across arms (per metric), and a label
                           ("significant" only when Holm p < 0.05; otherwise "descriptive")
  bootstrap_delta_auc.csv  per seed: ΔROC-AUC (arm − E00) on the same test rows, 95 % interval from
                           2,000 patient-clustered resamples (rows of one patient move together)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from ablation.analysis.aggregate_metrics import PROCESSED, RAW, predictions  # noqa: E402

BASELINE = "E00_full"
METRICS = ["roc_auc", "pr_auc", "f1", "sensitivity", "specificity", "balanced_accuracy"]
RESAMPLES = 2000


def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values().index
    m, running, out = p.notna().sum(), 0.0, pd.Series(np.nan, index=p.index)
    for i, idx in enumerate([i for i in order if pd.notna(p[i])]):
        running = max(running, min(1.0, (m - i) * p[idx]))
        out[idx] = running
    return out


def summary(combined: pd.DataFrame) -> pd.DataFrame:
    cols = [f"final_round_{m}" for m in METRICS] + ["final_best_roc_auc", "train_hours", "gb_transmitted",
                                                    "gb_aggregated_uplink", "clients_aggregated_mean"]
    g = combined.groupby("experiment")[cols]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_sd"))
    out["n_seeds"] = combined.groupby("experiment").seed.nunique()
    return out.reset_index()


def paired(combined: pd.DataFrame) -> pd.DataFrame:
    base = combined[combined.experiment == BASELINE].set_index("seed")
    rows = []
    for exp, g in combined[combined.experiment != BASELINE].groupby("experiment"):
        g = g.set_index("seed")
        seeds = sorted(set(g.index) & set(base.index))
        for m in METRICS:
            col = f"final_round_{m}"
            d = (g.loc[seeds, col] - base.loc[seeds, col]).to_numpy(dtype=float)
            n = len(d)
            mean = float(d.mean()) if n else np.nan
            sd = float(d.std(ddof=1)) if n > 1 else np.nan
            half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else np.nan
            p = float(stats.ttest_rel(g.loc[seeds, col], base.loc[seeds, col]).pvalue) if n > 1 and sd > 0 else np.nan
            rows.append({"experiment": exp, "metric": m, "n_paired_seeds": n, "seeds": seeds, "delta_mean": mean,
                         "delta_sd": sd, "ci95_low": mean - half, "ci95_high": mean + half, "p_paired_t": p,
                         "seeds_worse": int((d < 0).sum()), "seeds_better": int((d > 0).sum())})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = out.groupby("metric", group_keys=False).p_paired_t.apply(holm)
        out["finding"] = np.where(out.p_holm < 0.05, "significant", "descriptive")
    return out


def bootstrap(combined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for exp, seed in combined[combined.experiment != BASELINE][["experiment", "seed"]].itertuples(index=False):
        base_dir, arm_dir = (os.path.join(RAW, e, f"seed_{seed}") for e in (BASELINE, exp))
        if not os.path.isdir(base_dir):
            continue
        b, a = predictions(base_dir), predictions(arm_dir)
        j = b.merge(a[["isic_id", "y_prob"]], on="isic_id", suffixes=("_base", "_arm"), validate="one_to_one")
        groups = j.patient_id.fillna(j.isic_id).to_numpy()
        uniq, inv = np.unique(groups, return_inverse=True)
        members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
        rng = np.random.default_rng(seed)
        deltas = []
        for _ in range(RESAMPLES):
            idx = np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))])
            y = j.y_true.to_numpy()[idx]
            if y.min() == y.max():
                continue
            deltas.append(roc_auc_score(y, j.y_prob_arm.to_numpy()[idx]) - roc_auc_score(y, j.y_prob_base.to_numpy()[idx]))
        point = roc_auc_score(j.y_true, j.y_prob_arm) - roc_auc_score(j.y_true, j.y_prob_base)
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        rows.append({"experiment": exp, "seed": seed, "delta_auc": point, "ci95_low": lo, "ci95_high": hi,
                     "excludes_zero": bool(lo > 0 or hi < 0), "patients": len(uniq), "resamples": len(deltas)})
    return pd.DataFrame(rows)


def main() -> None:
    combined = pd.read_csv(os.path.join(PROCESSED, "combined_metrics.csv"))
    summary(combined).to_csv(os.path.join(PROCESSED, "experiment_summary.csv"), index=False)
    paired(combined).to_csv(os.path.join(PROCESSED, "statistical_summary.csv"), index=False)
    bootstrap(combined).to_csv(os.path.join(PROCESSED, "bootstrap_delta_auc.csv"), index=False)
    print("wrote experiment_summary.csv, statistical_summary.csv, bootstrap_delta_auc.csv")


if __name__ == "__main__":
    main()
