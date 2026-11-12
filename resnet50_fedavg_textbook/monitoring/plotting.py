"""
monitoring/plotting.py
========================
Experiment plotting/reporting module. Reads the SAVED log files under
logs/{op,math,audit}/<run_id>.jsonl (never in-memory-only variables) and
produces publication-quality PNGs plus the underlying tabular CSVs under
results/<run_id>/plots/ and results/<run_id>/metrics/.

Every function here degrades gracefully: if the columns/rows a plot needs
aren't present in the logs (e.g. no straggler events occurred, or
subgroup metrics weren't logged this run), that one plot is skipped with
a printed note instead of raising -- "only generate graphs for
metrics/data that actually exist."

This module is read-only with respect to selector.py/aggregation.py/
straggler.py/emd.py: it never imports or recomputes their formulas, only
plots values FedMelanomaStrategy already logged from them.
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless, same convention as evaluate.py
import matplotlib.pyplot as plt
import pandas as pd

from monitoring.experiment_logging import read_jsonl

_LINE_KW = dict(marker="o", linewidth=2, markersize=4)


def _save_fig(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved {path}")


def _save_table(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def _skip(name: str, reason: str) -> None:
    print(f"  [plot] skipped {name}: {reason}")


# ===========================================================================
# Log loading
# ===========================================================================

def load_logs(run_id: str, logs_root: str = "logs") -> dict:
    """Load every JSONL log tier for *run_id* into DataFrames (empty if absent)."""
    op = pd.DataFrame(list(read_jsonl(os.path.join(logs_root, "op", f"{run_id}.jsonl"))))
    math_df = pd.DataFrame(list(read_jsonl(os.path.join(logs_root, "math", f"{run_id}.jsonl"))))
    audit = pd.DataFrame(list(read_jsonl(os.path.join(logs_root, "audit", f"{run_id}.jsonl"))))
    straggler = pd.DataFrame(
        list(read_jsonl(os.path.join(logs_root, "op", "stragglers", f"{run_id}.jsonl")))
    )
    return {"op": op, "math": math_df, "audit": audit, "straggler": straggler}


# ===========================================================================
# Generic round-indexed line plot helper
# ===========================================================================

def _plot_metric_vs_round(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    ylabel: str,
    out_path: str,
    round_col: str = "round",
    color: str = "#4c8bf5",
) -> bool:
    if df.empty or value_col not in df.columns or round_col not in df.columns:
        _skip(out_path, f"missing column '{value_col}' or '{round_col}'")
        return False
    sub = df[[round_col, value_col]].dropna().drop_duplicates(subset=[round_col]).sort_values(round_col)
    if sub.empty:
        _skip(out_path, f"no non-null rows for '{value_col}'")
        return False

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sub[round_col], sub[value_col], color=color, **_LINE_KW)
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    _save_fig(fig, out_path)
    _save_table(sub, out_path.replace(".png", ".csv"))
    return True


def _plot_per_client_lines(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    ylabel: str,
    out_path: str,
    round_col: str = "round",
    id_col: str = "client_id",
) -> bool:
    required = {round_col, value_col, id_col}
    if df.empty or not required.issubset(df.columns):
        _skip(out_path, f"missing columns {required - set(df.columns)}")
        return False
    sub = df[[round_col, id_col, value_col]].dropna()
    if sub.empty:
        _skip(out_path, f"no non-null rows for '{value_col}'")
        return False

    fig, ax = plt.subplots(figsize=(8, 5))
    for client_id, group in sub.groupby(id_col):
        group = group.sort_values(round_col)
        ax.plot(group[round_col], group[value_col], marker="o", markersize=3, linewidth=1.5, label=f"C{client_id}")
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="best")
    _save_fig(fig, out_path)
    _save_table(sub, out_path.replace(".png", ".csv"))
    return True


# ===========================================================================
# Global training / evaluation curves
# ===========================================================================

def plot_training_loss(math_df: pd.DataFrame, out_dir: str) -> bool:
    return _plot_metric_vs_round(
        math_df, "fit_train_loss", "Federated Training Loss", "Train Loss (weighted avg)",
        os.path.join(out_dir, "training_loss.png"), color="#e05c5c",
    )


_GLOBAL_EVAL_METRICS = {
    "global_auroc": ("eval_roc_auc", "Global ROC-AUC vs Round", "ROC-AUC"),
    "global_auprc": ("eval_pr_auc", "Global PR-AUC (AUPRC) vs Round", "PR-AUC"),
    "global_sensitivity": ("eval_recall", "Global Sensitivity vs Round", "Sensitivity"),
    "global_specificity": ("eval_specificity", "Global Specificity vs Round", "Specificity"),
    "global_f1": ("eval_f1", "Global Malignant F1 vs Round", "F1"),
    "global_balanced_accuracy": ("eval_balanced_accuracy", "Global Balanced Accuracy vs Round", "Balanced Accuracy"),
}


def plot_global_eval_curves(math_df: pd.DataFrame, out_dir: str) -> List[str]:
    written = []
    for filename, (col, title, ylabel) in _GLOBAL_EVAL_METRICS.items():
        if _plot_metric_vs_round(math_df, col, title, ylabel, os.path.join(out_dir, f"{filename}.png")):
            written.append(filename)
    return written


# ===========================================================================
# Selector / reliability / diversity / exposure
# ===========================================================================

def plot_client_reliability(straggler_df: pd.DataFrame, out_dir: str) -> bool:
    return _plot_per_client_lines(
        straggler_df, "reliability_R_after", "Client Reliability R_i over Rounds", "R_i",
        os.path.join(out_dir, "client_reliability.png"),
    )


def plot_client_effective_interactions(straggler_df: pd.DataFrame, out_dir: str) -> bool:
    return _plot_per_client_lines(
        straggler_df, "n_eff_after", "Client Effective Interaction Count N_eff over Rounds", "N_eff",
        os.path.join(out_dir, "client_effective_interactions.png"),
    )


def plot_diversity_scores(straggler_df: pd.DataFrame, out_dir: str) -> bool:
    """D_i is a static prior (never recomputed per round) -- one bar per client, not a time series."""
    out_path = os.path.join(out_dir, "diversity_scores.png")
    if straggler_df.empty or "diversity_D_static" not in straggler_df.columns:
        _skip(out_path, "no diversity_D_static column logged")
        return False
    per_client = straggler_df.drop_duplicates(subset=["client_id"])[["client_id", "diversity_D_static"]]
    per_client = per_client.sort_values("client_id")
    if per_client.empty:
        _skip(out_path, "no client rows")
        return False

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([f"C{c}" for c in per_client["client_id"]], per_client["diversity_D_static"], color="#27ae60")
    ax.set_ylabel("D_i (static OT diversity prior)")
    ax.set_title("Per-Client Diversity Prior D_i (static)")
    ax.grid(alpha=0.3, axis="y")
    _save_fig(fig, out_path)
    _save_table(per_client, out_path.replace(".png", ".csv"))
    return True


def plot_client_selection_frequency(straggler_df: pd.DataFrame, out_dir: str) -> bool:
    out_path = os.path.join(out_dir, "client_selection_frequency.png")
    if straggler_df.empty or "selected" not in straggler_df.columns:
        _skip(out_path, "no 'selected' column logged")
        return False
    sub = straggler_df[straggler_df["selected"] == True]  # noqa: E712
    if sub.empty:
        _skip(out_path, "no selection events logged")
        return False
    counts = sub.groupby("client_id").size().sort_index()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([f"C{c}" for c in counts.index], counts.values, color="#4c8bf5")
    ax.set_ylabel("Times Selected")
    ax.set_title("Client Selection Frequency")
    ax.grid(alpha=0.3, axis="y")
    _save_fig(fig, out_path)
    _save_table(counts.rename("times_selected").reset_index(), out_path.replace(".png", ".csv"))
    return True


def plot_aggregation_weights(straggler_df: pd.DataFrame, out_dir: str) -> bool:
    out_path = os.path.join(out_dir, "aggregation_weights.png")
    required = {"round", "client_id", "aggregation_weight_c_i"}
    if straggler_df.empty or not required.issubset(straggler_df.columns):
        _skip(out_path, f"missing columns {required - set(straggler_df.columns)}")
        return False
    sub = straggler_df.dropna(subset=["aggregation_weight_c_i"])
    if sub.empty:
        _skip(out_path, "no client ever contributed a non-null c_i")
        return False

    fig, ax = plt.subplots(figsize=(8, 5))
    for client_id, group in sub.groupby("client_id"):
        group = group.sort_values("round")
        ax.plot(group["round"], group["aggregation_weight_c_i"], marker="o", markersize=3, label=f"C{client_id}")
    ax.set_xlabel("Round")
    ax.set_ylabel("Aggregation weight c_i")
    ax.set_title("Per-Client Aggregation Weight c_i over Rounds")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    _save_fig(fig, out_path)
    _save_table(sub[["round", "client_id", "aggregation_weight_c_i"]], out_path.replace(".png", ".csv"))
    return True


# ===========================================================================
# Straggler / window diagnostics
# ===========================================================================

def plot_adaptive_window(math_df: pd.DataFrame, out_dir: str) -> bool:
    return _plot_metric_vs_round(
        math_df, "window_w_t", "Adaptive Aggregation Window W_t over Rounds", "W_t (s)",
        os.path.join(out_dir, "adaptive_window.png"), color="#9b59b6",
    )


def plot_round_participation(math_df: pd.DataFrame, out_dir: str) -> List[str]:
    """Straggler count / on-time count / failed count per round -- three related lines, one figure."""
    out_path = os.path.join(out_dir, "staleness.png")
    required = {"round", "n_on_time", "n_straggler", "n_failed"}
    if math_df.empty or not required.issubset(math_df.columns):
        _skip(out_path, f"missing columns {required - set(math_df.columns)}")
        return []
    sub = math_df[list(required)].dropna(subset=["round"]).sort_values("round")
    if sub.empty:
        _skip(out_path, "no round-level participation rows")
        return []

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sub["round"], sub["n_on_time"], marker="o", label="on-time", color="#27ae60")
    ax.plot(sub["round"], sub["n_straggler"], marker="s", label="straggler (buffered)", color="#e6a817")
    ax.plot(sub["round"], sub["n_failed"], marker="^", label="failed", color="#e05c5c")
    ax.set_xlabel("Round")
    ax.set_ylabel("Client count")
    ax.set_title("Per-Round Client Outcome Counts")
    ax.grid(alpha=0.3)
    ax.legend()
    _save_fig(fig, out_path)
    _save_table(sub, out_path.replace(".png", ".csv"))
    return ["staleness"]


def plot_client_latency(straggler_df: pd.DataFrame, out_dir: str) -> bool:
    return _plot_per_client_lines(
        straggler_df, "latency_seconds", "Per-Client Simulated Response Latency over Rounds", "Latency (s)",
        os.path.join(out_dir, "client_latency.png"),
    )


def plot_broadcast_verification_timeline(math_df: pd.DataFrame, out_dir: str) -> bool:
    out_path = os.path.join(out_dir, "broadcast_verification_timeline.png")
    required = {"round", "broadcast_matches_previous_round_output"}
    if math_df.empty or not required.issubset(math_df.columns):
        _skip(out_path, f"missing columns {required - set(math_df.columns)}")
        return False
    sub = math_df[list(required)].dropna().sort_values("round")
    if sub.empty:
        _skip(out_path, "no broadcast-verification rows")
        return False

    fig, ax = plt.subplots(figsize=(8, 3.5))
    colors = ["#27ae60" if v else "#e05c5c" for v in sub["broadcast_matches_previous_round_output"]]
    ax.bar(sub["round"], [1] * len(sub), color=colors)
    ax.set_yticks([])
    ax.set_xlabel("Round")
    ax.set_title("Broadcast Hash Verification (green = round t received == round t-1 aggregated)")
    _save_fig(fig, out_path)
    _save_table(sub, out_path.replace(".png", ".csv"))
    return True


# ===========================================================================
# Fairness
# ===========================================================================

def plot_fairness_gap(math_df: pd.DataFrame, out_dir: str) -> bool:
    return _plot_metric_vs_round(
        math_df, "fitzpatrick_fairness_gap", "Fitzpatrick Fairness Gap vs Round",
        "max(sensitivity) - min(sensitivity)", os.path.join(out_dir, "fairness_gap.png"), color="#c0392b",
    )


# ===========================================================================
# Subgroup metrics (from saved CSVs, not logs -- these are per-evaluation
# snapshots, produced by evaluate.py / monitoring.subgroup_metrics)
# ===========================================================================

def plot_subgroup_bar(subgroup_csv_path: str, metric_col: str, title: str, out_path: str) -> bool:
    if not os.path.isfile(subgroup_csv_path):
        _skip(out_path, f"{subgroup_csv_path} not found")
        return False
    df = pd.read_csv(subgroup_csv_path)
    if metric_col not in df.columns:
        _skip(out_path, f"'{metric_col}' not in {subgroup_csv_path}")
        return False
    sub = df.dropna(subset=[metric_col])
    if sub.empty:
        _skip(out_path, f"every subgroup has NaN {metric_col}")
        return False

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(sub["subgroup"].astype(str), sub[metric_col], color="#4c8bf5")
    ax.set_ylabel(metric_col)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.3, axis="y")
    _save_fig(fig, out_path)
    return True


def plot_all_subgroup_bars(metrics_dir: str, out_dir: str) -> List[str]:
    """
    Reads results/metrics/{skin_type,image_type,confirm_type}_metrics.csv
    (written by evaluate.py) and produces the sensitivity/AUROC bar charts
    Requirement 7 names explicitly.
    """
    written = []
    specs = [
        ("skin_type_metrics.csv", "sensitivity", "Sensitivity by Fitzpatrick Skin Type", "subgroup_fitzpatrick_sensitivity.png"),
        ("skin_type_metrics.csv", "roc_auc", "ROC-AUC by Fitzpatrick Skin Type", "subgroup_fitzpatrick_auroc.png"),
        ("image_type_metrics.csv", "sensitivity", "Sensitivity by Imaging Modality", "subgroup_modality_sensitivity.png"),
        ("image_type_metrics.csv", "roc_auc", "ROC-AUC by Imaging Modality", "subgroup_modality_auroc.png"),
        ("confirm_type_metrics.csv", "sensitivity", "Sensitivity by Diagnosis Confirmation Type", "subgroup_diagnosis_sensitivity.png"),
        ("confirm_type_metrics.csv", "roc_auc", "ROC-AUC by Diagnosis Confirmation Type", "subgroup_diagnosis_auroc.png"),
    ]
    for csv_name, metric, title, out_name in specs:
        if plot_subgroup_bar(os.path.join(metrics_dir, csv_name), metric, title, os.path.join(out_dir, out_name)):
            written.append(out_name)
    return written


# ===========================================================================
# Top-level orchestration
# ===========================================================================

def generate_all_plots(
    run_id: str,
    logs_root: str = "logs",
    results_root: str = "results",
) -> List[str]:
    """
    Generate every plot that the currently available logs/CSVs support
    for *run_id*. Returns the list of plot names actually written.
    """
    logs = load_logs(run_id, logs_root)
    plots_dir = os.path.join(results_root, run_id, "plots")
    metrics_dir = os.path.join(results_root, run_id, "metrics")
    written: List[str] = []

    if plot_training_loss(logs["math"], plots_dir):
        written.append("training_loss")
    written += plot_global_eval_curves(logs["math"], plots_dir)
    if plot_fairness_gap(logs["math"], plots_dir):
        written.append("fairness_gap")
    if plot_adaptive_window(logs["math"], plots_dir):
        written.append("adaptive_window")
    written += plot_round_participation(logs["math"], plots_dir)
    if plot_broadcast_verification_timeline(logs["math"], plots_dir):
        written.append("broadcast_verification_timeline")

    if plot_client_reliability(logs["straggler"], plots_dir):
        written.append("client_reliability")
    if plot_client_effective_interactions(logs["straggler"], plots_dir):
        written.append("client_effective_interactions")
    if plot_diversity_scores(logs["straggler"], plots_dir):
        written.append("diversity_scores")
    if plot_client_selection_frequency(logs["straggler"], plots_dir):
        written.append("client_selection_frequency")
    if plot_aggregation_weights(logs["straggler"], plots_dir):
        written.append("aggregation_weights")
    if plot_client_latency(logs["straggler"], plots_dir):
        written.append("client_latency")

    written += plot_all_subgroup_bars(metrics_dir, plots_dir)

    return written
