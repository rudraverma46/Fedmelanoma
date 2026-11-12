"""Builds every table, curve and summary for a finished run, from its logs only.

Writes under results/<run_id>/: the reference's plots (monitoring.plotting), curves/, tables/,
schedule_comparison.json and report_data.json. Nothing here is typed in by hand, and any run,
complete or not, can be summarised: `completed_rounds` states how far it got.

Usage: python evaluation/build_report.py <run_id>
"""
from __future__ import annotations

import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding="utf-8")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from monitoring.experiment_logging import read_jsonl  # noqa: E402
from monitoring.plotting import generate_all_plots  # noqa: E402

INK, MUTED, GRID, SERIES = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"


def line_chart(x, y, title, ylabel, path, best_round=None):
    """One series per chart: the title names it, so no legend is needed."""
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(x, y, color=SERIES, linewidth=2, marker="o", markersize=3.5)
    if best_round is not None:
        ax.axvline(best_round, color=MUTED, linewidth=1, linestyle="--")
        ax.annotate(f"best checkpoint, round {best_round}", (best_round, 0.82), xycoords=("data", "axes fraction"),
                    xytext=(4, 0), textcoords="offset points", fontsize=8, color=MUTED)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel("Federated round", color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def compare_with_plan(summaries, plan):
    rows, mismatches = [], 0
    for got in summaries:
        want = plan["schedule"][got["round"] - 1]
        same = (got["selected_set"] == want["selected"] and got["failed_set"] == want["failed"]
                and got["straggler_set"] == want["new_stragglers"] and sorted(got["on_time_set"]) == want["aggregated"]
                and abs(got["adaptive_window_w_t"] - want["window_after_round"]) < 1e-9)
        mismatches += not same
        rows.append({"round": got["round"], "matches_plan": same})
    return {"rounds_compared": len(rows), "mismatches": mismatches, "rows": rows}


def main() -> None:
    run_id = sys.argv[1]
    out = os.path.join("results", run_id)
    curves, tables = os.path.join(out, "curves"), os.path.join(out, "tables")
    os.makedirs(curves, exist_ok=True)
    os.makedirs(tables, exist_ok=True)

    rounds = list(read_jsonl(f"logs/rounds/{run_id}.jsonl"))
    clients = pd.DataFrame(read_jsonl(f"logs/clients/{run_id}.jsonl"))
    straggler = list(read_jsonl(f"logs/op/stragglers/{run_id}.jsonl"))
    summaries = [r for r in straggler if r["event"] == "round_summary"]
    outcomes = pd.DataFrame([r for r in straggler if r["event"] == "client_round_outcome"])
    audit = list(read_jsonl(f"logs/audit/{run_id}.jsonl"))
    math_log = list(read_jsonl(f"logs/math/{run_id}.jsonl"))
    with open("results/schedule/planned_schedule.json", encoding="utf-8") as fh:
        plan = json.load(fh)

    # evaluate.py writes subgroup tables under its working directory; the reference's plotting reads
    # them from results/<run_id>/metrics, so hand it the best checkpoint's tables.
    subgroup_source = os.path.join(out, "final_best", "results", "metrics")
    if os.path.isdir(subgroup_source):
        shutil.copytree(subgroup_source, os.path.join(out, "metrics"), dirs_exist_ok=True)

    plots = generate_all_plots(run_id)

    per_round = pd.DataFrame([{
        "round": r["round"], "seconds": r["seconds"], "train_loss": r["fit"].get("train_loss"),
        "clients_aggregated": r["fit"].get("fedmelanoma_clients_aggregated"),
        "new_stragglers": r["fit"].get("fedmelanoma_new_stragglers_buffered"),
        "eval_1_minus_roc_auc": r["eval_loss"], "eval_focal_loss": r["eval"].get("eval_focal_loss"),
        **{k: r["eval"].get(k) for k in ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1", "specificity", "balanced_accuracy", "mcc")},
    } for r in rounds])
    per_round.to_csv(os.path.join(tables, "per_round_metrics.csv"), index=False)

    best = None
    for event in audit:
        if event.get("event") == "checkpoint_enriched" and event.get("became_best_checkpoint"):
            best = {"round": event["round"], "roc_auc": event["roc_auc"]}
    for r in math_log:  # the audit log survives a resume unfiltered, so confirm against kept rounds
        if best and r.get("event") == "round_evaluate" and r["round"] == best["round"]:
            best["roc_auc"] = r["eval_roc_auc"]
    best_round = best["round"] if best else None

    x = per_round["round"]
    line_chart(x, per_round.train_loss, "Training loss (selected clients, last local epoch)", "Class-weighted focal loss", os.path.join(curves, "training_loss.png"))
    line_chart(x, per_round.eval_focal_loss, "Evaluation focal loss (selected clients' test partitions)", "Class-weighted focal loss", os.path.join(curves, "evaluation_focal_loss.png"), best_round)
    line_chart(x, per_round.eval_1_minus_roc_auc, "Evaluation loss as the reference defines it", "1 - ROC-AUC", os.path.join(curves, "evaluation_loss_reference.png"), best_round)
    line_chart(x, per_round.roc_auc, "Evaluation ROC-AUC (selected clients' test partitions)", "ROC-AUC", os.path.join(curves, "evaluation_roc_auc.png"), best_round)

    # Per-client participation, latency and outcome
    participation = outcomes.groupby(["client_id", "tier"]).agg(
        selected=("round", "count"),
        on_time=("response_classification", lambda s: int((s == "on_time").sum())),
        late=("response_classification", lambda s: int((s == "straggler").sum())),
        failed=("response_classification", lambda s: int((s == "failed").sum())),
        latency_median=("latency_seconds", "median"), latency_max=("latency_seconds", "max"),
        final_R=("reliability_R_after", "last"), final_N_eff=("n_eff_after", "last"), D=("diversity_D_static", "first"),
    ).reset_index()
    aggregated = {c: sum(c in s["on_time_set"] for s in summaries) for c in participation.client_id}
    participation["aggregated"] = participation.client_id.map(aggregated)
    names = {h["client_id"]: h["hospital"] for h in plan["hospitals"]}
    participation.insert(1, "hospital", participation.client_id.map(names))
    participation.to_csv(os.path.join(tables, "client_participation.csv"), index=False)
    outcomes.to_csv(os.path.join(tables, "client_round_outcomes.csv"), index=False)

    fits = clients[clients.event == "client_fit"]
    fits.groupby(["client_id", "hospital"]).agg(fits=("round", "count"), mean_train_loss=("train_loss", "mean"), mean_seconds=("wall_seconds", "mean"), gpu_peak_mb=("gpu_peak_mb", "max")).reset_index().to_csv(os.path.join(tables, "client_training.csv"), index=False)

    windows = pd.Series([s["adaptive_window_w_t"] for s in summaries])
    stats = {
        "completed_rounds": len(rounds),
        "empty_rounds": int(sum(s["n_aggregated"] == 0 for s in summaries)),
        "quorum_label_rounds": int(sum(s["n_on_time"] >= s["quorum_threshold"] for s in summaries)),
        "mean_selected": float(np.mean([s["n_selected"] for s in summaries])),
        "mean_aggregated": float(np.mean([s["n_aggregated"] for s in summaries])),
        "mean_failed": float(np.mean([s["n_failed"] for s in summaries])),
        "mean_new_stragglers": float(np.mean([s["n_straggler"] for s in summaries])),
        "window_mean": float(windows.mean()), "window_min": float(windows.min()), "window_max": float(windows.max()),
    }
    pd.DataFrame([stats]).to_csv(os.path.join(tables, "straggler_statistics.csv"), index=False)

    comparison = compare_with_plan(summaries, plan)
    with open(os.path.join(out, "schedule_comparison.json"), "w", encoding="utf-8") as fh:
        json.dump(comparison, fh, indent=1)

    hash_failures = sum(1 for r in math_log if r.get("event") == "client_fit_returned" and r.get("client_received_matches_broadcast") is not True)
    broken_chain = sum(1 for r in math_log if r.get("event") == "round_broadcast" and r.get("broadcast_matches_previous_round_output") is False)

    resources = pd.read_csv(f"logs/resources/{run_id}.csv")
    finals = {}
    for tag in ("final_best", "final_round50"):
        path = os.path.join(out, tag, "metrics.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                finals[tag] = json.load(fh)
            finals[tag]["thresholds"] = pd.read_csv(os.path.join(out, tag, "threshold_analysis_requested.csv")).merge(
                pd.read_csv(os.path.join(out, tag, "threshold_confusion_requested.csv")), on="threshold").to_dict("records")
    if finals:
        keys = ["roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1", "specificity", "npv", "balanced_accuracy", "mcc", "tn", "fp", "fn", "tp"]
        pd.DataFrame({tag: {k: m[k] for k in keys} for tag, m in finals.items()}).to_csv(os.path.join(tables, "final_metrics.csv"))

    report = {
        "run_id": run_id, "stats": stats, "best": best, "finals": finals, "plots_written": plots,
        "schedule_mismatches": comparison["mismatches"], "rounds_compared": comparison["rounds_compared"],
        "hash_failures": hash_failures, "broken_hash_chain": broken_chain,
        "resumes": sum(1 for e in audit if e.get("event") == "resumed"),
        "safe_stops": sum(1 for e in audit if e.get("event") == "safe_stop"),
        "exceptions": [e.get("exception_message") for e in audit if e.get("event") == "exception"],
        "run_end_logged": any(e.get("event") == "run_end" for e in audit),
        "train_seconds": float(per_round.seconds.sum()),
        "resources": {"commit_free_min_gb": float(resources.commit_free_gb.min()), "ram_free_min_gb": float(resources.ram_free_gb.min()),
                      "process_rss_max_gb": float(resources.process_rss_gb.max()), "gpu_mem_max_mb": float(pd.to_numeric(resources.gpu_mem_mb, errors="coerce").max()),
                      "gpu_util_mean_pct": float(pd.to_numeric(resources.gpu_util_pct, errors="coerce").mean()), "gpu_temp_max_c": float(pd.to_numeric(resources.gpu_temp_c, errors="coerce").max()),
                      "commit_limit_max_gb": float(resources.commit_limit_gb.max())},
    }
    with open(os.path.join(out, "report_data.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=str)
    print(json.dumps({k: v for k, v in report.items() if k != "finals"}, indent=1, default=str))


if __name__ == "__main__":
    main()
