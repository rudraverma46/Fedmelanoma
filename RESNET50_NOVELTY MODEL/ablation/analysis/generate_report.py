"""Builds results/reports/ablation_report.md from collected runs (aggregate -> statistics -> plots).

    python -m ablation.analysis.generate_report

Every number comes from results/processed. Missing runs are listed, never estimated. Finding
sentences are generated from the statistics, so they cannot claim more than the data shows.
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from ablation.analysis import ablation_plots, aggregate_metrics, statistical_analysis  # noqa: E402
from ablation.analysis.aggregate_metrics import PROCESSED  # noqa: E402

REPORTS = os.path.join(os.path.dirname(PROCESSED), "reports")
DESCRIPTIONS = {
    "E00_full": "Full method: Boltzmann diversity-aware selection, c_i = size·δ^s·(1+γD), straggler buffer, focal loss",
    "E01_random_selection": "Selection replaced by uniform random choice of 6 (R_i, N_eff still updated)",
    "E02_selection_no_diversity": "Selection score without the diversity term (γ_sel = 0, score = R_i)",
    "E03_agg_no_diversity": "Aggregation without the diversity bonus (γ_agg = 0; size and staleness kept)",
    "E04_fedavg_weights": "Aggregation weights n_i/n (no staleness discount, no diversity)",
    "E05_no_straggler_buffer": "Late updates discarded instead of buffered (R_i, N_eff updates unchanged)",
    "E06_no_focal": "Class-weighted cross-entropy instead of focal loss (γ_f = 0)",
    "R01_fedavg_reference": "Reference FedAvg: uniform selection + n_i/n weights + no buffer (not an ablation)",
}


def pm(mean, sd, digits=4) -> str:
    if pd.isna(mean):
        return "n/a"
    return f"{mean:.{digits}f} ± {sd:.{digits}f}" if pd.notna(sd) else f"{mean:.{digits}f}"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def finding_sentence(row) -> str:
    direction = "lowered" if row.delta_mean < 0 else "raised"
    strength = ("statistically significant after Holm correction" if row.finding == "significant"
                else "not significant; descriptive only")
    return (f"- **{row.experiment}**: {direction} final-round ROC-AUC by {abs(row.delta_mean):.4f} "
            f"(95% CI {row.ci95_low:+.4f} to {row.ci95_high:+.4f}, n = {row.n_paired_seeds} paired seeds, "
            f"worse in {row.seeds_worse}/{row.n_paired_seeds} seeds; {strength}).")


def main() -> None:
    aggregate_metrics.main()
    combined = pd.read_csv(os.path.join(PROCESSED, "combined_metrics.csv"))
    if combined.empty:
        raise SystemExit("No completed runs yet; nothing to report.")
    statistical_analysis.main()
    ablation_plots.main()
    summary = pd.read_csv(os.path.join(PROCESSED, "experiment_summary.csv"))
    stats_df = pd.read_csv(os.path.join(PROCESSED, "statistical_summary.csv"))
    boot = pd.read_csv(os.path.join(PROCESSED, "bootstrap_delta_auc.csv"))
    subsets = pd.read_csv(os.path.join(PROCESSED, "subsets.csv"))
    with open(os.path.join(PROCESSED, "validation.json"), encoding="utf-8") as fh:
        checks = json.load(fh)
    with open(os.path.join(ROOT, "ablation", "experiments.yaml"), encoding="utf-8") as fh:
        registry = yaml.safe_load(fh)
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "experiment_config.yaml"), encoding="utf-8"))
    first = combined.iloc[0]

    comparison = pd.DataFrame({
        "Experiment": summary.experiment,
        "ROC-AUC": [pm(r.final_round_roc_auc_mean, r.final_round_roc_auc_sd) for r in summary.itertuples()],
        "PR-AUC": [pm(r.final_round_pr_auc_mean, r.final_round_pr_auc_sd) for r in summary.itertuples()],
        "F1 @0.5": [pm(r.final_round_f1_mean, r.final_round_f1_sd) for r in summary.itertuples()],
        "Balanced acc.": [pm(r.final_round_balanced_accuracy_mean, r.final_round_balanced_accuracy_sd) for r in summary.itertuples()],
        "Updates aggregated (GB)": [pm(r.gb_aggregated_uplink_mean, r.gb_aggregated_uplink_sd, 1) for r in summary.itertuples()],
        "Runtime (h)": [pm(r.train_hours_mean, r.train_hours_sd, 2) for r in summary.itertuples()],
        "Seeds": summary.n_seeds,
    })
    os.makedirs(REPORTS, exist_ok=True)
    comparison.to_csv(os.path.join(REPORTS, "ablation_comparison.csv"), index=False)

    roc = stats_df[stats_df.metric == "roc_auc"].sort_values("delta_mean")
    other = stats_df[stats_df.metric != "roc_auc"].copy()
    other_tbl = other.assign(**{"Δ mean [95% CI]": [f"{r.delta_mean:+.4f} [{r.ci95_low:+.4f}, {r.ci95_high:+.4f}]"
                                                   for r in other.itertuples()]})[
        ["experiment", "metric", "Δ mean [95% CI]", "p_holm", "finding"]]
    sub = subsets.groupby(["experiment", "subset"]).roc_auc.agg(["mean", "std"]).reset_index()
    sub_tbl = sub.pivot(index="experiment", columns="subset", values="mean").round(4).reset_index()
    boot_tbl = boot.assign(interval=[f"{r.delta_auc:+.4f} [{r.ci95_low:+.4f}, {r.ci95_high:+.4f}]" for r in boot.itertuples()])[
        ["experiment", "seed", "interval", "excludes_zero", "patients"]] if len(boot) else pd.DataFrame()
    best_tbl = summary[["experiment", "final_best_roc_auc_mean", "final_best_roc_auc_sd"]].round(4)
    incomplete = checks.get("incomplete_runs", {})
    best_arm = summary.sort_values("final_round_roc_auc_mean").iloc[-1]

    parts = [
        "# FedMelanoma ResNet-50 ablation report", "",
        f"Generated from {len(combined)} completed runs. Audit, discrepancy report and design rationale: "
        "`ablation/AUDIT_AND_PLAN.md`.", "",
        "## 1. Methodology",
        "One-factor-at-a-time ablation of the components that exist in the code. Every arm uses the same "
        f"hospital partitions (split id `{first.split_id}`), the same seeds {registry['common']['seeds']}, "
        f"{registry['common']['num_server_rounds']} rounds, {cfg['federation']['clients_per_round']} clients per round, "
        f"{cfg['federation']['local_epochs']} local epochs, batch {cfg['local_training']['batch_size']}, AdamW lr "
        f"{cfg['local_training']['learning_rate']} (reset every round). The simulated straggler environment uses common random "
        "numbers per (seed, round, client) (harness H1), so all arms meet the same failures and latencies. "
        "The primary metric is ROC-AUC of the round-30 global model on the pooled test sets of all 10 hospitals; "
        "the seed is the unit of replication and comparisons are paired by seed.", "",
        "## 2. Baseline", DESCRIPTIONS["E00_full"] + ".", "",
        "## 3. Ablation configurations",
        md_table(pd.DataFrame({"Experiment": list(registry["experiments"]),
                               "Change": [DESCRIPTIONS.get(e, "") for e in registry["experiments"]],
                               "Overrides": [json.dumps(v) for v in registry["experiments"].values()]})), "",
        "## 4. Dataset and partitions",
        "10 hospital partitions (`fedmel01`), 7,788 training and 1,952 test images, image-level split. "
        "**Leakage:** most test rows come from patients (and many from lesions) that also appear in training, so "
        "absolute scores are optimistic; Section 7 adds unseen-patient and unseen-lesion scores. Relative "
        "comparisons are unaffected in kind because every arm uses the same split.", "",
        "## 5. Hardware and execution",
        f"GPU(s): {', '.join(checks.get('gpus', []))}; torch {first.torch}, Flower {first.flwr}; code hash "
        f"`{first.code_hash}`. Runs executed as Kaggle background jobs, one run per GPU, resumed from per-round "
        "state across jobs. Validation: " + ", ".join(f"{k} = {v}" for k, v in checks.items()
                                                     if k in ("one_config_hash_per_experiment", "single_split_id",
                                                              "single_code_hash", "single_software",
                                                              "all_runs_full_rounds", "duplicate_runs")) + ".", "",
        "## 6. Results (mean ± SD over seeds, final-round model)", md_table(comparison), "",
        "Test-selected best checkpoint (optimistic: chosen on the evaluation clients' test splits), for reference only:",
        md_table(best_tbl), "",
        "![Final ROC-AUC](../plots/accuracy_comparison.png)", "",
        "## 7. Statistical analysis",
        "Paired differences vs E00 over shared seeds (t interval; Holm correction across arms per metric).", "",
        *[finding_sentence(r) for r in roc.itertuples()], "",
        "![ΔROC-AUC](../plots/delta_roc_auc.png)", "",
        "Other metrics:", md_table(other_tbl.round(4)), "",
        "Per seed, patient-clustered bootstrap of ΔROC-AUC on identical test rows (2,000 resamples):",
        md_table(boot_tbl) if len(boot_tbl) else "_no paired runs yet_", "",
        "ROC-AUC by test subset (mean over seeds):", md_table(sub_tbl), "",
        "## 8. Visualisations",
        "![Convergence](../plots/convergence.png)", "",
        "Per-round curves use the reference evaluation, which scores only the clients selected that round, so "
        "they are noisy and not a fixed test set.", "",
        "![Loss](../plots/loss_comparison.png)", "",
        "E06 trains with cross-entropy, so its loss curve is on a different scale from the focal-loss arms.", "",
        "![Per hospital](../plots/per_hospital_auc.png)", "",
        "![Communication](../plots/communication_cost.png)", "",
        "Every arm transmits the same volume (6 selected clients receive and return the full model each round); "
        "arms differ only in how many returned updates reach aggregation.", "",
        "![Runtime](../plots/runtime_comparison.png)", "",
        "## 9. Failed or incomplete experiments",
        md_table(pd.DataFrame([{"run": k, "state": v.get("state"), "rounds": v.get("rounds_done"),
                                "reason": v.get("reason")} for k, v in incomplete.items()]))
        if incomplete else "None.", "",
        "## 10. Interpretation",
        f"Highest mean final-round ROC-AUC: **{best_arm.experiment}** "
        f"({pm(best_arm.final_round_roc_auc_mean, best_arm.final_round_roc_auc_sd)}). "
        + ("At least one component change is statistically significant (see Section 7)."
           if (roc.finding == "significant").any() else
           "No component change reaches significance after correction with the available seeds; the "
           "differences in Section 7 are descriptive."), "",
        "## 11. Limitations",
        "- Three seeds give low power: small effects cannot be confirmed or excluded.",
        "- Image-level split with patient and lesion overlap inflates absolute scores (Section 4).",
        "- Stragglers, failures and latency are simulated; all clients actually train synchronously on one GPU.",
        "- Hospital reliability tiers are assigned round-robin by name, not measured.",
        "- Software differs from the original lock file (Python 3.13, torch 2.11 on Kaggle vs 3.11 / 2.2.1); "
        "all arms share it, so comparisons are internal to this environment.", "",
        "## 12. Conclusions",
        "Conclusions are limited to Section 7: changes labelled significant are supported; all others are "
        "descriptive and need more seeds before a claim is made.", "",
    ]
    path = os.path.join(REPORTS, "ablation_report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    print(f"report: {path}")


if __name__ == "__main__":
    main()
