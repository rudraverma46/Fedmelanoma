"""Report figures from results/processed into results/plots (PNG, light surface).

Colours: the validated 8-slot categorical palette in fixed order; an arm keeps its colour in every
figure. Text uses ink tokens, never series colours. One y-axis per chart.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from ablation.analysis.aggregate_metrics import PROCESSED  # noqa: E402

PLOTS = os.path.join(os.path.dirname(PROCESSED), "plots")
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BASELINE = "E00_full"


def arm_colors() -> dict:
    with open(os.path.join(ROOT, "ablation", "experiments.yaml"), encoding="utf-8") as fh:
        order = list(yaml.safe_load(fh)["experiments"])
    return {exp: PALETTE[i % len(PALETTE)] for i, exp in enumerate(order)}


def style(ax, title: str, ylabel: str = "", xlabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=10)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_xlabel(xlabel, color=INK2)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def figure(w=9, h=5):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def save(fig, name: str) -> None:
    os.makedirs(PLOTS, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, name), facecolor=SURFACE)
    plt.close(fig)


def convergence(per_round: pd.DataFrame, colors: dict, col: str, title: str, ylabel: str, name: str) -> None:
    fig, ax = figure()
    for exp, g in per_round.groupby("experiment"):
        stat = g.groupby("round")[col].agg(["mean", "std"])
        ax.plot(stat.index, stat["mean"], color=colors[exp], linewidth=2, label=exp)
        ax.fill_between(stat.index, stat["mean"] - stat["std"].fillna(0), stat["mean"] + stat["std"].fillna(0),
                        color=colors[exp], alpha=0.12, linewidth=0)
    style(ax, title, ylabel, "Communication round")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, ncol=2)
    save(fig, name)


def delta_dots(stats_df: pd.DataFrame, colors: dict, metric: str, name: str) -> None:
    d = stats_df[stats_df.metric == metric].sort_values("delta_mean")
    if d.empty:
        return
    fig, ax = figure(8, 0.6 * len(d) + 1.6)
    y = np.arange(len(d))
    ax.axvline(0, color=INK2, linewidth=1)
    for i, row in enumerate(d.itertuples()):
        ax.plot([row.ci95_low, row.ci95_high], [i, i], color=colors[row.experiment], linewidth=2, solid_capstyle="round")
        ax.plot(row.delta_mean, i, "o", markersize=8, color=colors[row.experiment], markeredgecolor=SURFACE,
                markeredgewidth=2)
    ax.set_yticks(y, [f"{r.experiment}  ({r.finding})" for r in d.itertuples()])
    style(ax, f"Change in final-round {metric.replace('_', '-').upper()} vs {BASELINE} (paired by seed, 95% CI)",
          "", f"Δ {metric} (arm − {BASELINE})")
    save(fig, name)


def final_metric_dots(combined: pd.DataFrame, colors: dict, name: str) -> None:
    fig, ax = figure(8, 0.6 * combined.experiment.nunique() + 1.6)
    order = combined.groupby("experiment").final_round_roc_auc.mean().sort_values().index
    for i, exp in enumerate(order):
        vals = combined[combined.experiment == exp].final_round_roc_auc
        ax.plot(vals, [i] * len(vals), "o", markersize=8, color=colors[exp], alpha=0.45, markeredgewidth=0)
        ax.plot(vals.mean(), i, "D", markersize=9, color=colors[exp], markeredgecolor=SURFACE, markeredgewidth=2)
    ax.set_yticks(range(len(order)), list(order))
    style(ax, "Final-round ROC-AUC on the pooled test set (dots: seeds, diamond: mean)", "", "ROC-AUC")
    save(fig, name)


def hbars(values: pd.Series, colors: dict, title: str, xlabel: str, name: str, fmt: str) -> None:
    values = values.sort_values()
    fig, ax = figure(8, 0.55 * len(values) + 1.6)
    ax.barh(range(len(values)), values.to_numpy(), color=[colors[e] for e in values.index], height=0.6)
    for i, v in enumerate(values.to_numpy()):
        ax.text(v, i, f"  {v:{fmt}}", va="center", color=INK2, fontsize=9)
    ax.set_yticks(range(len(values)), list(values.index))
    style(ax, title, "", xlabel)
    save(fig, name)


def hospital_heatmap(per_hospital: pd.DataFrame, name: str) -> None:
    table = per_hospital.groupby(["hospital", "experiment"]).roc_auc.mean().unstack()
    fig, ax = figure(10, 0.45 * len(table) + 2)
    im = ax.imshow(table.to_numpy(), cmap="Blues", vmin=np.nanmin(table.to_numpy()), vmax=1.0, aspect="auto")
    ax.set_xticks(range(table.shape[1]), table.columns, rotation=30, ha="right")
    ax.set_yticks(range(table.shape[0]), [h.replace("_", " ") for h in table.index])
    for (i, j), v in np.ndenumerate(table.to_numpy()):
        if not np.isnan(v):
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    color=SURFACE if v > np.nanmean(table.to_numpy()) else INK)
    style(ax, "Per-hospital ROC-AUC, final round (mean over seeds)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.03).outline.set_visible(False)
    save(fig, name)


def main() -> None:
    colors = arm_colors()
    combined = pd.read_csv(os.path.join(PROCESSED, "combined_metrics.csv"))
    per_round = pd.read_csv(os.path.join(PROCESSED, "per_round.csv"))
    stats_df = pd.read_csv(os.path.join(PROCESSED, "statistical_summary.csv"))
    per_hospital = pd.read_csv(os.path.join(PROCESSED, "per_hospital.csv"))
    convergence(per_round, colors, "eval_roc_auc", "Per-round ROC-AUC on the selected clients' test splits (mean ± SD over seeds)",
                "ROC-AUC", "convergence.png")
    convergence(per_round, colors, "train_loss", "Per-round training loss (mean ± SD over seeds)", "Focal / CE loss",
                "loss_comparison.png")
    final_metric_dots(combined, colors, "accuracy_comparison.png")
    for metric in ("roc_auc", "pr_auc", "balanced_accuracy"):
        delta_dots(stats_df, colors, metric, f"delta_{metric}.png")
    g = combined.groupby("experiment")
    hbars(g.train_hours.mean(), colors, "Training wall time per run (mean over seeds)", "hours", "runtime_comparison.png", ".2f")
    hbars(g.gb_aggregated_uplink.mean(), colors, "Client updates that reached aggregation (GB, mean over seeds)",
          "GB", "communication_cost.png", ".1f")
    hospital_heatmap(per_hospital, "per_hospital_auc.png")
    print(f"plots in {PLOTS}")


if __name__ == "__main__":
    main()
