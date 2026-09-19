"""
make_plots.py — Figures for the EfficientNet-B0 + FedProx/FedNova run.

Reads the training log written by train_fedprox_fednova.py and the
hospital partition CSVs, and writes four publication-style figures:

  fig1_training_progress.png  — metrics across federated rounds
  fig2_client_distribution.png — per-hospital size and malignancy rate (Non-IID)
  fig3_confusion_matrix.png   — best round's predictions on the pooled test set
  fig4_client_convergence.png — spread of per-client local training loss

Usage:
    python make_plots.py --log full_run_log.json \
        --data-root /path/to/hospital_partitions --out-dir .

Design notes: metrics are plotted as 0-1 scores on a single shared axis
(never a dual axis). Series colors come from a CVD-validated categorical
order; every series is also direct-labeled so identity never rests on
color alone.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Validated categorical palette (slots 1-4) + surface/ink tokens --------
C_BLUE = "#2a78d6"      # slot 1
C_ORANGE = "#eb6834"    # slot 2
C_AQUA = "#1baf7a"      # slot 3
C_YELLOW = "#eda100"    # slot 4
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e3e2de"
# Sequential blue ramp (for the confusion-matrix heatmap)
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SOFT,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "text.color": INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

HOSPITALS = [
    "CityGeneral_Multispecialty_Hospital",
    "GoldenSun_Public_Health_Screening_Camp",
    "Greenfield_Rural_Community_Health_Center",
    "Lakeside_Regional_Mixed_Hospital",
    "MetroSkin_Dermatology_Research_Clinic",
    "NorthCare_Tertiary_Melanoma_Institute",
    "Pinewood_Oncology_Biopsy_Center",
    "SunRay_Community_Outreach_Clinic",
    "TeleDerm_National_Screening_Network",
    "Westbridge_Academic_Dermoscopy_Unit",
]


def short(name: str) -> str:
    """Compact hospital label for axis ticks."""
    return name.split("_")[0]


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


# ===========================================================================
# Figure 1 — metrics across rounds
# ===========================================================================

def fig_training_progress(history, cm, out_path):
    n_mal = cm[1][0] + cm[1][1]
    n_ben = cm[0][0] + cm[0][1]

    rounds, bal, sens, spec, auc = [], [], [], [], []
    for r in history:
        acc, rec, n = r["accuracy"], r["recall"], r["n_eval"]
        correct_mal = round(rec * n_mal)
        correct_ben = round(acc * n) - correct_mal
        sp = correct_ben / n_ben
        rounds.append(r["round"])
        sens.append(rec)
        spec.append(sp)
        bal.append((rec + sp) / 2)
        auc.append(r["roc_auc"])

    best_i = int(np.argmax(auc))

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    series = [
        ("Balanced accuracy", bal, C_BLUE),
        ("ROC-AUC", auc, C_ORANGE),
        ("Sensitivity", sens, C_AQUA),
        ("Specificity", spec, C_YELLOW),
    ]
    for label, vals, color in series:
        ax.plot(rounds, vals, color=color, linewidth=2.0,
                marker="o", markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=1.2, label=label, zorder=3)

    # Direct labels at the line ends, de-collided: series that converge (here
    # ROC-AUC and specificity both land near 0.96) would otherwise overprint.
    # Text stays in ink; the line running into it carries the identity.
    ends = sorted(((vals[-1], label) for label, vals, _ in series), reverse=True)
    min_gap = 0.030
    placed = []
    for value, label in ends:
        y = value
        if placed and placed[-1][0] - y < min_gap:
            y = placed[-1][0] - min_gap
        placed.append((y, label, value))
    for y, label, value in placed:
        ax.annotate(f" {label}", (rounds[-1], y), color=INK,
                    fontsize=9, va="center", ha="left", zorder=4)
        if abs(y - value) > 1e-9:   # leader line back to the true endpoint
            ax.plot([rounds[-1], rounds[-1] + 0.16], [value, y],
                    color=INK_SOFT, linewidth=0.7, alpha=0.6, zorder=2)

    # Mark the selected model (peak ROC-AUC)
    ax.axvline(rounds[best_i], color=INK_SOFT, linestyle=":", linewidth=1.2, zorder=1)
    ax.annotate(f"best model\n(round {rounds[best_i]})",
                (rounds[best_i], 0.545), color=INK_SOFT, fontsize=8.5,
                ha="center", va="bottom")

    # Trivial baseline reference: predict "benign" for everything.
    ax.axhline(0.5, color=INK_SOFT, linestyle="--", linewidth=1.0, alpha=0.55, zorder=1)
    ax.annotate("all-benign baseline (balanced acc. 0.50)", (rounds[0], 0.512),
                color=INK_SOFT, fontsize=8.5, va="bottom")

    _style(ax)
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Score")
    ax.set_xticks(rounds)
    ax.set_ylim(0.48, 1.02)
    ax.set_xlim(rounds[0] - 0.3, rounds[-1] + 2.9)
    # Legend above the plot so it cannot collide with the baseline note or
    # the end labels inside the panel.
    ax.legend(frameon=False, ncol=4, fontsize=9, labelcolor=INK,
              loc="lower left", bbox_to_anchor=(0, 1.005), handlelength=1.6,
              columnspacing=1.6)
    ax.set_title("Global model performance across federated rounds",
                 color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=34)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ===========================================================================
# Figure 2 — per-hospital Non-IID distribution (two panels, one axis each)
# ===========================================================================

def fig_client_distribution(data_root, out_path):
    names, sizes, mal_rates = [], [], []
    for h in HOSPITALS:
        tr = pd.read_csv(Path(data_root) / h / "metadata_train.csv", low_memory=False)
        names.append(short(h))
        sizes.append(len(tr))
        mal_rates.append((tr["diagnosis_1"].str.lower() == "malignant").mean() * 100)

    order = np.argsort(sizes)[::-1]
    names = [names[i] for i in order]
    sizes = [sizes[i] for i in order]
    mal_rates = [mal_rates[i] for i in order]
    y = np.arange(len(names))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))

    # Panel A — training set size
    ax = axes[0]
    ax.barh(y, sizes, color=C_BLUE, height=0.62, zorder=3)
    for i, v in enumerate(sizes):
        ax.annotate(f" {v:,}", (v, i), va="center", ha="left",
                    fontsize=8.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Training images")
    ax.set_xlim(0, max(sizes) * 1.18)
    _style(ax); ax.grid(axis="y", visible=False)
    ax.set_title("Local dataset size", color=INK, fontsize=11,
                 fontweight="bold", loc="left", pad=10)

    # Panel B — malignancy rate (separate axis, never dual-axed onto panel A)
    ax = axes[1]
    ax.barh(y, mal_rates, color=C_ORANGE, height=0.62, zorder=3)
    for i, v in enumerate(mal_rates):
        ax.annotate(f" {v:.2f}%", (v, i), va="center", ha="left",
                    fontsize=8.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Malignant cases (%)")
    ax.set_xlim(0, max(mal_rates) * 1.25)
    _style(ax); ax.grid(axis="y", visible=False)
    ax.set_title("Local malignancy rate", color=INK, fontsize=11,
                 fontweight="bold", loc="left", pad=10)

    fig.suptitle("Non-IID data distribution across the 10 hospital clients",
                 color=INK, fontsize=12.5, fontweight="bold", x=0.007, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ===========================================================================
# Figure 3 — confusion matrix for the selected model
# ===========================================================================

def fig_confusion_matrix(cm, best_round, out_path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5.8, 4.8))

    # Sequential single-hue encoding, normalised per true-class row so the
    # rare malignant row is readable next to the 1,843-case benign row.
    row_norm = cm / cm.sum(axis=1, keepdims=True)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    ax.imshow(row_norm, cmap=cmap, vmin=0, vmax=1)

    labels = ["Benign", "Malignant"]
    for i in range(2):
        for j in range(2):
            pct = row_norm[i, j] * 100
            ax.text(j, i - 0.09, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=17, fontweight="bold",
                    color="#ffffff" if row_norm[i, j] > 0.55 else INK)
            ax.text(j, i + 0.17, f"{pct:.1f}% of row", ha="center", va="center",
                    fontsize=9,
                    color="#eaf2fd" if row_norm[i, j] > 0.55 else INK_SOFT)

    ax.set_xticks([0, 1]); ax.set_xticklabels([f"Predicted\n{l}" for l in labels], fontsize=10)
    ax.set_yticks([0, 1]); ax.set_yticklabels([f"True\n{l}" for l in labels], fontsize=10)
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=3)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    ax.set_title(f"Predictions on 1,952 held-out images (round {best_round})",
                 color=INK, fontsize=12, fontweight="bold", loc="left", pad=12)
    fig.text(0.5, -0.02,
             f"{tp} of {tp+fn} melanomas caught · {fn} missed · {fp} false alarms",
             ha="center", fontsize=9.5, color=INK_SOFT)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ===========================================================================
# Figure 4 — per-client local loss convergence (median + min/max envelope)
# ===========================================================================

def fig_client_convergence(history, out_path):
    rounds = [r["round"] for r in history]
    per_round = [list(r["client_avg_losses"].values()) for r in history]
    med = [float(np.median(v)) for v in per_round]
    lo = [float(np.min(v)) for v in per_round]
    hi = [float(np.max(v)) for v in per_round]

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.fill_between(rounds, lo, hi, color=C_BLUE, alpha=0.16, zorder=2,
                    label="Client spread (min–max)")
    ax.plot(rounds, med, color=C_BLUE, linewidth=2.0, marker="o", markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1.2,
            label="Median client loss", zorder=3)

    ax.annotate(f" spread {hi[0]-lo[0]:.4f}", (rounds[0], hi[0]), fontsize=8.5,
                color=INK_SOFT, va="bottom")
    ax.annotate(f" spread {hi[-1]-lo[-1]:.4f}", (rounds[-1], hi[-1]), fontsize=8.5,
                color=INK_SOFT, va="bottom", ha="right")

    _style(ax)
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Local training loss")
    ax.set_xticks(rounds)
    ax.set_ylim(0, max(hi) * 1.15)
    ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=INK)
    ax.set_title("Client loss convergence — spread narrows as clients align",
                 color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="full_run_log.json")
    p.add_argument("--data-root", default="hospital_partitions")
    p.add_argument("--out-dir", default=".")
    args = p.parse_args()

    with open(args.log) as f:
        data = json.load(f)
    history = data["history"]
    best = data["best_metrics"]
    cm = best["confusion_matrix"]
    best_round = max(history, key=lambda r: r["roc_auc"])["round"]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[Plots] {len(history)} rounds, best round {best_round}")

    fig_training_progress(history, cm, out / "fig1_training_progress.png")
    fig_client_distribution(args.data_root, out / "fig2_client_distribution.png")
    fig_confusion_matrix(cm, best_round, out / "fig3_confusion_matrix.png")
    fig_client_convergence(history, out / "fig4_client_convergence.png")
    print("[Plots] done.")


if __name__ == "__main__":
    main()
