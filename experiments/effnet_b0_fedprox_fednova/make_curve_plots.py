"""
make_curve_plots.py — Threshold-dependent figures for the selected model.

Reads test_predictions.csv (written by evaluate_checkpoint.py) and produces:

  fig5_roc_curve.png          — ROC curve with the operating point marked
  fig6_pr_curve.png           — precision-recall curve (the honest view under
                                 5.6% prevalence, where ROC looks flattering)
  fig7_threshold_analysis.png — sensitivity / specificity / precision as the
                                 decision threshold moves, for picking an
                                 operating point deliberately
  fig8_per_hospital.png       — per-client ROC-AUC, i.e. whether the single
                                 global model serves every hospital equally

Usage:
    python make_curve_plots.py --pred test_predictions.csv --out-dir figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    roc_auc_score,
)

C_BLUE = "#2a78d6"
C_ORANGE = "#eb6834"
C_AQUA = "#1baf7a"
C_YELLOW = "#eda100"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e3e2de"

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


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def short(name: str) -> str:
    return name.split("_")[0]


def fig_roc(y, p, out):
    fpr, tpr, thr = roc_curve(y, p)
    a = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.plot([0, 1], [0, 1], color=INK_SOFT, linestyle="--", linewidth=1.0,
            alpha=0.6, zorder=2)
    ax.annotate("random classifier", (0.55, 0.50), color=INK_SOFT, fontsize=8.5,
                rotation=33, va="bottom", ha="center")
    ax.plot(fpr, tpr, color=C_BLUE, linewidth=2.0, zorder=3)
    ax.fill_between(fpr, tpr, color=C_BLUE, alpha=0.10, zorder=1)

    # Operating point actually used (argmax decision = threshold 0.5)
    i = int(np.argmin(np.abs(thr - 0.5)))
    ax.plot(fpr[i], tpr[i], marker="o", markersize=9, color=C_ORANGE,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
    ax.annotate(f"  operating point (t = 0.50)\n  sensitivity {tpr[i]:.3f}, "
                f"FPR {fpr[i]:.3f}",
                (fpr[i], tpr[i]), color=INK, fontsize=9, va="top", ha="left")

    _style(ax)
    ax.set_xlabel("False positive rate (1 − specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"ROC curve — AUC {a:.4f}", color=INK, fontsize=12.5,
                 fontweight="bold", loc="left", pad=12)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


def fig_pr(y, p, out):
    prec, rec, thr = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    prevalence = float(np.mean(y))

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.axhline(prevalence, color=INK_SOFT, linestyle="--", linewidth=1.0, alpha=0.7, zorder=2)
    ax.annotate(f"random classifier (= prevalence, {prevalence:.3f})",
                (0.02, prevalence + 0.012), color=INK_SOFT, fontsize=8.5, va="bottom")
    ax.plot(rec, prec, color=C_ORANGE, linewidth=2.0, zorder=3)
    ax.fill_between(rec, prec, color=C_ORANGE, alpha=0.10, zorder=1)

    i = int(np.argmin(np.abs(thr - 0.5)))
    ax.plot(rec[i], prec[i], marker="o", markersize=9, color=C_BLUE,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
    ax.annotate(f"  operating point (t = 0.50)\n  recall {rec[i]:.3f}, "
                f"precision {prec[i]:.3f}",
                (rec[i], prec[i]), color=INK, fontsize=9, va="bottom", ha="left")

    _style(ax)
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1.02)
    ax.set_title(f"Precision–recall curve — average precision {ap:.4f}",
                 color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=12)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


def fig_threshold(y, p, out):
    ts = np.linspace(0.02, 0.98, 97)
    sens, spec, prec, f1 = [], [], [], []
    y = np.asarray(y); p = np.asarray(p)
    for t in ts:
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        se = tp / (tp + fn) if tp + fn else 0.0
        sp = tn / (tn + fp) if tn + fp else 0.0
        pr = tp / (tp + fp) if tp + fp else 0.0
        sens.append(se); spec.append(sp); prec.append(pr)
        f1.append(2 * pr * se / (pr + se) if pr + se else 0.0)

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for label, vals, color in [("Sensitivity", sens, C_AQUA),
                               ("Specificity", spec, C_YELLOW),
                               ("Precision", prec, C_BLUE),
                               ("F1", f1, C_ORANGE)]:
        ax.plot(ts, vals, color=color, linewidth=2.0, label=label, zorder=3)

    # Threshold markers, labelled inside the axes so nothing is clipped away
    # by the tight bounding box on save.
    ax.axvline(0.5, color=INK_SOFT, linestyle=":", linewidth=1.2, zorder=2)
    ax.annotate("default\nt = 0.50", (0.492, 0.055), color=INK_SOFT,
                fontsize=8.5, ha="right", va="bottom", linespacing=1.35)
    best = int(np.argmax(f1))
    ax.axvline(ts[best], color=C_ORANGE, linestyle=":", linewidth=1.4, zorder=2)
    ax.annotate(f"best F1 {f1[best]:.3f}\nat t = {ts[best]:.2f}",
                (ts[best] + 0.012, 0.055), color=C_ORANGE, fontsize=8.5,
                ha="left", va="bottom", linespacing=1.35)

    _style(ax)
    ax.set_xlabel("Decision threshold on P(malignant)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02); ax.set_xlim(0, 1)
    ax.legend(frameon=False, ncol=4, fontsize=9, labelcolor=INK,
              loc="lower left", bbox_to_anchor=(0, 1.06), handlelength=1.6)
    ax.set_title("Operating-point trade-off as the decision threshold moves",
                 color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=40)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


def fig_per_hospital(df, out):
    rows = []
    for h, g in df.groupby("hospital"):
        if g["label"].nunique() < 2:      # AUC undefined with one class present
            continue
        rows.append((short(h), roc_auc_score(g["label"], g["prob_malignant"]),
                     len(g), int(g["label"].sum())))
    rows.sort(key=lambda r: r[1])
    names = [r[0] for r in rows]; aucs = [r[1] for r in rows]
    counts = [r[2] for r in rows]; mals = [r[3] for r in rows]
    overall = roc_auc_score(df["label"], df["prob_malignant"])

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    y = np.arange(len(names))
    ax.barh(y, aucs, color=C_BLUE, height=0.62, zorder=3)
    for i, (a, c, m) in enumerate(zip(aucs, counts, mals)):
        ax.annotate(f"  {a:.3f}   ({m}/{c} malignant)", (a, i), va="center",
                    ha="left", fontsize=8.5, color=INK)
    ax.axvline(overall, color=C_ORANGE, linestyle="--", linewidth=1.4, zorder=4)
    ax.annotate(f"pooled {overall:.3f}", (overall, len(names) - 0.4),
                color=C_ORANGE, fontsize=9, ha="center", va="bottom")

    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("ROC-AUC on that hospital's own test split")
    ax.set_xlim(0, 1.32)
    _style(ax); ax.grid(axis="y", visible=False)
    ax.set_title("Does one global model serve every hospital equally?",
                 color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=12)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="test_predictions.csv")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()

    df = pd.read_csv(args.pred)
    y = df["label"].values
    p = df["prob_malignant"].values
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"[Curves] {len(df)} predictions, {int(y.sum())} malignant "
          f"({y.mean()*100:.2f}% prevalence)")

    fig_roc(y, p, out / "fig5_roc_curve.png")
    fig_pr(y, p, out / "fig6_pr_curve.png")
    fig_threshold(y, p, out / "fig7_threshold_analysis.png")
    fig_per_hospital(df, out / "fig8_per_hospital.png")
    print("[Curves] done.")


if __name__ == "__main__":
    main()
