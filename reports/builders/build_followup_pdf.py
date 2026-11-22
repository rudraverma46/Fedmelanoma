"""Short PDF of the one-factor follow-up (F01-F04 vs default H00). Run: python build_followup_pdf.py "<this folder>" """
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

D = sys.argv[1]
R = D + r"\results\followup"
M = pd.read_csv(R + r"\metrics_per_run.csv")
PV = pd.read_csv(R + r"\paired_vs_default.csv")
CONF = ["F01", "F02", "F03", "F04", "H00"]
LAB = {"F01": "F01 selection gamma 2.0", "F02": "F02 aggregation gamma 0.5", "F03": "F03 lambda 0.02", "F04": "F04 MAD k 2.0", "H00": "H00 default"}
COL = {"F01": "#c0392b", "F02": "#2a6fdb", "F03": "#8e8e8e", "F04": "#e0812b", "H00": "#333333"}
SEEDS = [42, 43, 44]
fr = M[M.checkpoint == "final_round"]

mets = [("roc_auc", "ROC-AUC"), ("recall", "Recall @0.5"), ("specificity", "Specificity @0.5"), ("balanced_accuracy", "Balanced accuracy"),
        ("fairness_gap_3", "Fairness gap (3 groups)")]
fig, axs = plt.subplots(1, len(mets), figsize=(13, 3.1))
for ax, (m, lab) in zip(axs, mets):
    for i, c in enumerate(CONF):
        v = fr[fr.config == c].sort_values("seed")[m].to_numpy()
        ax.scatter([i] * len(v), v, color=COL[c], s=20, zorder=3)
        ax.hlines(v.mean(), i - 0.3, i + 0.3, color=COL[c], lw=2)
    ax.set_xticks(range(len(CONF))); ax.set_xticklabels(CONF, fontsize=7); ax.set_title(lab, fontsize=9); ax.tick_params(labelsize=7)
fig.suptitle("Real test set, final round: dots = seeds 42/43/44, bar = mean. Each F variant = default with ONE change.", fontsize=9)
fig.tight_layout(); f1 = R + r"\fig_followup.png"; fig.savefig(f1, dpi=170); plt.close(fig)

ss = getSampleStyleSheet()
B = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12)
SM = ParagraphStyle("s", parent=B, fontSize=7.5, leading=9.5)
def tbl(rows, widths=None, fs=7.3, hl=None):
    t = Table(rows, colWidths=widths, repeatRows=1)
    st = [("FONT", (0, 0), (-1, -1), "Helvetica", fs), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", fs),
          ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf5")), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b0b0b0")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (1, 1), (-1, -1), "CENTER")]
    for r in (hl or []): st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#fff6d6")))
    t.setStyle(TableStyle(st)); return t
def sg(v, n=4): return "-" if pd.isna(v) else f"{v:+.{n}f}"

el = [Paragraph("Novelty model: one-factor follow-up", ss["Title"]),
      Paragraph("Question: H10 and H09 (the two best tuned configurations) share four changes from the default. Which of them, applied ALONE to the "
                "default, reproduces the higher recall and smaller fairness gap? Each variant was trained on the full Dirichlet-GMM training data, "
                "30 rounds, seeds 42/43/44, and evaluated on the pooled test set (2,393 images, 417 malignant), threshold 0.5. "
                "Comparison arm: the stage-2 default H00 runs at the same seeds. DGX run 21 Nov 2026 19:40 - 23:53 UTC.", B), Spacer(1, 5),
      Paragraph("Answer", ss["Heading2"]),
      Paragraph("<b>Aggregation gamma 0.5 (F02) and MAD k 2.0 (F04) each reproduce the effect on their own.</b> Both raise recall at all 3 seeds "
                "(+0.031 and +0.026), raise balanced accuracy at all 3 seeds, and shrink both fairness gaps at all 3 seeds (6 types -0.046, 3 groups "
                "-0.038 / -0.033), at a small specificity cost (-0.007 / -0.005). F04 is the cleaner of the two: F1, MCC and specificity at recall>=0.90 "
                "are also higher at all 3 seeds. ROC-AUC does not change for either (CIs include 0), so again this is a better operating point at 0.5, "
                "not better ranking.", B), Spacer(1, 3),
      Paragraph("<b>Lambda 0.02 (F03) does little:</b> mixed across seeds, apart from specificity at recall>=0.90 (+0.013, 3/3).", B), Spacer(1, 3),
      Paragraph("<b>Selection gamma 2.0 (F01) hurts:</b> ROC-AUC -0.005, patient-bootstrap 95% CI [-0.0087, -0.0018] excludes 0, DeLong significant "
                "at 2 of 3 seeds; no recall or fairness benefit. It is the only one of the four with a measurable negative effect, and it is shared by "
                "H10 and H09, which suggests that those configurations got their gain despite it, from aggregation gamma 0.5 and MAD k 2.0.", B)]

el += [PageBreak(), Image(f1, 26 * cm, 6.2 * cm)]
keys = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("recall", "Recall"), ("specificity", "Specificity"), ("precision", "Precision"),
        ("f1", "F1"), ("balanced_accuracy", "Balanced acc."), ("mcc", "MCC"), ("brier", "Brier (lower better)"),
        ("specificity_at_recall90", "Spec. @ recall>=0.90"), ("fairness_gap_6", "Fairness gap, 6 types"), ("fairness_gap_3", "Fairness gap, 3 groups"),
        ("specificity_gap_6", "Specificity gap, 6 types")]
rows = [["Metric (final round)"] + [f"{c} mean +- SD" for c in CONF]]
for k, lab in keys:
    rows.append([lab] + [f"{fr[fr.config == c][k].mean():.3f} +- {fr[fr.config == c][k].std(ddof=1):.3f}" for c in CONF])
el += [tbl(rows, [4.2 * cm] + [3.4 * cm] * 5)]
el += [PageBreak(), Paragraph("Paired differences vs the default (same seed), final round", ss["Heading2"])]
rows = [["Metric"] + [f"{c} - H00 (seeds better)" for c in CONF[:4]]]
for k, lab in keys:
    r = [lab]
    for c in CONF[:4]:
        p = PV[(PV.config == c) & (PV.checkpoint == "final_round")].iloc[0]
        r.append(f"{sg(p[f'd_{k}_mean'])}  ({int(p[f'd_{k}_seeds_better'])}/3)")
    rows.append(r)
el += [tbl(rows, [4.2 * cm] + [4.3 * cm] * 4, hl=[3, 7, 11, 12])]
rows = [["Variant", "AUC diff [95% CI], patient bootstrap", "DeLong p (s42 / s43 / s44)", "McNemar malignant: variant-only / default-only correct, p"]]
for c in CONF[:4]:
    p = PV[(PV.config == c) & (PV.checkpoint == "final_round")].iloc[0]
    rows.append([LAB[c], f"{sg(p.d_roc_auc_mean)} [{sg(p.d_auc_seedavg_boot_lo)}, {sg(p.d_auc_seedavg_boot_hi)}]",
                 " / ".join(f"{p[f'delong_p_s{s}']:.3f}" for s in SEEDS), " ; ".join(f"s{s} {p[f'mcnemar_mal_s{s}']}" for s in SEEDS)])
el += [Spacer(1, 6), tbl(rows, [4.4 * cm, 5.2 * cm, 4.4 * cm, 11.2 * cm])]
el += [Spacer(1, 8), Paragraph("Caveats", ss["Heading3"]), Paragraph(
    "(1) Three seeds; 'n/3 better' and the per-seed tests are descriptive, with no correction for the many metrics. "
    "(2) Runs are not bit-reproducible: two default runs with identical settings differed by 0.006 AUC. The F runs shared the machine 12 at a time, the "
    "H00 comparison runs 9 at a time; since the straggler handling depends on timing, this could matter most for F04 (MAD k sets the waiting window). "
    "(3) The fairness gap rests on few malignant images outside types I-III (6 in IV-VI), so it mainly reflects type I vs II-III sensitivity. "
    "(4) final_best (test-selected checkpoint) is in results/followup/summary.md and is not used for conclusions.", B), Spacer(1, 4),
    Paragraph("<b>Suggested next step (not started):</b> default + aggregation gamma 0.5 + MAD k 2.0 together, 3 seeds, ideally with a fresh H00 "
              "control run at the same time (same machine load), to test whether the two effects add up.", B), Spacer(1, 4),
    Paragraph("Files: results/followup/{summary.md, metrics_per_run.csv, summary_by_config.csv, paired_vs_default.csv, fairness_by_skin.csv}; "
              "raw_followup/F0*/seed_*/ (metrics, predictions, logs); DGX /workspace/novelty_hpo. Archive followup_results.tgz md5 541e741565217faa7d8bd2bca542599b.", SM)]
SimpleDocTemplate(D + r"\Novelty_followup_report.pdf", pagesize=landscape(A4), leftMargin=1.4 * cm, rightMargin=1.4 * cm,
                  topMargin=1.2 * cm, bottomMargin=1.2 * cm, title="Novelty one-factor follow-up").build(el)
print("ok")
