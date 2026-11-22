"""Short PDF of the combined follow-up (C01 = default + agg gamma 0.5 + MAD k 2.0 vs fresh concurrent default C00).
Run: python build_combo_pdf.py "<novelty_hpo folder>" """
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
R = D + r"\combo\results\combo"
M = pd.read_csv(R + r"\metrics_per_run.csv")
F = pd.read_csv(D + r"\followup\results\followup\metrics_per_run.csv")
PV = pd.read_csv(R + r"\paired_vs_default.csv")
fr = pd.concat([M, F[F.config.isin(["F02", "F04", "H00"])]])
fr = fr[fr.checkpoint == "final_round"]
MAIN = ["C00", "C01"]; CTX = ["H00", "F02", "F04"]; ALL = MAIN + CTX
COL = {"C00": "#333333", "C01": "#2a6fdb", "H00": "#9a9a9a", "F02": "#8fb3ee", "F04": "#f0b27a"}
SEEDS = [42, 43, 44]

mets = [("roc_auc", "ROC-AUC"), ("recall", "Recall @0.5"), ("specificity", "Specificity @0.5"), ("balanced_accuracy", "Balanced accuracy"),
        ("fairness_gap_3", "Fairness gap (3 groups)")]
fig, axs = plt.subplots(1, len(mets), figsize=(13, 3.1))
for ax, (m, lab) in zip(axs, mets):
    for i, c in enumerate(ALL):
        v = fr[fr.config == c].sort_values("seed")[m].to_numpy()
        ax.scatter([i] * len(v), v, color=COL[c], s=20, zorder=3)
        ax.hlines(v.mean(), i - 0.3, i + 0.3, color=COL[c], lw=2)
    ax.axvline(1.5, color="#bbbbbb", lw=0.8, ls="--")
    ax.set_xticks(range(len(ALL))); ax.set_xticklabels(ALL, fontsize=7); ax.set_title(lab, fontsize=9); ax.tick_params(labelsize=7)
fig.suptitle("Real test set, final round: dots = seeds 42/43/44, bar = mean. Left of dashed line: this run (6 concurrent). "
             "Right: earlier runs, for context only (H00 ran 9 concurrent, F02/F04 12 concurrent).", fontsize=8.5)
fig.tight_layout(); f1 = R + r"\fig_combo.png"; fig.savefig(f1, dpi=170); plt.close(fig)

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
mean = lambda c, k: fr[fr.config == c][k].mean()

el = [Paragraph("Novelty model: combined follow-up (aggregation gamma 0.5 + MAD k 2.0)", ss["Title"]),
      Paragraph("Question: in the one-factor follow-up, aggregation gamma 0.5 (F02) and MAD k 2.0 (F04) each raised recall and shrank the "
                "fairness gap versus the default H00. Do they add up when combined, and does the gain survive a control run under the same machine "
                "load? C01 = default + both changes; C00 = an exact copy of the default, run at the same time. Seeds 42/43/44, full Dirichlet-GMM "
                "training data, 30 rounds, pooled test set (2,393 images, 417 malignant), threshold 0.5. All 6 runs at once. DGX, 22 Nov 2026 "
                "04:07 - 06:13 UTC.", B), Spacer(1, 5),
      Paragraph("Answer", ss["Heading2"]),
      Paragraph(f"<b>No, the recall and fairness gain is not confirmed.</b> Against the concurrent control, C01 has the same mean recall "
                f"({mean('C01','recall'):.3f} vs {mean('C00','recall'):.3f}; better at only 1 of 3 seeds) and no smaller fairness gap "
                f"(3 groups {mean('C01','fairness_gap_3'):.3f} vs {mean('C00','fairness_gap_3'):.3f}). It is slightly higher on specificity "
                f"(+0.007), precision, F1 and MCC, and ROC-AUC +0.003, each at 2 of 3 seeds; the AUC CI includes 0. So the two effects do not add up.", B),
      Spacer(1, 3),
      Paragraph(f"<b>Why the earlier gain disappears:</b> the fresh default C00 itself reached recall {mean('C00','recall'):.3f}, close to "
                f"F02 ({mean('F02','recall'):.3f}) and F04 ({mean('F04','recall'):.3f}) and well above the earlier default H00 "
                f"({mean('H00','recall'):.3f}), which has identical settings but ran 9 at a time instead of 6. The machine load (and the timing "
                "it causes in the straggler handling) moves the operating point by about as much as the tuned settings did. The F02/F04 vs H00 "
                "differences from the previous report are therefore not reliable evidence for those two settings.", B)]

el += [PageBreak(), Image(f1, 26 * cm, 6.2 * cm)]
keys = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("recall", "Recall"), ("specificity", "Specificity"), ("precision", "Precision"),
        ("f1", "F1"), ("balanced_accuracy", "Balanced acc."), ("mcc", "MCC"), ("brier", "Brier (lower better)"),
        ("specificity_at_recall90", "Spec. @ recall>=0.90"), ("fairness_gap_6", "Fairness gap, 6 types"), ("fairness_gap_3", "Fairness gap, 3 groups")]
lab = {"C00": "C00 control (6 conc.)", "C01": "C01 combined (6 conc.)", "H00": "H00 earlier (9 conc.)", "F02": "F02 earlier (12 conc.)", "F04": "F04 earlier (12 conc.)"}
rows = [["Metric (final round), mean +- SD"] + [lab[c] for c in ALL]]
for k, l in keys:
    rows.append([l] + [f"{fr[fr.config == c][k].mean():.3f} +- {fr[fr.config == c][k].std(ddof=1):.3f}" for c in ALL])
el += [tbl(rows, [4.6 * cm] + [3.6 * cm] * 5)]
el += [PageBreak(), Paragraph("Paired differences, C01 - C00 (same seed, same run batch), final round", ss["Heading2"])]
p = PV[(PV.config == "C01") & (PV.checkpoint == "final_round")].iloc[0]
rows = [["Metric", "C01 - C00 mean", "Seeds better (of 3)"]]
for k, l in keys:
    rows.append([l, sg(p[f"d_{k}_mean"]), str(int(p[f"d_{k}_seeds_better"]))])
el += [tbl(rows, [5 * cm, 4 * cm, 4 * cm])]
rows = [["AUC diff [95% CI], patient bootstrap", "DeLong p (s42 / s43 / s44)", "McNemar malignant: C01-only / C00-only correct, p"],
        [f"{sg(p.d_roc_auc_mean)} [{sg(p.d_auc_seedavg_boot_lo)}, {sg(p.d_auc_seedavg_boot_hi)}]",
         " / ".join(f"{p[f'delong_p_s{s}']:.3f}" for s in SEEDS), " ; ".join(f"s{s} {p[f'mcnemar_mal_s{s}']}" for s in SEEDS)]]
el += [Spacer(1, 6), tbl(rows, [6 * cm, 5 * cm, 11 * cm])]
el += [Spacer(1, 8), Paragraph("Caveats", ss["Heading3"]), Paragraph(
    "(1) Three seeds; seed counts and per-seed tests are descriptive, with no correction for many metrics. "
    "(2) The H00/F02/F04 columns are context only: they ran in other batches with different concurrency, so they are not paired with C00/C01. "
    "(3) The fairness gap rests on few malignant images outside types I-III (6 in IV-VI), so it moves in coarse steps. "
    "(4) In results/combo/summary.md the labels 'H00' mean C00 (the report script was reused with C00 as the comparison arm). "
    "(5) final_best (test-selected checkpoint) is in summary.md and not used for conclusions.", B), Spacer(1, 4),
    Paragraph("<b>Practical reading:</b> keep the default configuration. None of the tuned or follow-up settings gives a ROC-AUC gain, and the "
              "operating-point gains at 0.5 are within what machine load alone produces. Selection gamma 2.0 (F01) remains the only setting "
              "with a measurable effect, and it is negative. Any claim about operating points should compare runs from the same batch.", B), Spacer(1, 4),
    Paragraph("Files: results/combo/{summary.md, metrics_per_run.csv, summary_by_config.csv, paired_vs_default.csv, fairness_by_skin.csv}; "
              "DGX /workspace/novelty_hpo (raw_combo/C0*/seed_*). Archive combo_results.tgz md5 722f6e64c7532f060d0db29fcffe95ea.", SM)]
SimpleDocTemplate(D + r"\combo\Novelty_combo_report.pdf", pagesize=landscape(A4), leftMargin=1.4 * cm, rightMargin=1.4 * cm,
                  topMargin=1.2 * cm, bottomMargin=1.2 * cm, title="Novelty combined follow-up").build(el)
print("ok")
