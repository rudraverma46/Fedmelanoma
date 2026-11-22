"""Best-round (final_best) results for all Novelty tuning batches -> Novelty_best_round_results.xlsx + Novelty_best_round_report.pdf.
Run: python build_best_round.py "<novelty_hpo folder>" """
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
SRC = [("Tuning (stage 2, 9 at once)", r"\results\final", ["H00", "H09", "H10"], "H00"),
       ("One-factor (12 at once)", r"\followup\results\followup", ["F01", "F02", "F03", "F04"], "H00"),
       ("Combined (6 at once)", r"\combo\results\combo", ["C00", "C01"], "C00")]
DESC = {"H00": "default", "H09": "tuned trial H09", "H10": "tuned trial H10", "F01": "default + selection gamma 2.0",
        "F02": "default + aggregation gamma 0.5", "F03": "default + lambda 0.02", "F04": "default + MAD k 2.0",
        "C00": "default (control, same batch as C01)", "C01": "default + aggregation gamma 0.5 + MAD k 2.0"}
K = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("recall", "Recall"), ("specificity", "Specificity"), ("precision", "Precision"),
     ("npv", "NPV"), ("f1", "F1"), ("balanced_accuracy", "Balanced acc."), ("mcc", "MCC"), ("brier", "Brier"),
     ("specificity_at_recall90", "Spec @ recall>=0.90"), ("fairness_gap_6", "Fairness gap 6 types"), ("fairness_gap_3", "Fairness gap 3 groups")]
KC = [k for k, _ in K]
SEEDS = [42, 43, 44]

runs, pairs = [], []
for batch, p, confs, dflt in SRC:
    m = pd.read_csv(D + p + r"\metrics_per_run.csv")
    m = m[(m.checkpoint == "final_best") & (m.config.isin(confs))].copy()
    m.insert(0, "batch", batch); m.insert(2, "description", m.config.map(DESC)); runs.append(m)
    q = pd.read_csv(D + p + r"\paired_vs_default.csv")
    q = q[(q.checkpoint == "final_best") & (q.config.isin(confs)) & (q.config != dflt)].copy()
    q.insert(0, "batch", batch); q.insert(2, "compared_with", dflt); pairs.append(q)
R = pd.concat(runs, ignore_index=True)
P = pd.concat(pairs, ignore_index=True)
R = R[["batch", "config", "description", "seed", "best_round_by_eval_auc"] + KC + [c for c in R.columns if c not in
      ["batch", "config", "description", "seed", "best_round_by_eval_auc", "checkpoint"] + KC]]
S = R.groupby(["batch", "config", "description"], sort=False).agg(
    best_rounds=("best_round_by_eval_auc", lambda s: "/".join(str(int(x)) for x in s)),
    **{f"{k}_mean": (k, "mean") for k in KC}, **{f"{k}_sd": (k, lambda s: s.std(ddof=1)) for k in KC}).reset_index()
ORD = ["H00", "H09", "H10", "F01", "F02", "F03", "F04", "C00", "C01"]
S = S.set_index("config").loc[ORD].reset_index()[S.columns]
R = R.assign(_o=R.config.map(ORD.index)).sort_values(["_o", "seed"]).drop(columns="_o").reset_index(drop=True)
pc = ["batch", "config", "compared_with", "d_roc_auc_mean", "d_auc_seedavg_boot_lo", "d_auc_seedavg_boot_hi"] + \
     [f"delong_p_s{s}" for s in SEEDS] + [f"mcnemar_mal_s{s}" for s in SEEDS] + \
     [c for k in KC if k != "roc_auc" for c in (f"d_{k}_mean", f"d_{k}_seeds_better")]
PS = P[[c for c in pc if c in P.columns]]

xl = D + r"\Novelty_best_round_results.xlsx"
with pd.ExcelWriter(xl, engine="openpyxl") as w:
    pd.DataFrame({"Novelty model, best-round results (all tuning batches)": [
        "Best round = per run, the round with the highest ROC-AUC on the evaluation set, which is the test set. Optimistic; conclusions use final_round.",
        "Data: Dirichlet-GMM, full training data, 30 rounds, pooled test set 2,393 images (417 malignant), threshold 0.5, seeds 42/43/44.",
        "Default = H00 for the tuning and one-factor batches; C00 for the combined batch (same batch as C01).",
        "Batches ran with different numbers of concurrent runs (9 / 12 / 6), which shifts the operating point; compare only within a batch.",
        "Fairness gap = max - min sensitivity over Fitzpatrick types (6) or groups (3); lower is better. Brier: lower is better.",
        "Sheets: summary (mean/SD per config), per_run (all 27 distinct runs, every metric), paired_vs_default (same-seed differences, CI, DeLong, McNemar).",
        "AUC CI = patient-cluster bootstrap (2000 resamples). seeds_better = number of seeds (of 3) where the config beats its default."]}).to_excel(w, sheet_name="README", index=False)
    S.round(4).to_excel(w, sheet_name="summary", index=False)
    R.round(4).to_excel(w, sheet_name="per_run", index=False)
    PS.round(4).to_excel(w, sheet_name="paired_vs_default", index=False)
    for ws in w.book.worksheets:
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = min(60 if ws.title == "README" else 22, max(10, max(len(str(c.value or "")) for c in col) + 2))
        ws.freeze_panes = "A2"

# figure
order = ["H00", "H09", "H10", "F01", "F02", "F03", "F04", "C00", "C01"]
lab = [f"{c}" if c != "H00" else "H00" for c in order]
COL = {"H00": "#333333", "H09": "#2e8b57", "H10": "#7fbf7f", "F01": "#c0392b", "F02": "#2a6fdb", "F03": "#8e8e8e", "F04": "#e0812b",
       "C00": "#555555", "C01": "#6a3fb5"}
mets = [("roc_auc", "ROC-AUC"), ("recall", "Recall @0.5"), ("specificity", "Specificity @0.5"), ("fairness_gap_3", "Fairness gap (3 groups)")]
RR = R.drop_duplicates(["config", "seed"])
fig, axs = plt.subplots(1, 4, figsize=(13, 3.2))
for ax, (m, t) in zip(axs, mets):
    for i, c in enumerate(order):
        v = RR[RR.config == c][m].to_numpy()
        ax.scatter([i] * len(v), v, color=COL[c], s=16, zorder=3); ax.hlines(v.mean(), i - 0.3, i + 0.3, color=COL[c], lw=2)
    for x in (2.5, 6.5): ax.axvline(x, color="#bbbbbb", lw=0.8, ls="--")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, fontsize=6.5); ax.set_title(t, fontsize=9); ax.tick_params(labelsize=7)
fig.suptitle("Best round (test-selected), dots = seeds, bar = mean. Dashed lines separate batches: tuning | one-factor (vs H00) | combined.", fontsize=8.5)
fig.tight_layout(); f1 = D + r"\fig_best_round.png"; fig.savefig(f1, dpi=170); plt.close(fig)

ss = getSampleStyleSheet()
B = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12)
SM = ParagraphStyle("s", parent=B, fontSize=7.5, leading=9.5)
def tbl(rows, widths=None, fs=7, hl=None):
    t = Table(rows, colWidths=widths, repeatRows=1)
    st = [("FONT", (0, 0), (-1, -1), "Helvetica", fs), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", fs),
          ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf5")), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b0b0b0")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (2, 1), (-1, -1), "CENTER")]
    for r in (hl or []): st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f2f2f2")))
    t.setStyle(TableStyle(st)); return t
sg = lambda v, n=3: "-" if pd.isna(v) else f"{v:+.{n}f}"

el = [Paragraph("Novelty model: results at the best round, all tuning batches", ss["Title"]),
      Paragraph("Best round = for each run, the round with the highest ROC-AUC on the evaluation set, which is the test set itself. This makes "
                "the numbers somewhat optimistic; the conclusions in the earlier reports use the final round (30). Dirichlet-GMM data, 30 rounds, "
                "pooled test set (2,393 images, 417 malignant), threshold 0.5, seeds 42/43/44. The three batches ran with different numbers of "
                "concurrent runs (9 / 12 / 6), which shifts the operating point, so compare configurations only within a batch.", B), Spacer(1, 6)]
mk = ["roc_auc", "pr_auc", "recall", "specificity", "precision", "f1", "balanced_accuracy", "mcc", "specificity_at_recall90", "fairness_gap_6", "fairness_gap_3"]
hdr = ["Batch", "Config", "Best rounds", "ROC-AUC", "PR-AUC", "Recall", "Spec.", "Prec.", "F1", "Bal. acc.", "MCC", "Spec@rec90", "Gap 6", "Gap 3"]
rows = [hdr]
for _, r in S.iterrows():
    rows.append([r.batch.split(" (")[0], r.config, r.best_rounds] + [f"{r[f'{k}_mean']:.3f}" for k in mk])
el += [tbl(rows, [3.0 * cm, 1.3 * cm, 2.0 * cm] + [1.75 * cm] * 11, hl=[1, 8])]
el += [Spacer(1, 4), Paragraph("Means over 3 seeds. Shaded rows are the default of each batch (H00 is shared by the tuning and one-factor batches). "
                               "Fairness gaps: lower is better. Configurations: " + "; ".join(f"{c} = {DESC[c]}" for c in order) + ".", SM)]
el += [Spacer(1, 6), Paragraph("Key points", ss["Heading3"]), Paragraph(
    "(1) No configuration changes ROC-AUC: every paired 95% CI includes 0. "
    "(2) The combined setting C01 (aggregation gamma 0.5 + MAD k 2.0) is below its same-batch default C00 at the best round: recall -0.012 and "
    "3-group gap +0.013 at all 3 seeds, AUC -0.003 (DeLong p 0.045 at seed 43 only). "
    "(3) H09 shows the most consistent operating-point gain vs H00 (recall +0.024, 3-group gap -0.027, at 3/3 seeds), but H00 ran in a "
    "different batch, and the combined test showed batch load alone can move recall by about 0.02.", B)]
el += [PageBreak(), Image(f1, 26 * cm, 6.4 * cm), Spacer(1, 6), Paragraph("Paired differences vs the default of the same batch (same seed), best round", ss["Heading2"])]
rows = [["Config", "vs", "AUC diff [95% CI]", "DeLong p s42/s43/s44", "Recall diff (better)", "Spec. diff (better)", "MCC diff (better)", "Gap 3 diff (better)"]]
for _, r in P.iterrows():
    f = lambda k: f"{sg(r[f'd_{k}_mean'])} ({int(r[f'd_{k}_seeds_better'])}/3)"
    rows.append([r.config, r.compared_with, f"{sg(r.d_roc_auc_mean, 4)} [{sg(r.d_auc_seedavg_boot_lo, 4)}, {sg(r.d_auc_seedavg_boot_hi, 4)}]",
                 " / ".join(f"{r[f'delong_p_s{s}']:.3f}" for s in SEEDS), f("recall"), f("specificity"), f("mcc"), f("fairness_gap_3")])
el += [tbl(rows, [1.4 * cm, 1.2 * cm, 5.0 * cm, 3.8 * cm, 3.4 * cm, 3.4 * cm, 3.4 * cm, 3.4 * cm])]
el += [Spacer(1, 4), Paragraph("For the fairness gap, 'better' means smaller. Seed counts and per-seed tests are descriptive (3 seeds, no correction "
                               "for multiple metrics). The per-run values for all 27 distinct runs are in Novelty_best_round_results.xlsx (sheet per_run).", SM)]
el += [PageBreak(), Paragraph("Per-run values at the best round", ss["Heading2"])]
rows = [["Config", "Seed", "Best round"] + hdr[3:]]
for _, r in RR.iterrows():
    rows.append([r.config, int(r.seed), int(r.best_round_by_eval_auc)] + [f"{r[k]:.3f}" for k in mk])
el += [tbl(rows, [1.5 * cm, 1.3 * cm, 1.8 * cm] + [2.0 * cm] * 11, fs=6.8)]
SimpleDocTemplate(D + r"\Novelty_best_round_report.pdf", pagesize=landscape(A4), leftMargin=1.2 * cm, rightMargin=1.2 * cm,
                  topMargin=1.1 * cm, bottomMargin=1.1 * cm, title="Novelty best-round results").build(el)
print("ok", len(R), "runs,", len(P), "pairs")


