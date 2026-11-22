"""Build the PDF report of the Novelty hyperparameter tuning (v3) from the downloaded results."""
import json, sys
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
R = D + r"\results"
M = pd.read_csv(R + r"\final\metrics_per_run.csv")
PV = pd.read_csv(R + r"\final\paired_vs_default.csv")
S1 = pd.read_csv(R + r"\stage1_table.csv")
sel = json.load(open(D + r"\logs\selection.json"))
trials = json.load(open(D + r"\code\ablation\hpo_trials.json"))
CONF = ["H10", "H09", "H00"]
SEEDS = [42, 43, 44]
COL = {"H10": "#2a6fdb", "H09": "#e0812b", "H00": "#555555"}

# figures
fig, ax = plt.subplots(figsize=(8, 3.2))
s1 = S1.sort_values("val_auc_last3_mean", ascending=True)
ax.barh(s1.trial, s1.val_auc_last3_mean, xerr=s1.val_auc_last3_sd, color=["#555555" if t == "H00" else "#9bb7e0" for t in s1.trial], capsize=3)
for t, v in zip(s1.trial, s1.val_auc_last3_mean): ax.text(v + 0.0125, t, f"{v:.4f}", va="center", fontsize=7)
ax.set_xlim(0.92, 0.985); ax.set_xlabel("Validation ROC-AUC, mean of last 3 rounds, mean +- SD over seeds 42-44")
ax.set_title("Stage 1: 12 configurations (H00 = default, grey)", fontsize=10)
fig.tight_layout(); f1 = D + r"\results\final\fig_stage1.png"; fig.savefig(f1, dpi=170); plt.close(fig)

mets = [("roc_auc", "ROC-AUC"), ("recall", "Recall @0.5"), ("specificity", "Specificity @0.5"), ("mcc", "MCC @0.5"), ("fairness_gap_3", "Fairness gap (3 groups)")]
fig, axs = plt.subplots(1, len(mets), figsize=(12, 3.0))
fr = M[(M.checkpoint == "final_round") & M.config.isin(CONF)]
for ax, (m, lab) in zip(axs, mets):
    for i, c in enumerate(CONF):
        v = fr[fr.config == c].sort_values("seed")[m].to_numpy()
        ax.scatter([i] * len(v), v, color=COL[c], s=22, zorder=3)
        ax.hlines(v.mean(), i - 0.25, i + 0.25, color=COL[c], lw=2)
    for s in SEEDS:
        ax.plot(range(3), [fr[(fr.config == c) & (fr.seed == s)][m].iloc[0] for c in CONF], color="#bbbbbb", lw=0.7, zorder=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(CONF); ax.set_title(lab, fontsize=9); ax.tick_params(labelsize=7)
fig.suptitle("Stage 2, real test set, final round: dots = seeds 42/43/44 (grey lines join the same seed), bar = mean", fontsize=9)
fig.tight_layout(); f2 = D + r"\results\final\fig_stage2.png"; fig.savefig(f2, dpi=170); plt.close(fig)

ss = getSampleStyleSheet()
B = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12)
SM = ParagraphStyle("s", parent=B, fontSize=7.5, leading=9.5)
H1, H2 = ss["Title"], ss["Heading2"]
def tbl(rows, widths=None, fs=7.5, hl=None):
    t = Table(rows, colWidths=widths, repeatRows=1)
    st = [("FONT", (0, 0), (-1, -1), "Helvetica", fs), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", fs),
          ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf5")), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b0b0b0")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (1, 1), (-1, -1), "CENTER")]
    for r in (hl or []): st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#fff6d6")))
    t.setStyle(TableStyle(st)); return t
def f(v, n=3): return "-" if pd.isna(v) else f"{v:.{n}f}"
def sg(v, n=4): return "-" if pd.isna(v) else f"{v:+.{n}f}"

el = []
el += [Paragraph("Novelty model: hyperparameter tuning report", H1),
       Paragraph("ResNet-50 Novelty (Boltzmann selection + straggler window + diversity-weighted aggregation), Dirichlet-GMM partition "
                 "(10 hospitals), diagnosis_confirm_type removed. DGX run 21 Nov 2026, 07:57-14:27 UTC. Folder /workspace/novelty_hpo.", B), Spacer(1, 6)]

fr_m = fr.groupby("config")[["roc_auc", "recall", "specificity", "mcc", "fairness_gap_3"]].mean()
el += [Paragraph("Bottom line", H2), Paragraph(
    "<b>Tuning did not find a configuration with a better ROC-AUC than the default.</b> On the real test set (final round, mean of 3 seeds) "
    f"ROC-AUC is H09 {f(fr_m.loc['H09','roc_auc'])}, default H00 {f(fr_m.loc['H00','roc_auc'])}, H10 {f(fr_m.loc['H10','roc_auc'])}; "
    "every patient-bootstrap 95% CI of the AUC difference includes 0, and no per-seed DeLong test is significant.", B), Spacer(1, 3),
    Paragraph("<b>What did change is the operating point at threshold 0.5.</b> Both tuned configurations catch more melanomas: recall "
              f"H10 {f(fr_m.loc['H10','recall'])}, H09 {f(fr_m.loc['H09','recall'])} vs default {f(fr_m.loc['H00','recall'])}. "
              "H09 does this at almost no cost in specificity (-0.003), with higher F1, MCC and balanced accuracy at all 3 seeds and a smaller "
              "3-group fairness gap at all 3 seeds. H10 pays for its recall with clearly lower specificity, precision and MCC, so it is not a better model.", B), Spacer(1, 3),
    Paragraph("<b>Recommendation:</b> if a configuration is adopted, H09 is the candidate, described honestly as 'same AUC, better sensitivity "
              "and fairness at 0.5' rather than 'better model'. The default stays a fully defensible choice. "
              "Caution: the run-to-run noise of identical settings is about 0.006 AUC (see the sanity check), larger than every tuned difference.", B)]

el += [Paragraph("Design", H2), Paragraph(
    "<b>Stage 1 (selection, validation only):</b> 15% of training patients held out (312 patients, 1,459 images, 262 malignant; patient-level, "
    "stratified, seed 42); the real test set is not used. 12 configurations = default H00 + 11 random draws (seed 2026) over 19 hyperparameters, "
    "each trained 12 rounds at seeds 42, 43 and 44 (36 runs). Score = validation ROC-AUC averaged over the last 3 rounds, then over the 3 seeds.", B),
    Paragraph("<b>Stage 2 (confirmation, real test set):</b> the 2 best non-default configurations (H10, H09) and the default H00, each retrained on the full "
              "training data for 30 rounds at seeds 42, 43, 44 (9 runs), then evaluated on the pooled test set (2,393 images, 417 malignant, 935 patients). "
              "Metrics at threshold 0.5. Fairness gap = max - min sensitivity over Fitzpatrick types with at least one malignant image (6 types, and 3 groups I-II / III-IV / V-VI).", B),
    Paragraph("<b>Tests:</b> per-seed paired differences vs H00 (same seed); patient-cluster bootstrap 95% CI (2,000 resamples) of the seed-averaged AUC difference; "
              "per-seed DeLong (AUC) and exact McNemar on malignant images. 'final_round' (the honest checkpoint) is the main result; 'final_best' is the "
              "round with the best test AUC, i.e. chosen on the test set, and is shown only for completeness.", B)]

el += [PageBreak(), Paragraph("The 19 tuned hyperparameters and the stage-2 configurations", H2)]
desc = {"learning_rate": "local learning rate (AdamW)", "weight_decay": "AdamW weight decay", "batch_size": "local batch size",
        "local_epochs": "local epochs per round", "dropout": "classifier-head dropout", "focal_gamma": "focal-loss gamma",
        "clients_per_round": "hospitals selected per round", "tau": "Boltzmann temperature", "gamma": "diversity weight in selection",
        "lambda": "diversity-bonus attenuation", "rho": "reliability EMA memory", "delta": "staleness discount (selector, straggler, aggregation)",
        "agg_gamma": "diversity weight in aggregation", "s_max": "max staleness kept (rounds)", "quorum_frac": "quorum to close a round",
        "mad_k": "MAD multiplier of the waiting window", "w_min": "minimum waiting window (s)", "history_len": "latency history length",
        "fitz_weight": "Fitzpatrick weight in diversity", "modality_weight": "modality weight (= 0.75 - Fitzpatrick)"}
grid = {"learning_rate": "5e-5, 1e-4, 2e-4, 3e-4", "weight_decay": "1e-5, 1e-4, 1e-3", "batch_size": "16, 20, 24", "local_epochs": "3, 4, 5",
        "dropout": "0.2, 0.3, 0.5", "focal_gamma": "1, 2, 3", "clients_per_round": "5, 6, 7", "tau": "0.1, 0.2, 0.35, 0.5", "gamma": "0.5, 1, 1.5, 2",
        "lambda": "0, 0.02, 0.05, 0.1", "rho": "0.7, 0.85, 0.95", "delta": "0.8, 0.9, 0.95", "agg_gamma": "0.5, 1, 1.5, 2", "s_max": "2, 3, 5",
        "quorum_frac": "0.6, 0.7, 0.8", "mad_k": "2.0, 2.5, 3.0", "w_min": "1, 2, 4", "history_len": "5, 10, 20", "fitz_weight": "0.25, 0.40, 0.55",
        "modality_weight": "(derived)"}
rows = [["Hyperparameter", "Meaning", "Values tried", "H00 (default)", "H10", "H09"]]
hl = []
for i, k in enumerate(desc, start=1):
    vals = [trials[c].get(k) for c in ("H00", "H10", "H09")]
    rows.append([k, desc[k], grid[k]] + [str(v) for v in vals])
    if vals[1] != vals[0] or vals[2] != vals[0]: pass
el += [tbl(rows, [3.2 * cm, 5.6 * cm, 3.6 * cm, 2.2 * cm, 1.6 * cm, 1.6 * cm], 7.5),
       Paragraph("Not tuned (fixed on purpose): ResNet-50 architecture and metadata MLP, AdamW, the 10 hospitals, the simulated straggler/failure environment, evaluation threshold.", SM)]

el += [PageBreak(), Paragraph("Stage 1: validation ranking (12 configurations x 3 seeds)", H2), Image(f1, 17 * cm, 6.8 * cm)]
rows = [["Rank", "Config", "mean", "SD", "seed 42", "seed 43", "seed 44", "min"]]
for i, r in enumerate(S1.sort_values("val_auc_last3_mean", ascending=False).itertuples(), start=1):
    rows.append([i, r.trial, f(r.val_auc_last3_mean, 4), f(r.val_auc_last3_sd, 4), f(r.val_auc_last3_s42, 4), f(r.val_auc_last3_s43, 4), f(r.val_auc_last3_s44, 4), f(r.val_auc_last3_min, 4)])
el += [tbl(rows, hl=[1, 2, 5]), Spacer(1, 4), Paragraph(
    "The top 5 lie within 0.004 of each other, the same size as the seed-to-seed SD, so the stage-1 ranking among them is weak. "
    "The clearest pattern is the learning rate: all top-5 configurations use 1e-4 or 2e-4, while the bottom four use the extremes "
    "(H01 3e-4; H03, H06, H11 5e-5), as do H07 (3e-4) and H08 (5e-5). Learning rate is the one hyperparameter this search points to.", B)]

el += [PageBreak(), Paragraph("Stage 2: real test set, final round (main result)", H2), Image(f2, 26 * cm, 6.5 * cm)]
keys = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("recall", "Recall"), ("specificity", "Specificity"), ("precision", "Precision"), ("f1", "F1"),
        ("balanced_accuracy", "Balanced acc."), ("accuracy", "Accuracy"), ("mcc", "MCC"), ("brier", "Brier (lower better)"),
        ("specificity_at_recall90", "Spec. @ recall>=0.90"), ("fairness_gap_6", "Fairness gap, 6 types"), ("fairness_gap_3", "Fairness gap, 3 groups"),
        ("specificity_gap_6", "Specificity gap, 6 types")]
for kind, title in (("final_round", "final round"), ("final_best", "best-test-AUC round (optimistic, test-selected)")):
    sub = M[(M.checkpoint == kind) & M.config.isin(CONF)]
    rows = [["Metric"] + [f"{c} mean +- SD" for c in CONF] + ["H10 - H00", "seeds", "H09 - H00", "seeds"]]
    for k, lab in keys:
        row = [lab]
        for c in CONF:
            v = sub[sub.config == c][k]; row.append(f"{v.mean():.3f} +- {v.std(ddof=1):.3f}")
        for t in ("H10", "H09"):
            p = PV[(PV.config == t) & (PV.checkpoint == kind)].iloc[0]
            row += [sg(p[f"d_{k}_mean"]), f"{int(p[f'd_{k}_seeds_better'])}/3 better"]
        rows.append(row)
    if kind == "final_best": el += [PageBreak(), Paragraph(f"Stage 2: {title}", H2)]
    else: el += [Paragraph(f"Test metrics at threshold 0.5, {title}", ss["Heading3"])]
    el += [tbl(rows, [4.2 * cm, 3.3 * cm, 3.3 * cm, 3.3 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm])]
    rows = [["Comparison", "AUC diff, 95% CI (patient bootstrap)", "DeLong p (s42 / s43 / s44)", "McNemar malignant: tuned-only / default-only correct, p"]]
    for t in ("H10", "H09"):
        p = PV[(PV.config == t) & (PV.checkpoint == kind)].iloc[0]
        rows.append([f"{t} vs H00", f"{sg(p.d_roc_auc_mean)}  [{sg(p.d_auc_seedavg_boot_lo)}, {sg(p.d_auc_seedavg_boot_hi)}]",
                     " / ".join(f"{p[f'delong_p_s{s}']:.3f}" for s in SEEDS), " ; ".join(f"s{s} {p[f'mcnemar_mal_s{s}']}" for s in SEEDS)])
    el += [Spacer(1, 4), tbl(rows, [2.6 * cm, 5.4 * cm, 4.6 * cm, 12 * cm])]

el += [Spacer(1, 6), Paragraph("Reading the stage-2 results", ss["Heading3"]), Paragraph(
    "<b>Discrimination (ROC-AUC, PR-AUC) is unchanged</b> by tuning: differences are at most 0.004, all CIs include 0, no DeLong test is significant. "
    "<b>Recall at 0.5 goes up</b> for both tuned configurations; McNemar on malignant images is significant for 2 of 3 seeds for each. "
    "Because the AUC is unchanged, this is a shift of the operating point (more positives), not better ranking. "
    "For H10 it costs specificity (-0.019), precision (-0.062) and MCC (-0.021, worse at all 3 seeds) and calibration (Brier +0.009). "
    "For H09 the shift is almost free: specificity -0.003, and F1, MCC, balanced accuracy and specificity at recall>=0.90 are higher at all 3 seeds. "
    "<b>Fairness:</b> the 3-group gap is smaller for both tuned configurations at all 3 seeds (H10 -0.043, H09 -0.024), consistent with their higher recall "
    "(the gap is mostly driven by type I-II sensitivity). The 6-type gap is driven by 1-2 type-I images and is noisy.", B)]

el += [PageBreak(), Paragraph("Sanity check and limitations", H2)]
sc = M[((M.config == "E00_full_ablation") | ((M.config == "H00") & (M.seed == 42)))]
rows = [["Run", "Checkpoint", "ROC-AUC", "PR-AUC", "Recall", "Specificity", "F1", "MCC", "Gap 6", "Gap 3"]]
for r in sc.itertuples():
    rows.append([("ablation E00_full" if r.config != "H00" else "this tuning, H00") + " s42", r.checkpoint, f(r.roc_auc, 4), f(r.pr_auc, 4), f(r.recall, 4),
                 f(r.specificity, 4), f(r.f1, 4), f(r.mcc, 4), f(r.fairness_gap_6, 4), f(r.fairness_gap_3, 4)])
el += [tbl(rows), Spacer(1, 4), Paragraph(
    "Both rows use the default settings, seed 42, the same data, cache and code path (our hooks set the default values). The final-round AUC still differs "
    "by 0.006 (0.970 vs 0.965), so <b>runs are not bit-reproducible</b>: the asynchronous straggler handling depends on timing, and GPU kernels are not deterministic, "
    "and both depend on how many runs share the machine (8 runs then vs 9 now). This noise is larger than every AUC difference between configurations, "
    "which is why seeds matter and why no AUC claim can be made from this tuning.", B), Spacer(1, 4),
    Paragraph("<b>Limitations.</b> (1) 12 random configurations over 19 hyperparameters is a sparse search: it can find a better combination if one was drawn, "
              "but it cannot attribute an effect to any single hyperparameter (H10 and H09 differ from the default in 13 and 14 of the 19 settings). "
              "(2) Stage 1 used 12 rounds; the ranking at 30 rounds could differ. (3) Three seeds give low power; 'n/3 seeds better' is descriptive. "
              "(4) The test set has 417 malignant images but only 6 in types IV-VI, so the fairness gap mostly measures type I vs type II-III sensitivity. "
              "(5) No per-test multiplicity correction across the many metrics; read the per-metric differences as descriptive.", B), Spacer(1, 4),
    Paragraph("<b>Shared features of H10 and H09</b> (both differ from the default in the same direction): selection gamma 2.0 (default 1.0), aggregation gamma 0.5 "
              "(default 1.0), lambda 0.02 (default 0.05), MAD k 2.0 (default 2.5). A one-factor follow-up of these four on the default would be the cheapest way "
              "to test whether they cause the recall/fairness shift.", B), Spacer(1, 4),
    Paragraph("<b>Files</b> (laptop: six_models_dgmm_results\\novelty_hpo, DGX: /workspace/novelty_hpo): results/final/summary.md, metrics_per_run.csv, "
              "summary_by_config.csv, paired_vs_default.csv, fairness_by_skin.csv; results/stage1_table.csv, stage1_per_seed.csv; logs/selection.json; "
              "raw_stage1/, raw_stage2/ (per-run metrics, predictions, logs; model weights are only on the DGX). Archive md5 70824636cf1a9bf940a40e39d80b2a59.", SM)]

doc = SimpleDocTemplate(D + r"\Novelty_HPO_report.pdf", pagesize=landscape(A4), leftMargin=1.4 * cm, rightMargin=1.4 * cm, topMargin=1.2 * cm, bottomMargin=1.2 * cm,
                        title="Novelty hyperparameter tuning report")
doc.build(el)
print("ok")
