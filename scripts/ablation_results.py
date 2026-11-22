"""Ablation results (Novelty, Dirichlet-GMM, no confirm type, seed 42, 30 rounds) from /workspace/abl_g3/raw (read only).

Same pooled test set, threshold 0.5 and fairness definition as compute_results.py / extra_analysis.py.
Per run and checkpoint (final_round, final_best): every metric + confusion counts, 6-type and 3-group fairness gap,
specificity gap, and specificity at the in-sample threshold that gives recall >= 0.90.
Paired vs E00_full (same checkpoint kind): DeLong (ROC-AUC), exact McNemar on malignant / benign / all images,
patient-cluster bootstrap CIs of the differences (2000 resamples), Holm over the 7 ablations.
Outputs: /workspace/six_models_dgmm/results/ablations/{metrics.csv, fairness_by_skin.csv, vs_full.csv, summary.md}
"""
import glob, math, os, sys
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

RAW = "/workspace/abl_g3/raw"
DATA = "/workspace/hospital_two_feature/fed_ready/3_Dirichlet_GMM"
OUT = "/workspace/six_models_dgmm/results/ablations"
os.makedirs(OUT, exist_ok=True)
FENCE = chr(96) * 3
RUNS = [("E00_full", "full method (A0)"), ("A1_no_diversity", "no diversity in selection + aggregation"),
        ("A2_no_attenuation", "no N_eff attenuation of diversity bonus"), ("A3_greedy_selection", "greedy top-K selection"),
        ("A4_binary_reliability", "binary reliability"), ("A5_static_reliability", "static reliability"),
        ("A6_tv_diversity", "total-variation diversity instead of EMD"), ("A8_synchronous", "synchronous (wait for all)")]
KINDS = ("final_round", "final_best")
THR, NBOOT, SEED = 0.5, 2000, 20261121
ROMAN = {"1": "I", "2": "II", "3": "III", "4": "IV", "5": "V", "6": "VI"}
GROUP3 = {"I": "I-II", "II": "I-II", "III": "III-IV", "IV": "III-IV", "V": "V-VI", "VI": "V-VI"}
def norm(v):
    if pd.isna(v): return "(missing)"
    s = str(v).strip().upper().replace("TYPE", "").strip()
    if s.endswith(".0"): s = s[:-2]
    return ROMAN.get(s, s)
meta = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(DATA + "/*/metadata_test.csv"))], ignore_index=True)
meta["skin6"] = meta.fitzpatrick_skin_type.map(norm); meta["skin3"] = meta.skin6.map(GROUP3).fillna("(missing)")
preds = {}
for r, _ in RUNS:
    for k in KINDS:
        d = pd.read_csv(os.path.join(RAW, r, "seed_42", k, "test_predictions.csv")).set_index("isic_id")
        assert d.index.is_unique; preds[(r, k)] = d
ids = sorted(preds[("E00_full", "final_round")].index)
for key, d in preds.items(): assert sorted(d.index) == ids, key
base = meta.drop_duplicates("isic_id").set_index("isic_id").loc[ids]
y = preds[("E00_full", "final_round")].loc[ids, "y_true"].to_numpy().astype(int)
for key, d in preds.items(): assert (d.loc[ids, "y_true"].to_numpy().astype(int) == y).all(), key
P = {key: d.loc[ids, "y_prob"].to_numpy().astype(float) for key, d in preds.items()}
for key, d in preds.items():
    if "y_pred" in d.columns: assert ((P[key] >= THR).astype(int) == d.loc[ids, "y_pred"].to_numpy()).all(), key
N = len(y); pid = base.patient_id.astype(str).to_numpy()
c6, lev6 = pd.factorize(base.skin6.to_numpy(), sort=True); c3, lev3 = pd.factorize(base.skin3.to_numpy(), sort=True)
print("test images", N, "malignant", int(y.sum()))

def gap(yv, yh, codes, G, cls):
    m = yv == cls; tot = np.bincount(codes[m], minlength=G)
    hit = np.bincount(codes[m & (yh if cls == 1 else ~yh)], minlength=G); ok = tot > 0
    if ok.sum() < 2: return float("nan")
    s = hit[ok] / tot[ok]; return float(s.max() - s.min())
def metrics(yv, pv, yh, cc6, cc3):
    tp = float((yh & (yv == 1)).sum()); fn = float((~yh & (yv == 1)).sum())
    fp = float((yh & (yv == 0)).sum()); tn = float((~yh & (yv == 0)).sum())
    rec, spec = tp / (tp + fn), tn / (tn + fp); prec = tp / (tp + fp) if tp + fp else float("nan")
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"roc_auc": roc_auc_score(yv, pv), "recall": rec, "specificity": spec, "precision": prec,
            "npv": tn / (tn + fn) if tn + fn else float("nan"), "f1": 2 * tp / (2 * tp + fp + fn),
            "balanced_accuracy": (rec + spec) / 2, "accuracy": (tp + tn) / len(yv),
            "mcc": (tp * tn - fp * fn) / den if den else float("nan"),
            "fairness_gap_6": gap(yv, yh, cc6, len(lev6), 1), "fairness_gap_3": gap(yv, yh, cc3, len(lev3), 1),
            "specificity_gap_6": gap(yv, yh, cc6, len(lev6), 0), "TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn)}
def spec_at_recall(yv, pv, target=0.90):
    u = np.unique(pv)[::-1]; order = np.argsort(-pv, kind="mergesort"); ps, ys = pv[order], yv[order]
    idx = np.searchsorted(-ps, -u, side="right") - 1
    tp, fp = np.cumsum(ys)[idx], np.cumsum(1 - ys)[idx]; rec = tp / ys.sum()
    k = int(np.argmax(rec >= target - 1e-12)); return float(u[k]), float(1 - fp[k] / (len(ys) - ys.sum()))

rows, fair = [], []
for r, desc in RUNS:
    rounds_csv = pd.read_csv(os.path.join(RAW, r, "seed_42", "metrics.csv"))
    best_round = int(rounds_csv.loc[rounds_csv.eval_roc_auc.idxmax(), "round"])
    for k in KINDS:
        pv = P[(r, k)]; yh = pv >= THR
        t90, s90 = spec_at_recall(y, pv)
        rows.append({"run": r, "description": desc, "checkpoint": k, "rounds": int(rounds_csv["round"].max()),
                     "best_round_by_eval_auc": best_round, **metrics(y, pv, yh, c6, c3),
                     "pr_auc": average_precision_score(y, pv), "brier": brier_score_loss(y, pv),
                     "thr_recall90": t90, "specificity_at_recall90": s90})
        for gname, cc, lev in (("6 types", c6, lev6), ("3 groups", c3, lev3)):
            for gi, g in enumerate(lev):
                s = cc == gi; mal = int((y[s] == 1).sum()); tp = int((yh & s & (y == 1)).sum())
                fair.append({"run": r, "checkpoint": k, "grouping": gname, "group": g, "n": int(s.sum()), "malignant": mal,
                             "TP": tp, "FN": mal - tp, "sensitivity": tp / mal if mal else float("nan")})
M = pd.DataFrame(rows); M.to_csv(os.path.join(OUT, "metrics.csv"), index=False)
pd.DataFrame(fair).to_csv(os.path.join(OUT, "fairness_by_skin.csv"), index=False)

def delong(yv, p1, p2):
    pos = yv == 1; m, n = int(pos.sum()), int((~pos).sum()); aucs, V10, V01 = [], [], []
    for p in (p1, p2):
        X, Y = p[pos], p[~pos]; tx, ty = stats.rankdata(X), stats.rankdata(Y); tz = stats.rankdata(np.concatenate([X, Y]))
        aucs.append((tz[:m].sum() - m * (m + 1) / 2) / (m * n)); V10.append((tz[:m] - tx) / n); V01.append(1 - (tz[m:] - ty) / m)
    S = np.cov(np.vstack(V10)) / m + np.cov(np.vstack(V01)) / n; var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    se = math.sqrt(var) if var > 0 else 0.0; d = aucs[0] - aucs[1]
    return d, (2 * stats.norm.sf(abs(d) / se) if se > 0 else 1.0)
def mcnemar(ca, cb):
    n01 = int((ca & ~cb).sum()); n10 = int((~ca & cb).sum())
    return n01, n10, (float(stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0)
def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); adj = np.empty_like(ps); run = 0.0
    for i, j in enumerate(o): run = max(run, min(1.0, (len(ps) - i) * ps[j])); adj[j] = run
    return adj
rng = np.random.default_rng(SEED)
groups = pd.Series(np.arange(N)).groupby(pid).apply(lambda s: s.to_numpy()).tolist()
IDX = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(NBOOT)]
BM = ["roc_auc", "recall", "specificity", "f1", "mcc", "fairness_gap_3"]
def bstats(pv, idx):
    yb = y[idx]; m = metrics(yb, pv[idx], pv[idx] >= THR, c6[idx], c3[idx]); return [m[s] for s in BM]
vs = []
for k in KINDS:
    ref = P[("E00_full", k)]; bref = np.array([bstats(ref, ix) for ix in IDX])
    cr = (ref >= THR) == (y == 1); block = []
    for r, _ in RUNS[1:]:
        pv = P[(r, k)]; d, p = delong(y, pv, ref); ca = (pv >= THR) == (y == 1)
        row = {"checkpoint": k, "ablation": r, "auc_diff_vs_full": d, "p_delong": p}
        for tag, msk in (("malignant", y == 1), ("benign", y == 0), ("all", np.ones(N, bool))):
            a, b, pm = mcnemar(ca[msk], cr[msk])
            row["abl_only_correct_" + tag], row["full_only_correct_" + tag], row["p_mcnemar_" + tag] = a, b, pm
        diff = np.array([bstats(pv, ix) for ix in IDX]) - bref
        lo, hi = np.nanpercentile(diff, [2.5, 97.5], axis=0)
        pt = M[(M.run == r) & (M.checkpoint == k)].iloc[0]; pr = M[(M.run == "E00_full") & (M.checkpoint == k)].iloc[0]
        for i, s in enumerate(BM):
            row["diff_" + s] = pt[s] - pr[s]; row["diff_" + s + "_lo"] = lo[i]; row["diff_" + s + "_hi"] = hi[i]
        block.append(row)
    for col in ("p_delong", "p_mcnemar_malignant", "p_mcnemar_benign", "p_mcnemar_all"):
        for row, v in zip(block, holm([b[col] for b in block])): row[col + "_holm"] = v
    vs += block
V = pd.DataFrame(vs); V.to_csv(os.path.join(OUT, "vs_full.csv"), index=False)
pd.set_option("display.width", 400); pd.set_option("display.max_columns", 80)
with open(os.path.join(OUT, "summary.md"), "w") as fh:
    fh.write("# Ablations, Dirichlet-GMM, seed 42, 30 rounds\n\n" + FENCE + "\n" + M.drop(columns=["description"]).round(4).to_string(index=False)
             + "\n" + FENCE + "\n\n## vs E00_full\n\n" + FENCE + "\n" + V.round(4).to_string(index=False) + "\n" + FENCE + "\n")
print("DONE")
