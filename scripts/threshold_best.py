"""Threshold analysis of the six BEST checkpoints on Dirichlet-GMM (requested 2026-11-21). New file; reads only.

Inputs: each model's results/<run>/best/test_predictions.csv (pooled 2,393 test images) + metadata_test.csv
(Fitzpatrick type, patient_id). There is no validation split in this data, so thresholds are reported two ways:
  in-sample   : threshold chosen on the whole pooled test set (optimistic: chosen and scored on the same images)
  cross-fitted: 5-fold GroupKFold by patient; threshold chosen on 4 folds, applied to the 5th, predictions pooled
                (honest estimate of what the selection rule gives on unseen patients)
Rules: default 0.5; Youden J max; F1 max; MCC max; highest threshold with recall >= 0.90; same with recall >= 0.95.
Grid: thresholds 0.05..0.95 step 0.05. Prediction rule everywhere: malignant if y_prob >= threshold.
Fairness (FedMelanoma definition): pooled TP/(TP+FN) per group, groups without malignant excluded, gap = max - min;
6 Fitzpatrick types and 3 groups (I-II, III-IV, V-VI); specificity gap analogous over groups with benign images.
CIs: 95% patient-cluster bootstrap (NBOOT resamples) at the in-sample thresholds, threshold held fixed.
Outputs: results/threshold_best/{grid.csv, operating_points.csv, operating_points_crossfit.csv,
fairness_by_operating_point.csv, summary.md}.
"""
import glob, math, os, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import GroupKFold

B = "/workspace/six_models_dgmm"
DATA = "/workspace/hospital_two_feature/fed_ready/3_Dirichlet_GMM"
OUT = os.path.join(B, "results", "threshold_best")
os.makedirs(OUT, exist_ok=True)
MODELS = [("FedAvg", "fedavg", "src_resnet50-fedavg-textbook/resnet50_fedavg_textbook"),
          ("FedProx", "fedprox", "src_resnet50-fedprox-textbook/resnet50_fedprox_textbook"),
          ("Novelty", "novelty", "src_resnet50-no-confirm-type/RESNET50_NOVELTY MODEL"),
          ("FedLesScan", "fedlesscan", "src_resnet_50_fedlesscan/resnet50_fedlesscan"),
          ("FedAsync", "fedasync", "src_resnet50-fedasync/resnet50_fedasync"),
          ("FedStrag", "fedstrag", "src_resnet_50_fedstrag")]
KIND, NBOOT, SEED, FOLDS = "best", 1000, 20261121, 5
GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)
RULES = ["default_0.5", "youden", "max_f1", "max_mcc", "recall_ge_0.90", "recall_ge_0.95"]
ROMAN = {"1": "I", "2": "II", "3": "III", "4": "IV", "5": "V", "6": "VI"}
GROUP3 = {"I": "I-II", "II": "I-II", "III": "III-IV", "IV": "III-IV", "V": "V-VI", "VI": "V-VI"}

def norm(v):
    if pd.isna(v): return "(missing)"
    s = str(v).strip().upper().replace("TYPE", "").strip()
    if s.endswith(".0"): s = s[:-2]
    return ROMAN.get(s, s)

meta = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(DATA + "/*/metadata_test.csv"))], ignore_index=True)
meta["skin6"] = meta.fitzpatrick_skin_type.map(norm)
meta["skin3"] = meta.skin6.map(GROUP3).fillna("(missing)")
preds = {}
for label, name, rel in MODELS:
    d = pd.read_csv(os.path.join(B, rel, "results", name + "-dgmm-seed42", KIND, "test_predictions.csv")).set_index("isic_id")
    assert d.index.is_unique
    preds[label] = d
ids = sorted(preds["FedAvg"].index)
for l, d in preds.items(): assert sorted(d.index) == ids, l
base = meta.drop_duplicates("isic_id").set_index("isic_id").loc[ids]
y = preds["FedAvg"].loc[ids, "y_true"].to_numpy().astype(int)
for l, d in preds.items(): assert (d.loc[ids, "y_true"].to_numpy().astype(int) == y).all(), l
P = {l: d.loc[ids, "y_prob"].to_numpy().astype(float) for l, d in preds.items()}
N = len(y)
pid = base.patient_id.astype(str).to_numpy()
c6, lev6 = pd.factorize(base.skin6.to_numpy(), sort=True)
c3, lev3 = pd.factorize(base.skin3.to_numpy(), sort=True)

def gap(yv, yh, codes, G, cls):
    m = yv == cls
    tot = np.bincount(codes[m], minlength=G)
    hit = np.bincount(codes[m & (yh if cls == 1 else ~yh)], minlength=G)
    ok = tot > 0
    if ok.sum() < 2: return float("nan")
    s = hit[ok] / tot[ok]
    return float(s.max() - s.min())

def metrics(yv, pv, yh, cc6, cc3, auc=None):
    tp = float((yh & (yv == 1)).sum()); fn = float((~yh & (yv == 1)).sum())
    fp = float((yh & (yv == 0)).sum()); tn = float((~yh & (yv == 0)).sum())
    n = len(yv); rec = tp / (tp + fn); spec = tn / (tn + fp)
    prec = tp / (tp + fp) if tp + fp else float("nan"); npv = tn / (tn + fn) if tn + fn else float("nan")
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
            "roc_auc": roc_auc_score(yv, pv) if auc is None else auc,
            "recall": rec, "specificity": spec, "precision": prec, "npv": npv,
            "f1": 2 * tp / (2 * tp + fp + fn), "f2": 5 * tp / (5 * tp + 4 * fn + fp),
            "balanced_accuracy": (rec + spec) / 2, "accuracy": (tp + tn) / n,
            "mcc": (tp * tn - fp * fn) / den if den else float("nan"), "youden_j": rec + spec - 1,
            "fpr": 1 - spec, "fnr": 1 - rec, "lr_pos": rec / (1 - spec) if spec < 1 else float("inf"),
            "lr_neg": (1 - rec) / spec if spec > 0 else float("nan"),
            "flagged": int(tp + fp), "flagged_rate": (tp + fp) / n,
            "fairness_gap_6": gap(yv, yh, cc6, len(lev6), 1), "fairness_gap_3": gap(yv, yh, cc3, len(lev3), 1),
            "specificity_gap_6": gap(yv, yh, cc6, len(lev6), 0)}

def curve(yv, pv):
    u = np.unique(pv)[::-1]
    order = np.argsort(-pv, kind="mergesort"); ps, ys = pv[order], yv[order]
    ctp, cfp = np.cumsum(ys), np.cumsum(1 - ys)
    idx = np.searchsorted(-ps, -u, side="right") - 1
    tp, fp = ctp[idx].astype(float), cfp[idx].astype(float)
    Pn, Nn = ys.sum(), len(ys) - ys.sum()
    fn, tn = Pn - tp, Nn - fp
    rec, spec = tp / Pn, 1 - fp / Nn
    f1 = 2 * tp / (tp + fp + Pn)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)); den[den == 0] = np.nan
    mcc = (tp * tn - fp * fn) / den
    return u, rec, spec, f1, mcc

def choose(yv, pv):
    u, rec, spec, f1, mcc = curve(yv, pv)
    out = {"default_0.5": 0.5, "youden": u[np.argmax(rec + spec - 1)], "max_f1": u[np.argmax(f1)],
           "max_mcc": u[np.nanargmax(mcc)]}
    for t in (0.90, 0.95):
        out["recall_ge_%.2f" % t] = u[np.argmax(rec >= t - 1e-12)]
    return {k: float(v) for k, v in out.items()}

t0 = time.time()
grid_rows, op_rows, cf_rows, fair_rows = [], [], [], []
TH = {}
for l, _, _ in MODELS:
    pv = P[l]; auc = roc_auc_score(y, pv)
    common = {"pr_auc": average_precision_score(y, pv), "brier": brier_score_loss(y, pv)}
    for t in GRID:
        grid_rows.append({"model": l, "threshold": t, **metrics(y, pv, pv >= t, c6, c3, auc), **common})
    TH[l] = choose(y, pv)
    for r in RULES:
        t = TH[l][r]; yh = pv >= t
        op_rows.append({"model": l, "rule": r, "threshold": t, **metrics(y, pv, yh, c6, c3, auc), **common})
        for gname, cc, lev in (("6 types", c6, lev6), ("3 groups", c3, lev3)):
            for gi, g in enumerate(lev):
                s = cc == gi; mal = int((y[s] == 1).sum()); ben = int((y[s] == 0).sum())
                tp = int((yh & s & (y == 1)).sum()); tn = int((~yh & s & (y == 0)).sum())
                fair_rows.append({"model": l, "rule": r, "threshold": t, "grouping": gname, "group": g, "n": int(s.sum()),
                                  "malignant": mal, "TP": tp, "FN": mal - tp, "sensitivity": tp / mal if mal else float("nan"),
                                  "benign": ben, "TN": tn, "FP": ben - tn, "specificity": tn / ben if ben else float("nan")})
    # cross-fitted
    yh_cf = {r: np.zeros(N, bool) for r in RULES}; th_cf = {r: [] for r in RULES}
    for tr, te in GroupKFold(n_splits=FOLDS).split(pv, y, groups=pid):
        ch = choose(y[tr], pv[tr])
        for r in RULES:
            yh_cf[r][te] = pv[te] >= ch[r]; th_cf[r].append(ch[r])
    for r in RULES:
        cf_rows.append({"model": l, "rule": r, "threshold_in_sample": TH[l][r],
                        "fold_thresholds": " ".join("%.3f" % v for v in th_cf[r]),
                        **metrics(y, pv, yh_cf[r], c6, c3, auc), **common})
print("point estimates done", round(time.time() - t0), "s", flush=True)

# patient-cluster bootstrap at in-sample thresholds
rng = np.random.default_rng(SEED)
groups = pd.Series(np.arange(N)).groupby(pid).apply(lambda s: s.to_numpy()).tolist()
CI_M = ["roc_auc", "recall", "specificity", "precision", "npv", "f1", "f2", "balanced_accuracy", "accuracy", "mcc",
        "fairness_gap_6", "fairness_gap_3"]
boot = {(l, r): [] for l, _, _ in MODELS for r in RULES}
for b in range(NBOOT):
    idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
    yb = y[idx]
    for l, _, _ in MODELS:
        pb = P[l][idx]; auc = roc_auc_score(yb, pb)
        for r in RULES:
            m = metrics(yb, pb, pb >= TH[l][r], c6[idx], c3[idx], auc)
            boot[(l, r)].append([m[k] for k in CI_M])
    if b % 200 == 0: print("bootstrap", b, round(time.time() - t0), "s", flush=True)
for row in op_rows:
    arr = np.array(boot[(row["model"], row["rule"])], float)
    lo, hi = np.nanpercentile(arr, [2.5, 97.5], axis=0)
    for i, k in enumerate(CI_M):
        row[k + "_lo"], row[k + "_hi"] = lo[i], hi[i]

grid, ops, cf, fair = map(pd.DataFrame, (grid_rows, op_rows, cf_rows, fair_rows))
grid.to_csv(os.path.join(OUT, "grid.csv"), index=False)
ops.to_csv(os.path.join(OUT, "operating_points.csv"), index=False)
cf.to_csv(os.path.join(OUT, "operating_points_crossfit.csv"), index=False)
fair.to_csv(os.path.join(OUT, "fairness_by_operating_point.csv"), index=False)

f3 = lambda v: "nan" if v != v else ("inf" if v == float("inf") else "%.3f" % v)
COLS = ["threshold", "recall", "specificity", "precision", "npv", "f1", "f2", "balanced_accuracy", "accuracy", "mcc",
        "youden_j", "lr_pos", "lr_neg", "flagged", "fairness_gap_6", "fairness_gap_3", "specificity_gap_6"]
L = ["# Threshold analysis, best checkpoints, Dirichlet-GMM", "",
     "Pooled test set %d images, %d malignant. In-sample thresholds are chosen on this test set; cross-fitted = 5-fold by patient." % (N, int(y.sum())), ""]
for title, df in (("In-sample operating points", ops), ("Cross-fitted operating points (thresholds = in-sample value; metrics from per-fold thresholds)", cf.rename(columns={"threshold_in_sample": "threshold"}))):
    L += ["## " + title, "", "| model | rule | " + " | ".join(COLS) + " | TP/FN/FP/TN |", "|---" * (len(COLS) + 3) + "|"]
    for _, r in df.iterrows():
        L.append("| %s | %s | %s | %d/%d/%d/%d |" % (r["model"], r["rule"], " | ".join(f3(r[c]) if c != "flagged" else str(int(r[c])) for c in COLS), r.TP, r.FN, r.FP, r.TN))
    L.append("")
L += ["## Threshold-free", "", "| model | roc_auc | pr_auc | brier |", "|---|---|---|---|"]
for l, _, _ in MODELS:
    r = ops[(ops.model == l) & (ops.rule == "default_0.5")].iloc[0]
    L.append("| %s | %s | %s | %s |" % (l, f3(r.roc_auc), f3(r.pr_auc), f3(r.brier)))
with open(os.path.join(OUT, "summary.md"), "w") as fh: fh.write("\n".join(L) + "\n")
print("DONE", round(time.time() - t0), "s")
