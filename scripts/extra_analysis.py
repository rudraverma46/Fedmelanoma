"""Extra analysis for the six Dirichlet-GMM models (requested 2026-11-21). New file; reads only.

1. Grouped fairness gap: Fitzpatrick I-II, III-IV, V-VI. Same definition as compute_results.py: pooled TP/FN per
   group, sensitivity = TP/(TP+FN), groups with no malignant image excluded, gap = max - min.
2. 95% bootstrap CIs per model and checkpoint (NBOOT resamples of the pooled test images; resampled by patient
   when the metadata has a patient id, so images of one patient stay together).
3. Paired comparisons, all 15 model pairs, same test images: DeLong test for ROC-AUC; exact McNemar on errors at 0.5
   (all images, malignant only = recall, benign only = specificity); paired-bootstrap CIs of the differences (same
   resample indices for every model). Holm correction across the 15 pairs, per test and checkpoint.
Inputs: the same test_predictions.csv + metadata_test.csv as compute_results.py.
Outputs: results/extra_grouped_fairness.csv, extra_bootstrap_ci.csv, extra_pairwise.csv, extra_summary.md.
"""
import glob, itertools, math, os, sys, time
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

B = "/workspace/six_models_dgmm"
DATA = "/workspace/hospital_two_feature/fed_ready/3_Dirichlet_GMM"
OUT = os.path.join(B, "results")
MODELS = [("FedAvg", "fedavg", "src_resnet50-fedavg-textbook/resnet50_fedavg_textbook"),
          ("FedProx", "fedprox", "src_resnet50-fedprox-textbook/resnet50_fedprox_textbook"),
          ("Novelty", "novelty", "src_resnet50-no-confirm-type/RESNET50_NOVELTY MODEL"),
          ("FedLesScan", "fedlesscan", "src_resnet_50_fedlesscan/resnet50_fedlesscan"),
          ("FedAsync", "fedasync", "src_resnet50-fedasync/resnet50_fedasync"),
          ("FedStrag", "fedstrag", "src_resnet_50_fedstrag")]
KINDS = ("final", "best")
THR, NBOOT, SEED = 0.5, 2000, 20261121
ROMAN = {"1": "I", "2": "II", "3": "III", "4": "IV", "5": "V", "6": "VI"}
GROUP3 = {"I": "I-II", "II": "I-II", "III": "III-IV", "IV": "III-IV", "V": "V-VI", "VI": "V-VI"}

def norm(v):
    if pd.isna(v): return "(missing)"
    s = str(v).strip().upper().replace("TYPE", "").strip()
    if s.endswith(".0"): s = s[:-2]
    return ROMAN.get(s, s)

frames = []
for p in sorted(glob.glob(DATA + "/*/metadata_test.csv")):
    m = pd.read_csv(p); m["hospital"] = os.path.basename(os.path.dirname(p)); frames.append(m)
meta = pd.concat(frames, ignore_index=True)
print("metadata columns:", list(meta.columns))
PID = next((c for c in ("patient_id", "patient") if c in meta.columns), None)
meta["skin6"] = meta.fitzpatrick_skin_type.map(norm)
meta["skin3"] = meta.skin6.map(GROUP3).fillna("(missing)")

preds = {}
for label, name, rel in MODELS:
    for kind in KINDS:
        f = os.path.join(B, rel, "results", name + "-dgmm-seed42", kind, "test_predictions.csv")
        d = pd.read_csv(f).set_index("isic_id")
        assert d.index.is_unique, f
        preds[(label, kind)] = d
ids = sorted(preds[(MODELS[0][0], "final")].index)
for k, d in preds.items():
    assert sorted(d.index) == ids, ("different test images", k)
base = meta.drop_duplicates("isic_id").set_index("isic_id").loc[ids]
y = preds[(MODELS[0][0], "final")].loc[ids, "y_true"].to_numpy().astype(int)
for k, d in preds.items():
    assert (d.loc[ids, "y_true"].to_numpy().astype(int) == y).all(), ("labels differ", k)
P = {k: d.loc[ids, "y_prob"].to_numpy().astype(float) for k, d in preds.items()}
N = len(y)
codes6, lev6 = pd.factorize(base.skin6.to_numpy(), sort=True)
codes3, lev3 = pd.factorize(base.skin3.to_numpy(), sort=True)
print("test images", N, "malignant", int(y.sum()), "patient id column:", PID)

def gap(yv, yh, codes, G):
    mal = np.bincount(codes[yv == 1], minlength=G)
    tp = np.bincount(codes[(yv == 1) & yh], minlength=G)
    ok = mal > 0
    if ok.sum() < 2: return float("nan")
    s = tp[ok] / mal[ok]
    return float(s.max() - s.min())

STATS = ["roc_auc", "recall", "specificity", "precision", "f1", "balanced_accuracy", "accuracy", "mcc",
         "fairness_gap_6", "fairness_gap_3"]

def point(yv, pv, c6, c3):
    yh = pv >= THR
    tp = float(((yh) & (yv == 1)).sum()); fn = float(((~yh) & (yv == 1)).sum())
    fp = float(((yh) & (yv == 0)).sum()); tn = float(((~yh) & (yv == 0)).sum())
    rec = tp / (tp + fn); spec = tn / (tn + fp); prec = tp / (tp + fp) if tp + fp else float("nan")
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return [roc_auc_score(yv, pv), rec, spec, prec, 2 * tp / (2 * tp + fp + fn), (rec + spec) / 2,
            (tp + tn) / len(yv), (tp * tn - fp * fn) / den if den else float("nan"),
            gap(yv, yh, c6, len(lev6)), gap(yv, yh, c3, len(lev3))]

keys = [(l, k) for l, _, _ in MODELS for k in KINDS]
PT = {k: np.array(point(y, P[k], codes6, codes3)) for k in keys}

# ---- 1. grouped fairness table
rows = []
for k in keys:
    yh = P[k] >= THR
    for gname, codes, lev in (("6 types", codes6, lev6), ("3 groups", codes3, lev3)):
        for gi, g in enumerate(lev):
            s = codes == gi
            mal = int(y[s].sum()); tp = int(((yh) & (y == 1) & s).sum())
            rows.append({"model": k[0], "checkpoint": k[1], "grouping": gname, "group": g, "n": int(s.sum()),
                         "malignant": mal, "TP": tp, "FN": mal - tp,
                         "sensitivity": tp / mal if mal else float("nan"), "included_in_gap": mal > 0})
fair = pd.DataFrame(rows)
fair.to_csv(os.path.join(OUT, "extra_grouped_fairness.csv"), index=False)

# ---- 2. bootstrap (shared indices: paired across models)
rng = np.random.default_rng(SEED)
if PID:
    pid = base[PID].fillna("(none)").astype(str).to_numpy()
    groups = pd.Series(np.arange(N)).groupby(pid).apply(lambda s: s.to_numpy()).tolist()
    def draw():
        pick = rng.integers(0, len(groups), len(groups))
        return np.concatenate([groups[i] for i in pick])
    print("cluster bootstrap over", len(groups), "patients")
else:
    def draw(): return rng.integers(0, N, N)
BOOT = {k: np.full((NBOOT, len(STATS)), np.nan) for k in keys}
t0 = time.time()
for b in range(NBOOT):
    idx = draw()
    yb = y[idx]
    if yb.min() == yb.max(): continue
    for k in keys:
        BOOT[k][b] = point(yb, P[k][idx], codes6[idx], codes3[idx])
    if b % 250 == 0: print("bootstrap", b, round(time.time() - t0), "s", flush=True)
ci = []
for k in keys:
    lo, hi = np.nanpercentile(BOOT[k], [2.5, 97.5], axis=0)
    for i, s in enumerate(STATS):
        ci.append({"model": k[0], "checkpoint": k[1], "metric": s, "point": PT[k][i], "ci_low": lo[i], "ci_high": hi[i]})
ci = pd.DataFrame(ci)
ci.to_csv(os.path.join(OUT, "extra_bootstrap_ci.csv"), index=False)

# ---- 3. paired tests
def delong(yv, p1, p2):
    pos = yv == 1; m, n = int(pos.sum()), int((~pos).sum())
    aucs, V10, V01 = [], [], []
    for p in (p1, p2):
        X, Y = p[pos], p[~pos]
        tx, ty = stats.rankdata(X), stats.rankdata(Y)
        tz = stats.rankdata(np.concatenate([X, Y]))
        aucs.append((tz[:m].sum() - m * (m + 1) / 2) / (m * n))
        V10.append((tz[:m] - tx) / n); V01.append(1 - (tz[m:] - ty) / m)
    S = np.cov(np.vstack(V10)) / m + np.cov(np.vstack(V01)) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    se = math.sqrt(var) if var > 0 else 0.0
    d = aucs[0] - aucs[1]
    return aucs, d, se, (2 * stats.norm.sf(abs(d) / se) if se > 0 else 1.0)

def mcnemar(ca, cb):
    n01 = int((ca & ~cb).sum()); n10 = int((~ca & cb).sum())
    p = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, float(p)

def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); adj = np.empty_like(ps); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj

pair_rows = []
for kind in KINDS:
    block = []
    for a, b in itertools.combinations([l for l, _, _ in MODELS], 2):
        ka, kb = (a, kind), (b, kind)
        aucs, d, se, p_auc = delong(y, P[ka], P[kb])
        assert abs(aucs[0] - PT[ka][0]) < 1e-9 and abs(aucs[1] - PT[kb][0]) < 1e-9, "DeLong AUC mismatch"
        ha, hb = P[ka] >= THR, P[kb] >= THR
        ca, cb = ha == (y == 1), hb == (y == 1)
        r = {"checkpoint": kind, "A": a, "B": b, "auc_A": aucs[0], "auc_B": aucs[1], "auc_diff": d,
             "auc_diff_delong_lo": d - 1.96 * se, "auc_diff_delong_hi": d + 1.96 * se, "p_delong": p_auc}
        for tag, mask in (("all", np.ones(N, bool)), ("malignant", y == 1), ("benign", y == 0)):
            n01, n10, p = mcnemar(ca[mask], cb[mask])
            r["A_only_correct_" + tag], r["B_only_correct_" + tag], r["p_mcnemar_" + tag] = n01, n10, p
        diff = BOOT[ka] - BOOT[kb]
        lo, hi = np.nanpercentile(diff, [2.5, 97.5], axis=0)
        for i, s in enumerate(STATS):
            r["diff_" + s] = PT[ka][i] - PT[kb][i]; r["diff_" + s + "_lo"] = lo[i]; r["diff_" + s + "_hi"] = hi[i]
        block.append(r)
    for col in ("p_delong", "p_mcnemar_all", "p_mcnemar_malignant", "p_mcnemar_benign"):
        adj = holm([r[col] for r in block])
        for r, v in zip(block, adj): r[col + "_holm"] = v
    pair_rows += block
pairs = pd.DataFrame(pair_rows)
pairs.to_csv(os.path.join(OUT, "extra_pairwise.csv"), index=False)

# ---- summary
f3 = lambda v: "nan" if v != v else ("%.3f" % v)
L = ["# Six models on Dirichlet-GMM: grouped fairness, CIs, paired tests", "",
     "Pooled test set: %d images, %d malignant. Threshold 0.5. Bootstrap: %d resamples, seed %d, %s." %
     (N, int(y.sum()), NBOOT, SEED, ("by patient (" + PID + ")") if PID else "by image (no patient id in metadata)"), ""]
for kind in KINDS:
    L += ["## Grouped fairness (" + kind + ")", "", "| Model | I-II | III-IV | V-VI | gap (3 groups) | gap (6 types) |", "|---|---|---|---|---|---|"]
    for l, _, _ in MODELS:
        g = fair[(fair.model == l) & (fair.checkpoint == kind) & (fair.grouping == "3 groups")].set_index("group")
        cell = lambda G: ("%d/%d = %s" % (g.loc[G, "TP"], g.loc[G, "malignant"], f3(g.loc[G, "sensitivity"]))) if G in g.index else "-"
        L.append("| %s | %s | %s | %s | %s | %s |" % (l, cell("I-II"), cell("III-IV"), cell("V-VI"), f3(PT[(l, kind)][9]), f3(PT[(l, kind)][8])))
    L += ["", "## 95%% CIs (%s)" % kind, "", "| Model | " + " | ".join(STATS) + " |", "|---" * (len(STATS) + 1) + "|"]
    for l, _, _ in MODELS:
        c = ci[(ci.model == l) & (ci.checkpoint == kind)].set_index("metric")
        L.append("| " + l + " | " + " | ".join("%s [%s, %s]" % (f3(c.loc[s, "point"]), f3(c.loc[s, "ci_low"]), f3(c.loc[s, "ci_high"])) for s in STATS) + " |")
    L += ["", "## Paired tests (%s), Holm-adjusted p" % kind, "",
          "| A vs B | AUC diff [DeLong 95%] | p DeLong (Holm) | recall diff [boot 95%] | p McNemar malignant (Holm) | spec diff [boot 95%] | p McNemar benign (Holm) | p McNemar all (Holm) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in pair_rows:
        if r["checkpoint"] != kind: continue
        L.append("| %s vs %s | %s [%s, %s] | %s (%s) | %s [%s, %s] | %s (%s) | %s [%s, %s] | %s (%s) | %s (%s) |" % (
            r["A"], r["B"], f3(r["auc_diff"]), f3(r["auc_diff_delong_lo"]), f3(r["auc_diff_delong_hi"]),
            f3(r["p_delong"]), f3(r["p_delong_holm"]), f3(r["diff_recall"]), f3(r["diff_recall_lo"]), f3(r["diff_recall_hi"]),
            f3(r["p_mcnemar_malignant"]), f3(r["p_mcnemar_malignant_holm"]), f3(r["diff_specificity"]),
            f3(r["diff_specificity_lo"]), f3(r["diff_specificity_hi"]), f3(r["p_mcnemar_benign"]),
            f3(r["p_mcnemar_benign_holm"]), f3(r["p_mcnemar_all"]), f3(r["p_mcnemar_all_holm"])))
    L.append("")
with open(os.path.join(OUT, "extra_summary.md"), "w") as fh:
    fh.write("\n".join(L) + "\n")
print("DONE", round(time.time() - t0), "s")
