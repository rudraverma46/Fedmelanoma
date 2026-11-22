"""Final report of the Novelty hyperparameter tuning (v3) on Dirichlet-GMM, no confirm type.

Stage 2 runs: the top-2 non-default configurations from stage 1 (validation ranking, mean over seeds 42-44) and the
default H00, each retrained on the full training data for 30 rounds at seeds 42, 43, 44 (raw_stage2).
Metric code = /workspace/six_models_dgmm/ablation_results.py (pooled test set, threshold 0.5, fairness gap = max - min
sensitivity over Fitzpatrick types with >= 1 malignant image; 6 types and 3 groups I-II / III-IV / V-VI).
Per run and checkpoint (final_round, final_best): all metrics. Per configuration: mean +- SD over the 3 seeds.
Paired vs H00 (same seed, same checkpoint kind): per-seed differences, mean and SD, per-seed DeLong and McNemar
(malignant images), and a patient-cluster bootstrap CI (2000) of the seed-averaged difference.
Sanity check: /workspace/abl_g3/raw/E00_full/seed_42 (default, same protocol) next to H00 seed 42.
NOTE final_best = round with the best test AUC during training, i.e. chosen on the test set (optimistic).
Env overrides (testing only): HPO_RAW2, HPO_DATA, HPO_OUT, HPO_CONFIGS, HPO_SKIP_REF.
"""
import glob, json, math, os
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

H = "/workspace/novelty_hpo"
RAW2 = os.environ.get("HPO_RAW2", H + "/raw_stage2")
DATA = os.environ.get("HPO_DATA", "/workspace/hospital_two_feature/fed_ready/3_Dirichlet_GMM")
OUT = os.environ.get("HPO_OUT", H + "/results/final")
REF = "/workspace/abl_g3/raw/E00_full/seed_42"
SEEDS = (42, 43, 44)
KINDS = ("final_round", "final_best")
THR, NBOOT, BSEED = 0.5, 2000, 20261121
ROMAN = {"1": "I", "2": "II", "3": "III", "4": "IV", "5": "V", "6": "VI"}
GROUP3 = {"I": "I-II", "II": "I-II", "III": "III-IV", "IV": "III-IV", "V": "V-VI", "VI": "V-VI"}
os.makedirs(OUT, exist_ok=True)
sel = json.load(open(H + "/logs/selection.json"))
CONFIGS = os.environ.get("HPO_CONFIGS", " ".join(sel["top2_non_default"] + ["C00"])).split()
TUNED = [c for c in CONFIGS if c != "C00"]
trials = json.load(open(H + "/code/ablation/hpo_trials.json"))

def norm(v):
    if pd.isna(v): return "(missing)"
    s = str(v).strip().upper().replace("TYPE", "").strip()
    if s.endswith(".0"): s = s[:-2]
    return ROMAN.get(s, s)
meta = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(DATA + "/*/metadata_test.csv"))], ignore_index=True)
meta["skin6"] = meta.fitzpatrick_skin_type.map(norm); meta["skin3"] = meta.skin6.map(GROUP3).fillna("(missing)")

runs = [(c, s, os.path.join(RAW2, c, f"seed_{s}")) for c in CONFIGS for s in SEEDS]
if not os.environ.get("HPO_SKIP_REF"): runs.append(("E00_full_ablation", 42, REF))
preds = {}
for c, s, d in runs:
    for k in KINDS:
        p = pd.read_csv(os.path.join(d, k, "test_predictions.csv")).set_index("isic_id")
        assert p.index.is_unique; preds[(c, s, k)] = p
ids = sorted(preds[(CONFIGS[0], SEEDS[0], "final_round")].index)
for key, d in preds.items(): assert sorted(d.index) == ids, key
base = meta.drop_duplicates("isic_id").set_index("isic_id").loc[ids]
y = preds[(CONFIGS[0], SEEDS[0], "final_round")].loc[ids, "y_true"].to_numpy().astype(int)
for key, d in preds.items(): assert (d.loc[ids, "y_true"].to_numpy().astype(int) == y).all(), key
P = {key: d.loc[ids, "y_prob"].to_numpy().astype(float) for key, d in preds.items()}
for key, d in preds.items():
    if "y_pred" in d.columns: assert ((P[key] >= THR).astype(int) == d.loc[ids, "y_pred"].to_numpy()).all(), key
N = len(y); pid = base.patient_id.astype(str).to_numpy()
c6, lev6 = pd.factorize(base.skin6.to_numpy(), sort=True); c3, lev3 = pd.factorize(base.skin3.to_numpy(), sort=True)
print("test images", N, "malignant", int(y.sum()), "patients", len(set(pid)))

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

rows, fair = [], []
for c, s, d in runs:
    rc = pd.read_csv(os.path.join(d, "metrics.csv"))
    best_round = int(rc.loc[rc.eval_roc_auc.idxmax(), "round"])
    for k in KINDS:
        pv = P[(c, s, k)]; yh = pv >= THR; t90, s90 = spec_at_recall(y, pv)
        rows.append({"config": c, "seed": s, "checkpoint": k, "rounds": int(rc["round"].max()), "best_round_by_eval_auc": best_round,
                     **metrics(y, pv, yh, c6, c3), "pr_auc": average_precision_score(y, pv), "brier": brier_score_loss(y, pv),
                     "thr_recall90": t90, "specificity_at_recall90": s90})
        for gname, cc, lev in (("6 types", c6, lev6), ("3 groups", c3, lev3)):
            for gi, g in enumerate(lev):
                m_ = cc == gi; mal = int((y[m_] == 1).sum()); tp = int((yh & m_ & (y == 1)).sum())
                fair.append({"config": c, "seed": s, "checkpoint": k, "grouping": gname, "group": g, "n": int(m_.sum()),
                             "malignant": mal, "TP": tp, "FN": mal - tp, "sensitivity": tp / mal if mal else float("nan")})
M = pd.DataFrame(rows); M.to_csv(OUT + "/metrics_per_run.csv", index=False)
pd.DataFrame(fair).to_csv(OUT + "/fairness_by_skin.csv", index=False)
KEYM = ["roc_auc", "pr_auc", "recall", "specificity", "precision", "f1", "balanced_accuracy", "accuracy", "mcc", "brier",
        "specificity_at_recall90", "fairness_gap_6", "fairness_gap_3", "specificity_gap_6"]
S2 = M[M.config.isin(CONFIGS)]
agg = S2.groupby(["config", "checkpoint"])[KEYM].agg(["mean", "std"])
agg.columns = [a + "_" + b for a, b in agg.columns]; agg = agg.reset_index(); agg.to_csv(OUT + "/summary_by_config.csv", index=False)

rng = np.random.default_rng(BSEED); upid, inv = np.unique(pid, return_inverse=True)
members = [np.flatnonzero(inv == i) for i in range(len(upid))]
boots = [np.concatenate([members[i] for i in rng.integers(0, len(upid), len(upid))]) for _ in range(NBOOT)]
def auc_fast(yv, pv):
    r = stats.rankdata(pv); m = yv.sum(); n = len(yv) - m
    return (r[yv == 1].sum() - m * (m + 1) / 2) / (m * n)
pair = []
for t in TUNED:
    for k in KINDS:
        rec = {"config": t, "checkpoint": k}
        for met in KEYM:
            dif = [float(M[(M.config == t) & (M.seed == s) & (M.checkpoint == k)][met].iloc[0] -
                         M[(M.config == "C00") & (M.seed == s) & (M.checkpoint == k)][met].iloc[0]) for s in SEEDS]
            rec[f"d_{met}_mean"] = float(np.mean(dif)); rec[f"d_{met}_sd"] = float(np.std(dif, ddof=1))
            for s, dv in zip(SEEDS, dif): rec[f"d_{met}_s{s}"] = dv
            rec[f"d_{met}_seeds_better"] = int(sum((dv < 0) if met in ("brier", "fairness_gap_6", "fairness_gap_3", "specificity_gap_6") else (dv > 0) for dv in dif))
        for s in SEEDS:
            pa, pb = P[(t, s, k)], P[("C00", s, k)]
            rec[f"delong_p_s{s}"] = delong(y, pa, pb)[1]
            mal = y == 1; ca, cb = (pa >= THR)[mal], (pb >= THR)[mal]
            n01, n10, pm = mcnemar(ca, cb); rec[f"mcnemar_mal_s{s}"] = f"{n01}/{n10} p={pm:.3g}"
        bd = []
        for b in boots:
            yb = y[b]
            bd.append(np.mean([auc_fast(yb, P[(t, s, k)][b]) - auc_fast(yb, P[("C00", s, k)][b]) for s in SEEDS]))
        rec["d_auc_seedavg_boot_lo"], rec["d_auc_seedavg_boot_hi"] = float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))
        pair.append(rec)
PV = pd.DataFrame(pair); PV.to_csv(OUT + "/paired_vs_default.csv", index=False)

def f(v, nd=3): return "nan" if pd.isna(v) else f"{v:.{nd}f}"
L = ["# Novelty hyperparameter tuning: final report (Dirichlet-GMM, no confirm type)", "",
     f"Pooled test set: {N} images, {int(y.sum())} malignant, {len(upid)} patients. Threshold 0.5. Seeds {SEEDS}.",
     f"Stage 1 ranking (validation, mean over 3 seeds): {', '.join(sel['ranking'])}; default H00 rank {sel['default_rank']}.",
     f"Stage 2 configurations: {', '.join(CONFIGS)} (30 rounds, full training data, real test set).",
     "final_best is the round with the best test AUC during training (chosen on the test set, optimistic); final_round is the honest one.", ""]
L.append("## Hyperparameters of the stage-2 configurations"); L.append("")
hp = pd.DataFrame({c: trials[c] for c in CONFIGS}); L.append(hp.to_markdown()); L.append("")
for k in KINDS:
    L.append(f"## Test metrics, {k}: mean +- SD over seeds"); L.append("")
    a = agg[agg.checkpoint == k].set_index("config")
    tab = pd.DataFrame({met: [f(a.loc[c, met + "_mean"]) + " +- " + f(a.loc[c, met + "_std"]) for c in CONFIGS] for met in KEYM}, index=CONFIGS).T
    L.append(tab.to_markdown()); L.append("")
    L.append(f"## Paired difference vs H00, {k} (tuned - default, same seed)"); L.append("")
    for t in TUNED:
        r = PV[(PV.config == t) & (PV.checkpoint == k)].iloc[0]
        tb = pd.DataFrame({"mean diff": [f(r[f"d_{m}_mean"], 4) for m in KEYM], "SD": [f(r[f"d_{m}_sd"], 4) for m in KEYM],
                           **{f"s{s}": [f(r[f"d_{m}_s{s}"], 4) for m in KEYM] for s in SEEDS},
                           "seeds better": [f"{int(r[f'd_{m}_seeds_better'])}/3" for m in KEYM]}, index=KEYM)
        L.append(f"### {t} vs H00"); L.append(""); L.append(tb.to_markdown()); L.append("")
        L.append(f"Seed-averaged ROC-AUC difference, patient bootstrap 95% CI: [{f(r['d_auc_seedavg_boot_lo'], 4)}, {f(r['d_auc_seedavg_boot_hi'], 4)}]. "
                 + "Per-seed DeLong p: " + ", ".join(f"s{s} {r[f'delong_p_s{s}']:.3g}" for s in SEEDS)
                 + ". McNemar on malignant images (tuned-only correct / default-only correct): " + ", ".join(f"s{s} {r[f'mcnemar_mal_s{s}']}" for s in SEEDS) + ".")
        L.append("")
ref = M[M.config == "E00_full_ablation"]
if len(ref):
    L.append("## Sanity check: ablation E00_full (default, seed 42) vs our H00 seed 42"); L.append("")
    cols = ["config", "seed", "checkpoint", "roc_auc", "pr_auc", "recall", "specificity", "f1", "mcc", "fairness_gap_6", "fairness_gap_3"]
    L.append(pd.concat([ref, M[(M.config == "C00") & (M.seed == 42)]])[cols].round(4).to_markdown(index=False)); L.append("")
for name in ("stage1_table.csv", "stage1_per_seed.csv"):
    p = H + "/results/" + name
    if os.path.exists(p):
        t1 = pd.read_csv(p); keep = [c for c in t1.columns if c in ("trial", "seed", "status", "val_auc_last3_mean", "val_auc_last3_sd", "val_auc_last3_min",
             "val_auc_final_mean", "val_pr_auc_last3_mean", "val_auc_last3_s42", "val_auc_last3_s43", "val_auc_last3_s44", "val_auc_last3", "val_auc_final", "val_auc_best")]
        L.append(f"## Stage 1: {name} (validation patients, 12 rounds)"); L.append(""); L.append(t1[keep].round(4).to_markdown(index=False)); L.append("")
open(OUT + "/summary.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
