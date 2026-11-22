"""Results for the six models on Dirichlet-GMM: one script, same pooled test set, same threshold (0.5) for all.

Reads each model's results/<run>/{final,best}/test_predictions.csv (written by its evaluation/evaluate_checkpoint.py),
joins Fitzpatrick skin type from the partition's metadata_test.csv by isic_id, and writes
    results/six_models_metrics.csv           every metric + confusion counts, per model and checkpoint
    results/six_models_fairness_by_skin.csv  per skin type: n, malignant, TP, FN, sensitivity (+ FP, TN, specificity)
    results/six_models_summary.md            readable tables
Fairness gap (FedMelanoma definition): per Fitzpatrick group, sensitivity = TP / (TP + FN) from POOLED counts (the
server sums TP/FN, it never averages client sensitivities); undefined where a group has no malignant image, those
groups are excluded and listed; gap = max - min over defined groups. Computed with the Novelty code's
monitoring/subgroup_metrics.compute_fairness_gap and checked against a direct computation.
"""
import glob, json, math, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, brier_score_loss, log_loss
B = "/workspace/six_models_dgmm"
DATA = "/workspace/hospital_two_feature/fed_ready/3_Dirichlet_GMM"
NOV = B + "/src_resnet50-no-confirm-type/RESNET50_NOVELTY MODEL"
sys.path.insert(0, NOV)
from monitoring.subgroup_metrics import compute_fairness_gap  # noqa: E402
MODELS = [("FedAvg (textbook)", "fedavg", "src_resnet50-fedavg-textbook/resnet50_fedavg_textbook"),
          ("FedProx (textbook)", "fedprox", "src_resnet50-fedprox-textbook/resnet50_fedprox_textbook"),
          ("Novelty", "novelty", "src_resnet50-no-confirm-type/RESNET50_NOVELTY MODEL"),
          ("FedLesScan", "fedlesscan", "src_resnet_50_fedlesscan/resnet50_fedlesscan"),
          ("FedAsync", "fedasync", "src_resnet50-fedasync/resnet50_fedasync"),
          ("FedStrag", "fedstrag", "src_resnet_50_fedstrag")]
THR = 0.5
meta = pd.concat([pd.read_csv(p, usecols=["isic_id", "fitzpatrick_skin_type", "image_type"])
                  for p in sorted(glob.glob(DATA + "/*/metadata_test.csv"))], ignore_index=True)
meta["skin"] = meta.fitzpatrick_skin_type.fillna("(missing)").astype(str).str.strip()

def safe(a, b): return float(a) / float(b) if b else float("nan")

def metrics(y, p):
    yhat = (p >= THR).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum()); fn = int(((yhat == 0) & (y == 1)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
    rec, spec, prec = safe(tp, tp + fn), safe(tn, tn + fp), safe(tp, tp + fp)
    f1 = safe(2 * prec * rec, prec + rec) if not math.isnan(prec) else float("nan")
    return {"n": len(y), "malignant": int(y.sum()), "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "accuracy": safe(tp + tn, len(y)), "balanced_accuracy": (rec + spec) / 2, "recall_sensitivity": rec,
            "specificity": spec, "precision_ppv": prec, "npv": safe(tn, tn + fn), "f1": f1,
            "mcc": float(matthews_corrcoef(y, yhat)), "roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))), "fpr": safe(fp, fp + tn), "fnr": safe(fn, fn + tp)}

rows, skin_rows = [], []
for label, name, rel in MODELS:
    run = f"{name}-dgmm-seed42"
    for kind in ("final", "best"):
        f = os.path.join(B, rel, "results", run, kind, "test_predictions.csv")
        if not os.path.exists(f):
            print("missing", f); continue
        d = pd.read_csv(f).merge(meta, on="isic_id", how="left", validate="one_to_one")
        if d.skin.isna().any(): sys.exit(f"{f}: images without metadata")
        y, p = d.y_true.to_numpy().astype(int), d.y_prob.to_numpy().astype(float)
        assert ((p >= THR).astype(int) == d.y_pred.to_numpy()).all(), "y_pred is not y_prob >= 0.5"
        m = metrics(y, p)
        groups = []
        for g, s in d.groupby("skin"):
            yy, pp = s.y_true.to_numpy().astype(int), s.y_prob.to_numpy().astype(float)
            yh = (pp >= THR).astype(int)
            tp = int(((yh == 1) & (yy == 1)).sum()); fn = int(((yh == 0) & (yy == 1)).sum())
            fp = int(((yh == 1) & (yy == 0)).sum()); tn = int(((yh == 0) & (yy == 0)).sum())
            r = {"model": label, "checkpoint": kind, "subgroup": g, "n": len(yy), "malignant": int(yy.sum()),
                 "TP": tp, "FN": fn, "FP": fp, "TN": tn, "sensitivity": safe(tp, tp + fn), "specificity": safe(tn, tn + fp)}
            groups.append(r); skin_rows.append(r)
        gdf = pd.DataFrame(groups)
        gap = compute_fairness_gap(gdf, metric_col="sensitivity", group_col="subgroup")
        defined = gdf.dropna(subset=["sensitivity"])
        direct = float(defined.sensitivity.max() - defined.sensitivity.min()) if len(defined) else float("nan")
        assert (math.isnan(direct) and math.isnan(gap["gap"])) or abs(direct - gap["gap"]) < 1e-12
        sdef = gdf.dropna(subset=["specificity"])
        rows.append({"model": label, "checkpoint": kind, **m, "fairness_gap_sensitivity": gap["gap"],
                     "fairness_best_group": f"{gap['best_group']} ({gap['best_value']:.3f})" if len(defined) else "",
                     "fairness_worst_group": f"{gap['worst_group']} ({gap['worst_value']:.3f})" if len(defined) else "",
                     "fairness_excluded_groups": ",".join(map(str, gap["excluded_groups"])),
                     "specificity_gap_skin": float(sdef.specificity.max() - sdef.specificity.min()) if len(sdef) else float("nan")})
out = os.path.join(B, "results"); os.makedirs(out, exist_ok=True)
res, skin = pd.DataFrame(rows), pd.DataFrame(skin_rows)
res.to_csv(os.path.join(out, "six_models_metrics.csv"), index=False)
skin.to_csv(os.path.join(out, "six_models_fairness_by_skin.csv"), index=False)
cols = ["model", "roc_auc", "pr_auc", "recall_sensitivity", "specificity", "precision_ppv", "f1", "balanced_accuracy",
        "accuracy", "mcc", "fairness_gap_sensitivity", "TP", "FN", "FP", "TN"]
with open(os.path.join(out, "six_models_summary.md"), "w") as fh:
    for kind in ("final", "best"):
        fh.write(f"## {kind} checkpoint (threshold 0.5, pooled test set)\n\n")
        fh.write(res[res.checkpoint == kind][cols].to_markdown(index=False, floatfmt=".3f") + "\n\n")
    fh.write("## Sensitivity by Fitzpatrick skin type (final checkpoint)\n\n")
    piv = skin[skin.checkpoint == "final"].pivot(index="subgroup", columns="model", values="sensitivity")
    fh.write(piv.to_markdown(floatfmt=".3f") + "\n")
print(res[res.checkpoint == "final"][cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
