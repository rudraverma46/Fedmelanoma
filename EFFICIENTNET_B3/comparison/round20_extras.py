"""Extra analysis for the round-20 result PDF. Reads saved predictions only; evaluation/ stays unchanged.

The 20-round report needs figures that leakage_analysis.py and compare_models.py do not write, because
they only cover final_best and final_round50. EfficientNet-B3's best round is 24, so its round 20
has its own folder, final_round20. Writes results/<run>/final_round20/round20_extras.json:

  - the leakage views, using the functions in leakage_analysis.py
  - the round-20 selected hospitals, to explain why the per-round score differs from the full test set
  - a paired comparison with ResNet-50 at the same round, using compare_models.paired_bootstrap

Run from this folder with the training Python and PYTHONPATH=.deps.
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "comparison")]

import leakage_analysis as leak  # noqa: E402
from compare_models import paired_bootstrap  # noqa: E402
from runtime.config import load_config  # noqa: E402

RUN, STOP = "efficientnetb3-fixedmap-seed42", 20
RESNET = os.path.join(ROOT, "..", "RESNET 50_ fed novelty", "RESNET50_EFFICIENTNET_B3_EQUIVALENT", "results", "resnet50-seed42", "final_best")
RES = os.path.join(ROOT, "results", RUN)

root = load_config()["data"]["partitions_root"]
train, test = leak.read_split(root, "train"), leak.read_split(root, "test")
test = test.assign(seen_patient=test.patient_id.isin(set(train.patient_id.dropna())), seen_lesion=test.lesion_id.isin(set(train.lesion_id.dropna())))
pred = pd.read_csv(os.path.join(RES, "final_round20", "test_predictions.csv"))
frame = pred.merge(test, on="isic_id", how="left", validate="one_to_one")
assert len(frame) == 1952
histo = frame.diagnosis_confirm_type.fillna("").str.contains("histopathology", case=False)
out = {"leakage": {"all_rows": {**leak.scores(frame), "patient_clustered_bootstrap": leak.clustered_interval(frame)},
                   "neither_patient_nor_lesion_seen": leak.scores(frame[~frame.seen_patient & ~frame.seen_lesion]),
                   "histopathology_confirmed_only": {**leak.scores(frame[histo]), "patient_clustered_bootstrap": leak.clustered_interval(frame[histo])}}}

# Clients are the sorted hospitals. Per-round scores are example-weighted means of each client's own score.
hospitals = sorted(pred.hospital.unique())
outcomes = pd.read_csv(os.path.join(RES, "tables", "client_round_outcomes.csv"))
selected = [hospitals[i] for i in sorted(outcomes[outcomes["round"] == STOP].client_id)]
per_hospital = {h: {"rows": len(g), "malignant": int(g.y_true.sum()), "roc_auc": float(roc_auc_score(g.y_true, g.y_prob))} for h, g in pred.groupby("hospital")}
rows = sum(per_hospital[h]["rows"] for h in selected)
out["round_selection"] = {"selected": selected, "not_selected": [h for h in hospitals if h not in selected], "per_hospital": per_hospital,
                          "selected_rows": rows, "selected_malignant": sum(per_hospital[h]["malignant"] for h in selected),
                          "rebuilt_weighted_roc_auc": sum(per_hospital[h]["rows"] * per_hospital[h]["roc_auc"] for h in selected) / rows,
                          "logged_round_roc_auc": float(pd.read_csv(os.path.join(RES, "tables", "per_round_metrics.csv")).set_index("round").roc_auc[STOP])}

resnet = pd.read_csv(os.path.join(RESNET, "test_predictions.csv"))
both = resnet.merge(pred, on="isic_id", suffixes=("_a", "_b"), validate="one_to_one").merge(test, on="isic_id", validate="one_to_one")
assert len(both) == 1952 and (both.y_true_a == both.y_true_b).all()
paired = pd.DataFrame({"patient": both.patient_id.fillna(both.isic_id), "y_true": both.y_true_a, "p_a": both.y_prob_a, "p_b": both.y_prob_b})
out["paired_with_resnet50_round20"] = paired_bootstrap(paired)
with open(os.path.join(RES, "final_round20", "round20_extras.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps(out, indent=1))
