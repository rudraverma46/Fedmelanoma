"""Quantifies the reference's known leakage limits on a run's saved test predictions.

The evaluation itself is unchanged. This only re-reads test_predictions.csv and splits the same
1,952 rows by what the model could have seen, so the headline numbers can be read in context:

  - rows whose patient / lesion also appears in any hospital's training partition
  - rows confirmed by histopathology: every malignant test row is, so within this group
    diagnosis_confirm_type carries no information and cannot act as a label shortcut
  - confirm_type alone as a classifier, with no image at all
  - patient-clustered bootstrap intervals, since rows from one patient are not independent

Writes results/<run_id>/leakage_analysis.json. Usage: python evaluation/leakage_analysis.py <run_id>
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from runtime.config import load_config  # noqa: E402

RESAMPLES = 2000
ID_COLUMNS = ["isic_id", "patient_id", "lesion_id", "diagnosis_confirm_type"]


def read_split(root: str, split: str) -> pd.DataFrame:
    frames = [pd.read_csv(os.path.join(root, h, f"metadata_{split}.csv"), usecols=ID_COLUMNS, low_memory=False)
              for h in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, h))]
    return pd.concat(frames, ignore_index=True)


def scores(frame: pd.DataFrame) -> dict:
    y = frame.y_true.to_numpy()
    out = {"rows": int(len(y)), "malignant": int(y.sum())}
    if 0 < y.sum() < len(y):
        out["roc_auc"] = float(roc_auc_score(y, frame.y_prob))
        out["pr_auc"] = float(average_precision_score(y, frame.y_prob))
    if y.sum():
        out["recall_at_0.5"] = float(((frame.y_prob >= 0.5) & (y == 1)).sum() / y.sum())
    return out


def clustered_interval(frame: pd.DataFrame, seed: int = 42) -> dict:
    """Resamples whole patients. Rows without a patient_id each count as their own patient."""
    groups = frame.patient_id.fillna(frame.isic_id).to_numpy()
    index = pd.Series(np.arange(len(frame))).groupby(groups).apply(lambda s: s.to_numpy()).to_list()
    y, p = frame.y_true.to_numpy(), frame.y_prob.to_numpy()
    rng = np.random.default_rng(seed)
    roc, pr = [], []
    for _ in range(RESAMPLES):
        rows = np.concatenate([index[i] for i in rng.integers(0, len(index), len(index))])
        if 0 < y[rows].sum() < len(rows):
            roc.append(roc_auc_score(y[rows], p[rows]))
            pr.append(average_precision_score(y[rows], p[rows]))
    return {"resamples": len(roc), "patients": len(index), "roc_auc_95": [float(np.percentile(roc, q)) for q in (2.5, 97.5)],
            "pr_auc_95": [float(np.percentile(pr, q)) for q in (2.5, 97.5)]}


def main() -> None:
    run_id = sys.argv[1]
    root = load_config()["data"]["partitions_root"]
    train, test = read_split(root, "train"), read_split(root, "test")
    train_patients, train_lesions = set(train.patient_id.dropna()), set(train.lesion_id.dropna())
    test = test.assign(seen_patient=test.patient_id.isin(train_patients), seen_lesion=test.lesion_id.isin(train_lesions))

    out = {"note": "Same rows and predictions as the reported evaluation; nothing here was used to select a model.",
           "seen_patient_rows": int(test.seen_patient.sum()), "seen_lesion_rows": int(test.seen_lesion.sum()),
           "missing_patient_id": int(test.patient_id.isna().sum()), "missing_lesion_id": int(test.lesion_id.isna().sum()),
           "confirm_type_counts": test.diagnosis_confirm_type.fillna("(missing)").value_counts().to_dict(), "models": {}}

    for tag in ("final_best", "final_round50"):
        pred = pd.read_csv(os.path.join("results", run_id, tag, "test_predictions.csv"))
        frame = pred.merge(test, on="isic_id", how="left", validate="one_to_one")
        assert len(frame) == len(pred) == len(test), "predictions and test metadata differ"
        histo = frame.diagnosis_confirm_type.fillna("").str.contains("histopathology", case=False)
        if "confirm_type_alone" not in out:
            rate = frame.groupby(frame.diagnosis_confirm_type.fillna("(missing)")).y_true.transform("mean")
            out["confirm_type_alone"] = {"roc_auc": float(roc_auc_score(frame.y_true, rate)), "pr_auc": float(average_precision_score(frame.y_true, rate)),
                                         "note": "Scored by each confirmation type's malignant rate in this test set. No image used."}
        out["models"][tag] = {
            "all_rows": {**scores(frame), "patient_clustered_bootstrap": clustered_interval(frame)},
            "lesion_seen_in_training": scores(frame[frame.seen_lesion]),
            "lesion_not_seen_in_training": scores(frame[~frame.seen_lesion]),
            "patient_seen_in_training": scores(frame[frame.seen_patient]),
            "patient_not_seen_in_training": scores(frame[~frame.seen_patient]),
            "neither_patient_nor_lesion_seen": scores(frame[~frame.seen_patient & ~frame.seen_lesion]),
            "histopathology_confirmed_only": {**scores(frame[histo]), "patient_clustered_bootstrap": clustered_interval(frame[histo])},
            "histopathology_and_lesion_not_seen": scores(frame[histo & ~frame.seen_lesion]),
        }
    path = os.path.join("results", run_id, "leakage_analysis.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
