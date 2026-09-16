"""
evaluate_best_model.py
======================
Standalone, production-quality evaluation script for the trained
FedMelanomaModel.

Loads  : best_fedmelanoma_model.pth
Runs   : full inference on the held-out test set
Outputs:
    confusion_matrix.png
    roc_curve.png
    pr_curve.png
    threshold_analysis.csv
    classification_report.txt
    final_test_results.txt
"""

import os
import csv
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

import matplotlib
matplotlib.use("Agg")           # headless – no display required
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)

# ---------------------------------------------------------------------------
# Project-local imports  (same API as training)
# ---------------------------------------------------------------------------
from client.dataset import MelanomaDataset
from models.model import FedMelanomaModel
from client.local_trainer import (
    configure_multiprocessing_env,
    run_inference,
    load_model,
)

# ---------------------------------------------------------------------------
# Configuration  – mirror training constants exactly
# ---------------------------------------------------------------------------
CHECKPOINT_PATH: str  = "checkpoints/fedmelanoma-20260820T103819Z-305acc19/best_global_model.pth"
BATCH_SIZE: int        = 20
POSITIVE_CLASS_IDX: int = 1           # Malignant = 1
THRESHOLDS: List[float] = [0.30, 0.40, 0.50, 0.60, 0.70]

DEVICE: torch.device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Hospital list – must match the training script exactly
HOSPITAL_NAMES: List[str] = [
    "CityGeneral_Multispecialty_Hospital",
    "GoldenSun_Public_Health_Screening_Camp",
    "Greenfield_Rural_Community_Health_Center",
    "Lakeside_Regional_Mixed_Hospital",
    "MetroSkin_Dermatology_Research_Clinic",
    "NorthCare_Tertiary_Melanoma_Institute",
    "Pinewood_Oncology_Biopsy_Center",
    "SunRay_Community_Outreach_Clinic",
    "TeleDerm_National_Screening_Network",
    "Westbridge_Academic_Dermoscopy_Unit",
]


# ===========================================================================
# 1. Model loading
# ===========================================================================
# load_model() is now defined in client/local_trainer.py (shared with the
# future Flower ClientApp). (imported above)


# ===========================================================================
# 2. Dataset / DataLoader construction
# ===========================================================================

def build_test_loader(
    hospital_names: List[str],
    batch_size: int,
) -> Tuple[DataLoader, List[str], "pd.DataFrame", np.ndarray]:
    """
    Build the held-out test DataLoader using the same MelanomaDataset and
    preprocessing as training.  No augmentations are added.

    Returns:
      loader          : the combined DataLoader (unchanged behavior)
      loaded          : flat list of hospital names actually loaded
      metadata_df     : DataFrame with columns fitzpatrick_skin_type,
                         image_type, diagnosis_confirm_type, in the exact
                         same row order run_inference()/evaluate_model()
                         will produce y_true/y_pred/y_prob in (loader uses
                         shuffle=False, so ConcatDataset order == this
                         concatenation order) -- used for subgroup
                         evaluation (Requirement 4), added in Phase 5.
      hospital_labels : per-sample originating-hospital name, same row
                         order, for the blueprint's worst-group-by-client-
                         type metric (Section 5.3) -- also new in Phase 5.

    These two additions are read-only bookkeeping over data this function
    already loads; no dataset-loading behavior changed for existing
    callers of the first two return values.
    """
    # Fix OpenCV + PyTorch multiprocessing conflict on Linux
    configure_multiprocessing_env()

    test_datasets: List[MelanomaDataset] = []
    loaded: List[str] = []

    for hospital in hospital_names:
        csv_path = f"hospital_partitions/{hospital}/metadata_test.csv"
        img_dir  = f"hospital_partitions/{hospital}/images/Test"

        if not os.path.isfile(csv_path):
            print(f"  [WARN] CSV not found, skipping: {csv_path}")
            continue

        test_datasets.append(MelanomaDataset(csv_path, img_dir))
        loaded.append(hospital)

    if not test_datasets:
        raise RuntimeError(
            "No test CSVs were found under hospital_partitions/. "
            "Verify the directory structure."
        )

    combined = ConcatDataset(test_datasets)

    loader = DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=False,          # preserve order for reproducible analysis
        num_workers=4,
        pin_memory=True,
    )

    metadata_columns = ["fitzpatrick_skin_type", "image_type", "diagnosis_confirm_type"]
    metadata_df = pd.concat(
        [ds.df[metadata_columns] for ds in test_datasets], ignore_index=True
    )
    hospital_labels = np.concatenate(
        [np.full(len(ds), hospital, dtype=object) for hospital, ds in zip(loaded, test_datasets)]
    )

    return loader, loaded, metadata_df, hospital_labels


# ===========================================================================
# 3. Inference
# ===========================================================================
# run_inference() is now defined in client/local_trainer.py, shared with
# train.py's evaluate_model() instead of duplicating the inference loop.
# (imported above)


# ===========================================================================
# 4. Metrics computation
# ===========================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """
    Compute the full suite of classification metrics.

    Standard metrics
    ----------------
    Accuracy   – fraction of correct predictions overall.
    Precision  – TP / (TP + FP); of predicted positives, how many are real.
    Recall     – TP / (TP + FN); of actual positives, how many were found.
    F1         – harmonic mean of precision and recall.
    ROC-AUC    – area under the ROC curve; threshold-independent rank metric.

    Medical AI metrics
    ------------------
    Sensitivity – identical to Recall (TP rate); how well the model detects
                  malignant cases.  Critical: a missed malignant is dangerous.
    Specificity – TN / (TN + FP); how well benign cases are correctly cleared.
    PPV         – Positive Predictive Value = Precision; confidence in a
                  positive (malignant) prediction.
    NPV         – Negative Predictive Value = TN / (TN + FN); confidence
                  that a negative prediction is truly benign.

    Imbalanced-dataset metrics
    --------------------------
    Balanced Accuracy – mean of sensitivity and specificity; robust to
                        class imbalance.
    MCC               – Matthews Correlation Coefficient; single scalar in
                        [-1, 1] that accounts for all four quadrants of the
                        confusion matrix; considered the most informative
                        single metric for imbalanced binary classification.
    PR-AUC            – area under the Precision-Recall curve; preferred over
                        ROC-AUC when the positive class is rare.

    Phase 5 fix: `labels=[0, 1]` is passed explicitly so the confusion
    matrix is always 2x2, even when y_true/y_pred contain only one class
    (previously `confusion_matrix(y_true, y_pred)` without `labels=`
    returns a 1x1 matrix in that case, and `cm.ravel()` below would raise
    ValueError: "not enough values to unpack"). This never triggered
    against the full centralized dataset (both classes are always
    present at that scale), but monitoring/subgroup_metrics.py calls this
    same function on small per-category subgroup subsets, where a
    single-class subset is common -- this fix is what makes that reuse
    safe. `run_threshold_analysis` below already uses this same
    `labels=[0, 1]` convention; this brings compute_metrics() in line
    with it.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, pos_label=POSITIVE_CLASS_IDX, zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, pos_label=POSITIVE_CLASS_IDX, zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, pos_label=POSITIVE_CLASS_IDX, zero_division=0
    )

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = float("nan")

    # Medical metrics derived from the confusion matrix
    sensitivity  = recall                                        # same as TP rate
    specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # TN rate
    ppv          = precision                                     # same as precision
    npv          = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    # Imbalanced-dataset metrics
    bal_accuracy = balanced_accuracy_score(y_true, y_pred)
    mcc          = matthews_corrcoef(y_true, y_pred)

    pr_precision, pr_recall, _ = precision_recall_curve(
        y_true, y_prob, pos_label=POSITIVE_CLASS_IDX
    )
    pr_auc = auc(pr_recall, pr_precision)

    return {
        # Standard
        "accuracy":         accuracy,
        "precision":        precision,
        "recall":           recall,
        "f1":               f1,
        "roc_auc":          roc_auc,
        # Medical AI
        "sensitivity":      sensitivity,
        "specificity":      specificity,
        "ppv":              ppv,
        "npv":              npv,
        # Imbalanced dataset
        "balanced_accuracy": bal_accuracy,
        "mcc":              mcc,
        "pr_auc":           pr_auc,
        # Raw confusion matrix values (used in reports)
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


# ===========================================================================
# 5. Console printing
# ===========================================================================

def print_metrics(metrics: Dict[str, float]) -> None:
    """Print all metrics in a clean, aligned format."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("FINAL TEST SET EVALUATION")
    print(sep)

    print("\n  — Standard Metrics —")
    print(f"  Accuracy           : {metrics['accuracy']*100:.2f}%")
    print(f"  Precision          : {metrics['precision']:.4f}")
    print(f"  Recall             : {metrics['recall']:.4f}")
    print(f"  F1 Score           : {metrics['f1']:.4f}")
    print(f"  ROC-AUC            : {metrics['roc_auc']:.4f}")

    print("\n  — Medical AI Metrics —")
    print(f"  Sensitivity        : {metrics['sensitivity']:.4f}  (= Recall)")
    print(f"  Specificity        : {metrics['specificity']:.4f}")
    print(f"  PPV                : {metrics['ppv']:.4f}  (= Precision)")
    print(f"  NPV                : {metrics['npv']:.4f}")

    print("\n  — Imbalanced-Dataset Metrics —")
    print(f"  Balanced Accuracy  : {metrics['balanced_accuracy']:.4f}")
    print(f"  MCC                : {metrics['mcc']:.4f}")
    print(f"  PR-AUC             : {metrics['pr_auc']:.4f}")

    print("\n  — Confusion Matrix —")
    tn = int(metrics["tn"]); fp = int(metrics["fp"])
    fn = int(metrics["fn"]); tp = int(metrics["tp"])
    print(f"                   Pred Benign    Pred Malignant")
    print(f"  True Benign       {tn:>9,}       {fp:>9,}")
    print(f"  True Malignant    {fn:>9,}       {tp:>9,}")
    print(f"\n{sep}\n")


def print_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Dict[str, float],
) -> None:
    """Print a labelled confusion matrix to stdout."""
    tn = int(metrics["tn"]); fp = int(metrics["fp"])
    fn = int(metrics["fn"]); tp = int(metrics["tp"])
    print("  Confusion Matrix:")
    print(f"                   Pred Benign    Pred Malignant")
    print(f"  True Benign       {tn:>9,}       {fp:>9,}")
    print(f"  True Malignant    {fn:>9,}       {tp:>9,}")


# ===========================================================================
# 6. Plots
# ===========================================================================

def save_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str = "confusion_matrix.png",
) -> None:
    """Save a styled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Benign", "Malignant"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted Label",
        ylabel="True Label",
        title="Confusion Matrix – Test Set",
    )
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Saved: {output_path}")


def save_roc_curve_plot(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    roc_auc: float,
    output_path: str = "roc_curve.png",
) -> None:
    """Save the ROC curve with AUC annotated in the title."""
    fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=POSITIVE_CLASS_IDX)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(
        fpr, tpr,
        color="#e05c5c", linewidth=2,
        label=f"ROC curve (AUC = {roc_auc:.4f})",
    )
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
    ax.set(
        xlabel="False Positive Rate (1 – Specificity)",
        ylabel="True Positive Rate (Sensitivity)",
        title=f"ROC Curve – Test Set  |  AUC = {roc_auc:.4f}",
        xlim=[0.0, 1.0],
        ylim=[0.0, 1.02],
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Saved: {output_path}")


def save_pr_curve_plot(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    pr_auc: float,
    output_path: str = "pr_curve.png",
) -> None:
    """Save the Precision-Recall curve with PR-AUC annotated in the title."""
    precision_vals, recall_vals, _ = precision_recall_curve(
        y_true, y_prob, pos_label=POSITIVE_CLASS_IDX
    )

    # Baseline = fraction of positive samples
    baseline = float(np.sum(y_true == POSITIVE_CLASS_IDX)) / len(y_true)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(
        recall_vals, precision_vals,
        color="#4c8bf5", linewidth=2,
        label=f"PR curve (AUC = {pr_auc:.4f})",
    )
    ax.axhline(
        baseline, linestyle="--", color="gray", linewidth=1,
        label=f"Random classifier (baseline = {baseline:.3f})",
    )
    ax.set(
        xlabel="Recall (Sensitivity)",
        ylabel="Precision (PPV)",
        title=f"Precision-Recall Curve – Test Set  |  PR-AUC = {pr_auc:.4f}",
        xlim=[0.0, 1.0],
        ylim=[0.0, 1.02],
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Saved: {output_path}")


# ===========================================================================
# 7. Threshold analysis
# ===========================================================================

def run_threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: List[float],
    csv_path: str = "threshold_analysis.csv",
) -> None:
    """
    Evaluate every threshold in *thresholds* and print + save a summary table.

    For each threshold t, a sample is predicted malignant iff y_prob >= t.
    Metrics computed per threshold:
      Precision  – TP / (TP + FP)
      Recall     – TP / (TP + FN)   [= Sensitivity]
      F1         – harmonic mean of precision and recall
      Specificity– TN / (TN + FP)
    """
    header = f"{'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | " \
             f"{'F1':>10} | {'Specificity':>12}"
    sep = "-" * len(header)

    print(f"\n  — Threshold Analysis —")
    print(f"  {header}")
    print(f"  {sep}")

    rows = []
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(np.int64)
        cm_t = confusion_matrix(y_true, y_pred_t, labels=[0, 1])
        tn_t, fp_t, fn_t, tp_t = cm_t.ravel()

        prec_t  = precision_score(
            y_true, y_pred_t, pos_label=POSITIVE_CLASS_IDX, zero_division=0
        )
        rec_t   = recall_score(
            y_true, y_pred_t, pos_label=POSITIVE_CLASS_IDX, zero_division=0
        )
        f1_t    = f1_score(
            y_true, y_pred_t, pos_label=POSITIVE_CLASS_IDX, zero_division=0
        )
        spec_t  = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0.0

        print(
            f"  {t:>10.2f} | {prec_t:>10.4f} | {rec_t:>10.4f} | "
            f"{f1_t:>10.4f} | {spec_t:>12.4f}"
        )
        rows.append({
            "threshold":   t,
            "precision":   round(prec_t, 6),
            "recall":      round(rec_t, 6),
            "f1":          round(f1_t, 6),
            "specificity": round(spec_t, 6),
        })

    print(f"  {sep}\n")

    # Save CSV
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["threshold", "precision", "recall", "f1", "specificity"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [✓] Saved: {csv_path}")


# ===========================================================================
# 8. Text-file reports
# ===========================================================================

def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str = "classification_report.txt",
) -> None:
    """Save sklearn's classification report to a plain-text file."""
    report = classification_report(
        y_true, y_pred,
        target_names=["Benign", "Malignant"],
        zero_division=0,
    )
    print("\n  Classification Report:")
    print(report)
    with open(output_path, "w") as fh:
        fh.write("Classification Report – FedMelanomaModel Test Set\n")
        fh.write("=" * 55 + "\n\n")
        fh.write(report)
    print(f"  [✓] Saved: {output_path}")


def save_final_results(
    metrics: Dict[str, float],
    output_path: str = "final_test_results.txt",
) -> None:
    """
    Write a professionally formatted plain-text results file containing
    all computed metrics and the confusion matrix.
    """
    tn = int(metrics["tn"]); fp = int(metrics["fp"])
    fn = int(metrics["fn"]); tp = int(metrics["tp"])

    lines = [
        "=" * 60,
        "FEDMELANOMAMODEL – FINAL TEST RESULTS",
        "=" * 60,
        "",
        "Standard Metrics",
        "-" * 40,
        f"  Accuracy           : {metrics['accuracy']*100:.2f}%",
        f"  Precision          : {metrics['precision']:.4f}",
        f"  Recall             : {metrics['recall']:.4f}",
        f"  F1 Score           : {metrics['f1']:.4f}",
        f"  ROC-AUC            : {metrics['roc_auc']:.4f}",
        "",
        "Medical AI Metrics",
        "-" * 40,
        f"  Sensitivity        : {metrics['sensitivity']:.4f}  (= Recall / TP rate)",
        f"  Specificity        : {metrics['specificity']:.4f}  (= TN / (TN + FP))",
        f"  PPV                : {metrics['ppv']:.4f}  (= Precision)",
        f"  NPV                : {metrics['npv']:.4f}  (= TN / (TN + FN))",
        "",
        "Imbalanced-Dataset Metrics",
        "-" * 40,
        f"  Balanced Accuracy  : {metrics['balanced_accuracy']:.4f}",
        f"  MCC                : {metrics['mcc']:.4f}",
        f"  PR-AUC             : {metrics['pr_auc']:.4f}",
        "",
        "Confusion Matrix",
        "-" * 40,
        f"                   Pred Benign    Pred Malignant",
        f"  True Benign       {tn:>9,}       {fp:>9,}",
        f"  True Malignant    {fn:>9,}       {tp:>9,}",
        "",
        "=" * 60,
    ]

    text = "\n".join(lines)
    print(text)

    with open(output_path, "w") as fh:
        fh.write(text + "\n")
    print(f"\n  [✓] Saved: {output_path}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print("\n" + "=" * 60)
    print("FedMelanomaModel – Evaluation Script")
    print("=" * 60)
    print(f"Device     : {DEVICE}")
    print(f"Checkpoint : {CHECKPOINT_PATH}")
    print(f"Batch size : {BATCH_SIZE}\n")

    # ------------------------------------------------------------------
    # Step 1 – Load model
    # ------------------------------------------------------------------
    model = load_model(CHECKPOINT_PATH, DEVICE)

    # ------------------------------------------------------------------
    # Step 2 – Build test DataLoader
    # ------------------------------------------------------------------
    print("\nBuilding test DataLoader …")
    test_loader, loaded_hospitals, metadata_df, hospital_labels = build_test_loader(
        HOSPITAL_NAMES, BATCH_SIZE
    )
    print(f"  Loaded {len(loaded_hospitals)} hospital partition(s).")
    print(f"  Total test samples : {len(test_loader.dataset):,}")

    # ------------------------------------------------------------------
    # Step 3 – Inference
    # ------------------------------------------------------------------
    print("\nRunning inference …")
    t_start = time.time()
    y_true, y_pred, y_prob = run_inference(
        model, test_loader, DEVICE, verbose=True, positive_class_idx=POSITIVE_CLASS_IDX
    )
    print(f"  Inference completed in {time.time() - t_start:.1f}s")

    # ------------------------------------------------------------------
    # Step 4 – Compute metrics
    # ------------------------------------------------------------------
    print("\nComputing metrics …")
    metrics = compute_metrics(y_true, y_pred, y_prob)

    # ------------------------------------------------------------------
    # Step 5 – Console output
    # ------------------------------------------------------------------
    print_metrics(metrics)

    # ------------------------------------------------------------------
    # Step 5b – Subgroup evaluation (Phase 5, Requirement 4)
    #
    # This IS the "global centralized evaluation dataset" case the
    # requirement asks to document explicitly: metadata_df/hospital_labels
    # above come from the server-held, ALL-hospitals combined test split
    # (build_test_loader's ConcatDataset), not from any single client's
    # local data -- consistent with the blueprint's own assumption of a
    # server-held validation/test set for calibration (Section 3.11.2).
    # This is a separate, centralized computation from the federated
    # per-round GLOBAL subgroup snapshot FedMelanomaStrategy.
    # aggregate_evaluate() derives privacy-preservingly from clients'
    # local confusion counts (server/strategy.py) -- both are legitimate,
    # documented, and clearly distinguished in the Phase 5 report.
    #
    # Metric computation reuses compute_metrics() (this file, unmodified
    # in meaning) via monitoring.subgroup_metrics -- not duplicated.
    # ------------------------------------------------------------------
    from monitoring.subgroup_metrics import (
        compute_all_subgroup_axes,
        compute_fairness_gap,
        compute_worst_group,
        compute_subgroup_metrics,
    )

    print("\nComputing subgroup metrics (skin type / imaging modality / confirmation type) …")
    subgroup_tables = compute_all_subgroup_axes(
        y_true, y_pred, y_prob, metadata_df, positive_class_idx=POSITIVE_CLASS_IDX
    )

    os.makedirs("results/metrics", exist_ok=True)
    pd.DataFrame([metrics]).to_csv("results/metrics/overall_metrics.csv", index=False)
    axis_to_filename = {
        "skin_type": "skin_type_metrics.csv",
        "image_type": "image_type_metrics.csv",
        "confirm_type": "confirm_type_metrics.csv",
    }
    for axis, filename in axis_to_filename.items():
        path = os.path.join("results/metrics", filename)
        subgroup_tables[axis].to_csv(path, index=False)
        print(f"  [✓] Saved: {path}")

    # Fitzpatrick fairness gap -- blueprint Section 5.3/8 definition
    # (max sensitivity - min sensitivity), computed only over skin-type
    # groups where sensitivity is defined; excluded groups (e.g. Types V/
    # VI when 0% locally-observed malignancy makes sensitivity undefined)
    # are reported explicitly, never silently dropped or fabricated.
    # NOTE: low/zero malignancy in Fitzpatrick V/VI is a documented
    # dataset-representation artifact (blueprint Section 2.3), not
    # interpreted here as any biological conclusion.
    fairness = compute_fairness_gap(subgroup_tables["skin_type"], metric_col="sensitivity")
    worst_group_by_axis = {
        axis: compute_worst_group(df, metric_col="roc_auc") for axis, df in subgroup_tables.items()
    }

    # Worst-group AUROC by ORIGINATING HOSPITAL ("client type", blueprint
    # Section 5.3's own WGA definition) -- possible here because this
    # centralized evaluation set carries hospital_labels; the federated
    # per-round path does not have this (each client only ever sees its
    # own hospital), which is why WGA-by-hospital is only reported from
    # this centralized script, not from server/strategy.py's per-round log.
    hospital_table = compute_subgroup_metrics(
        y_true, y_pred, y_prob, hospital_labels, loaded_hospitals, positive_class_idx=POSITIVE_CLASS_IDX
    )
    hospital_table.to_csv("results/metrics/hospital_metrics.csv", index=False)
    worst_group_by_hospital = compute_worst_group(hospital_table, metric_col="roc_auc")

    fairness_lines = [
        "=" * 60,
        "FAIRNESS / SUBGROUP SUMMARY",
        "=" * 60,
        f"Fitzpatrick fairness gap (max-min sensitivity): "
        f"{fairness['gap']:.4f}" if not pd.isna(fairness["gap"]) else "Fitzpatrick fairness gap: NaN (unavailable)",
        f"  Best group : {fairness['best_group']} (sensitivity={fairness['best_value']:.4f})"
        if fairness["best_group"] is not None else "  Best group : unavailable",
        f"  Worst group: {fairness['worst_group']} (sensitivity={fairness['worst_value']:.4f})"
        if fairness["worst_group"] is not None else "  Worst group: unavailable",
        f"  Excluded groups (sensitivity undefined): {fairness['excluded_groups']}",
        "",
        f"Worst-group AUROC by hospital (client type, Blueprint WGA): "
        f"{worst_group_by_hospital['worst_group']} = {worst_group_by_hospital['worst_value']:.4f}"
        if worst_group_by_hospital["worst_group"] is not None else "Worst-group AUROC by hospital: unavailable",
        "",
        "Worst-group AUROC by metadata axis:",
    ]
    for axis, info in worst_group_by_axis.items():
        if info["worst_group"] is not None:
            fairness_lines.append(f"  {axis}: {info['worst_group']} = {info['worst_value']:.4f}")
        else:
            fairness_lines.append(f"  {axis}: unavailable ({info['reason']})")

    fairness_text = "\n".join(fairness_lines)
    print("\n" + fairness_text)
    with open("results/metrics/fairness_summary.txt", "w") as fh:
        fh.write(fairness_text + "\n")
    print("  [✓] Saved: results/metrics/fairness_summary.txt")

    # ------------------------------------------------------------------
    # Step 6 – Confusion matrix plot
    # ------------------------------------------------------------------
    print("Generating plots …")
    save_confusion_matrix_plot(y_true, y_pred,  "confusion_matrix.png")
    save_roc_curve_plot(y_true, y_prob, metrics["roc_auc"], "roc_curve.png")
    save_pr_curve_plot(y_true, y_prob, metrics["pr_auc"],   "pr_curve.png")

    # ------------------------------------------------------------------
    # Step 7 – Threshold analysis
    # ------------------------------------------------------------------
    run_threshold_analysis(y_true, y_prob, THRESHOLDS, "threshold_analysis.csv")

    # ------------------------------------------------------------------
    # Step 8 – Text reports
    # ------------------------------------------------------------------
    save_classification_report(y_true, y_pred, "classification_report.txt")
    save_final_results(metrics, "final_test_results.txt")

    print("\n[✓] Evaluation complete.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
