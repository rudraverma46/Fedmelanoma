"""
monitoring/subgroup_metrics.py
================================
Non-IID subgroup evaluation, reused identically by:
  - evaluate.py's centralized final evaluation (server-held test set)
  - client/client.py's federated per-client local evaluation

Metadata axes and category lists are imported directly from
client/dataset.py's MelanomaDataset (read-only) -- NOT reinvented here.
That is the single source of truth for the real
metadata_train.csv/metadata_test.csv schema:

    fitzpatrick_skin_type  -> I, II, III, IV, V, VI
    image_type             -> dermoscopic, clinical: close-up,
                               clinical: overview
    diagnosis_confirm_type -> histopathology, serial imaging showing no
                               change, single image expert consensus,
                               single contributor clinical assessment

Metric reuse: per-subgroup metrics are computed by calling
evaluate.compute_metrics() (unmodified) on the boolean-masked subset,
then NaN-guarding the specific fields that are mathematically undefined
for a single-class subgroup -- evaluate.compute_metrics() itself uses
sklearn's zero_division=0 convention, which silently reports 0.0 for an
undefined quantity; that is exactly the "silently report an invalid
metric" failure mode this module exists to prevent for subgroup rows
(evaluate.compute_metrics()'s own output is untouched for the
whole-dataset case, where the imbalance is real, not degenerate).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from client.dataset import MelanomaDataset

# ---------------------------------------------------------------------------
# Metadata axes: (human label, dataframe/array column meaning, categories)
# Pulled directly from MelanomaDataset's own class constants.
# ---------------------------------------------------------------------------
SUBGROUP_AXES: Dict[str, Sequence[str]] = {
    "skin_type": list(MelanomaDataset._SKIN_TYPES),
    "image_type": list(MelanomaDataset._IMAGE_TYPES),
    "confirm_type": list(MelanomaDataset._CONFIRM_TYPES),
}

# Metrics that are mathematically undefined when the subgroup has zero
# malignant (positive) samples -- see module docstring.
_UNDEFINED_IF_NO_MALIGNANT = ("recall", "sensitivity", "f1", "roc_auc", "pr_auc", "mcc", "balanced_accuracy")
# Metrics undefined when the subgroup has zero benign (negative) samples.
_UNDEFINED_IF_NO_BENIGN = ("specificity", "roc_auc", "pr_auc", "mcc", "balanced_accuracy")

_ALL_METRIC_KEYS = (
    "accuracy", "precision", "recall", "f1", "roc_auc",
    "sensitivity", "specificity", "ppv", "npv",
    "balanced_accuracy", "mcc", "pr_auc",
)


def _nan_row(category: str, reason: str) -> Dict[str, object]:
    row: Dict[str, object] = {
        "subgroup": category,
        "n_samples": 0,
        "n_benign": 0,
        "n_malignant": 0,
        "prevalence": np.nan,
        "unavailable_reason": reason,
    }
    row.update({k: np.nan for k in _ALL_METRIC_KEYS})
    return row


def compute_subgroup_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    group_labels: Sequence[str],
    categories: Sequence[str],
    positive_class_idx: int = 1,
) -> pd.DataFrame:
    """
    Compute the full evaluate.compute_metrics() metric family separately
    for each category in *categories*, over the boolean mask
    group_labels == category.

    Any sample whose group label doesn't match any known category is
    silently excluded from every per-category row (it is not a fabricated
    "unknown" bucket -- callers can detect this from
    sum(n_samples) < len(y_true) if needed).

    Returns a DataFrame with one row per category. Metrics undefined for
    a single-class subgroup are explicitly NaN (never a fabricated 0.0),
    with `unavailable_reason` explaining why.
    """
    from evaluate import compute_metrics  # local import: avoids import cost when unused

    group_labels = np.asarray(group_labels)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    rows: List[Dict[str, object]] = []
    for category in categories:
        mask = group_labels == category
        n = int(mask.sum())
        if n == 0:
            rows.append(_nan_row(category, "no samples in this subgroup"))
            continue

        yt, yp, ypr = y_true[mask], y_pred[mask], y_prob[mask]
        n_malignant = int((yt == positive_class_idx).sum())
        n_benign = n - n_malignant

        metrics = compute_metrics(yt, yp, ypr)
        row: Dict[str, object] = {
            "subgroup": category,
            "n_samples": n,
            "n_benign": n_benign,
            "n_malignant": n_malignant,
            "prevalence": n_malignant / n,
        }
        row.update({k: metrics.get(k, np.nan) for k in _ALL_METRIC_KEYS})

        reasons = []
        if n_malignant == 0:
            for key in _UNDEFINED_IF_NO_MALIGNANT:
                row[key] = np.nan
            reasons.append(f"{', '.join(_UNDEFINED_IF_NO_MALIGNANT)} unavailable: 0 malignant samples")
        if n_benign == 0:
            for key in _UNDEFINED_IF_NO_BENIGN:
                row[key] = np.nan
            reasons.append(f"{', '.join(_UNDEFINED_IF_NO_BENIGN)} unavailable: 0 benign samples")
        row["unavailable_reason"] = "; ".join(reasons) if reasons else ""

        rows.append(row)

    return pd.DataFrame(rows)


def compute_all_subgroup_axes(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    metadata_df: pd.DataFrame,
    positive_class_idx: int = 1,
) -> Dict[str, pd.DataFrame]:
    """
    Convenience wrapper: compute subgroup metrics for all three blueprint
    metadata axes at once from a metadata DataFrame with the real
    metadata_train.csv/metadata_test.csv column names.

    metadata_df must have the same row order/length as y_true/y_pred/y_prob
    (i.e. row i's metadata corresponds to prediction i).
    """
    column_by_axis = {
        "skin_type": "fitzpatrick_skin_type",
        "image_type": "image_type",
        "confirm_type": "diagnosis_confirm_type",
    }
    results: Dict[str, pd.DataFrame] = {}
    for axis, categories in SUBGROUP_AXES.items():
        column = column_by_axis[axis]
        group_labels = metadata_df[column].astype(str).str.strip().to_numpy()
        results[axis] = compute_subgroup_metrics(
            y_true, y_pred, y_prob, group_labels, categories, positive_class_idx
        )
    return results


# ===========================================================================
# Fairness gap / worst-group summaries (Blueprint Section 5.3 / 8)
# ===========================================================================

def compute_fairness_gap(
    subgroup_df: pd.DataFrame,
    metric_col: str = "sensitivity",
    group_col: str = "subgroup",
) -> Dict[str, object]:
    """
    Delta_fair = max_g metric(g) - min_g metric(g), computed only over
    subgroups where *metric_col* is available (not NaN) -- exactly the
    blueprint's Fitzpatrick fairness gap definition (Section 5.3 /
    Section 8), generalised to any subgroup table this module produces.

    Returns {"gap": float | NaN, "best_group": ..., "best_value": ...,
    "worst_group": ..., "worst_value": ..., "excluded_groups": [...]}.
    excluded_groups lists subgroups skipped because the metric was NaN
    there (e.g. Fitzpatrick V/VI when 0% malignancy makes sensitivity
    undefined) -- reported explicitly, never silently dropped.
    """
    valid = subgroup_df.dropna(subset=[metric_col])
    excluded = subgroup_df.loc[~subgroup_df.index.isin(valid.index), group_col].tolist()

    if valid.empty:
        return {
            "gap": np.nan,
            "best_group": None,
            "best_value": np.nan,
            "worst_group": None,
            "worst_value": np.nan,
            "excluded_groups": excluded,
            "reason": f"no subgroup has a defined {metric_col}",
        }

    best_row = valid.loc[valid[metric_col].idxmax()]
    worst_row = valid.loc[valid[metric_col].idxmin()]
    return {
        "gap": float(best_row[metric_col] - worst_row[metric_col]),
        "best_group": best_row[group_col],
        "best_value": float(best_row[metric_col]),
        "worst_group": worst_row[group_col],
        "worst_value": float(worst_row[metric_col]),
        "excluded_groups": excluded,
        "reason": "",
    }


# ===========================================================================
# Federated / privacy-preserving path (Requirement 5)
# ===========================================================================
#
# Clients must never transmit raw samples or raw metadata. What CAN cross
# the client->server boundary, consistent with how weighted_average()
# already treats fit/evaluate metrics (small aggregate numbers, not raw
# data), is a per-category CONFUSION-MATRIX COUNT (tn/fp/fn/tp) -- a
# handful of small integers per category, not per-sample data. Summing
# these across clients before computing rates (rather than averaging
# each client's already-computed rate) is also the mathematically exact
# aggregation -- summed-then-divided reproduces the true global
# sensitivity/specificity/etc.; averaged-rates would not (classic
# Simpson's-paradox risk under unequal per-client subgroup sizes).
#
# ROC-AUC / PR-AUC require per-sample probability scores, which confusion
# counts cannot reconstruct -- they are intentionally left NaN in the
# federated per-round subgroup snapshot (documented limitation; the full
# centralized evaluate.py path computes them from real per-sample scores
# on the server's own held-out set).

def local_subgroup_confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_labels: Sequence[str],
    categories: Sequence[str],
    positive_class_idx: int = 1,
) -> Dict[str, Dict[str, int]]:
    """
    CLIENT-SIDE: per-category confusion-matrix counts only (tn, fp, fn, tp,
    n). No raw sample, image, or metadata leaves the client -- only these
    small integers, exactly analogous in spirit to the num_examples int
    already transmitted in every FitRes/EvaluateRes.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    group_labels = np.asarray(group_labels)

    report: Dict[str, Dict[str, int]] = {}
    for category in categories:
        mask = group_labels == category
        n = int(mask.sum())
        if n == 0:
            report[category] = {"n": 0, "tn": 0, "fp": 0, "fn": 0, "tp": 0}
            continue
        yt, yp = y_true[mask], y_pred[mask]
        tp = int(((yt == positive_class_idx) & (yp == positive_class_idx)).sum())
        tn = int(((yt != positive_class_idx) & (yp != positive_class_idx)).sum())
        fp = int(((yt != positive_class_idx) & (yp == positive_class_idx)).sum())
        fn = int(((yt == positive_class_idx) & (yp != positive_class_idx)).sum())
        report[category] = {"n": n, "tn": tn, "fp": fp, "fn": fn, "tp": tp}
    return report


def local_all_axes_confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metadata_df: pd.DataFrame,
    positive_class_idx: int = 1,
) -> Dict[str, Dict[str, Dict[str, int]]]:
    """CLIENT-SIDE convenience: confusion counts for all three metadata axes at once."""
    column_by_axis = {
        "skin_type": "fitzpatrick_skin_type",
        "image_type": "image_type",
        "confirm_type": "diagnosis_confirm_type",
    }
    out: Dict[str, Dict[str, Dict[str, int]]] = {}
    for axis, categories in SUBGROUP_AXES.items():
        column = column_by_axis[axis]
        group_labels = metadata_df[column].astype(str).str.strip().to_numpy()
        out[axis] = local_subgroup_confusion_counts(y_true, y_pred, group_labels, categories, positive_class_idx)
    return out


def metrics_from_confusion_counts(tn: int, fp: int, fn: int, tp: int) -> Dict[str, float]:
    """
    SERVER-SIDE: rates from SUMMED confusion counts, using the same
    formulas evaluate.compute_metrics() uses on a real confusion matrix
    (same mathematical meaning, reapplied here because the federated path
    only has summed counts, not per-sample arrays to hand to sklearn).
    NaN-guarded exactly like compute_subgroup_metrics() above.
    """
    n = tn + fp + fn + tp
    n_malignant = tp + fn
    n_benign = tn + fp

    row: Dict[str, float] = {
        "n_samples": n, "n_benign": n_benign, "n_malignant": n_malignant,
        "prevalence": (n_malignant / n) if n else np.nan,
        "accuracy": ((tp + tn) / n) if n else np.nan,
        "roc_auc": np.nan,  # requires per-sample scores -- unavailable from counts alone
        "pr_auc": np.nan,
    }
    row["precision"] = row["ppv"] = (tp / (tp + fp)) if (tp + fp) > 0 else np.nan
    row["npv"] = (tn / (tn + fn)) if (tn + fn) > 0 else np.nan

    if n_malignant == 0:
        row["recall"] = row["sensitivity"] = np.nan
        row["f1"] = np.nan
        row["mcc"] = np.nan
        row["balanced_accuracy"] = np.nan
    else:
        row["recall"] = row["sensitivity"] = tp / n_malignant
        row["f1"] = (
            2 * row["precision"] * row["sensitivity"] / (row["precision"] + row["sensitivity"])
            if row["precision"] and (row["precision"] + row["sensitivity"]) > 0
            else np.nan
        )

    if n_benign == 0:
        row["specificity"] = np.nan
    else:
        row["specificity"] = tn / n_benign

    if n_malignant > 0 and n_benign > 0:
        row["balanced_accuracy"] = np.nanmean([row["sensitivity"], row["specificity"]])
        denom = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
        row["mcc"] = ((tp * tn - fp * fn) / denom) if denom > 0 else np.nan

    return row


def aggregate_client_confusion_reports(
    client_reports: Sequence[Dict[str, Dict[str, Dict[str, int]]]],
) -> Dict[str, pd.DataFrame]:
    """
    SERVER-SIDE: sum per-category confusion counts across clients (one
    report per client, each shaped like local_all_axes_confusion_counts'
    output), then compute rates once from the summed counts. This is the
    server/GLOBAL federated subgroup snapshot for one round -- distinct
    from, and computed only from, each client's already-local-only
    confusion counts (never raw predictions or metadata).
    """
    results: Dict[str, pd.DataFrame] = {}
    for axis, categories in SUBGROUP_AXES.items():
        rows = []
        for category in categories:
            tn = fp = fn = tp = 0
            for report in client_reports:
                counts = report.get(axis, {}).get(category)
                if not counts:
                    continue
                tn += counts.get("tn", 0)
                fp += counts.get("fp", 0)
                fn += counts.get("fn", 0)
                tp += counts.get("tp", 0)
            row = {"subgroup": category}
            row.update(metrics_from_confusion_counts(tn, fp, fn, tp))
            rows.append(row)
        results[axis] = pd.DataFrame(rows)
    return results


def compute_worst_group(
    subgroup_df: pd.DataFrame,
    metric_col: str = "roc_auc",
    group_col: str = "subgroup",
) -> Dict[str, object]:
    """
    Worst-group value of *metric_col* among subgroups where it is defined
    -- the blueprint's WGA = min_g metric(g) (Section 5.3), applied here
    per metadata axis (skin_type / image_type / confirm_type) rather than
    per-hospital "client type" grouping, since the centralized evaluation
    pipeline's held-out set is not currently tagged by originating
    hospital. This is a documented scope choice, not a redefinition of
    the metric itself.
    """
    valid = subgroup_df.dropna(subset=[metric_col])
    if valid.empty:
        return {"worst_group": None, "worst_value": np.nan, "reason": f"no subgroup has a defined {metric_col}"}
    worst_row = valid.loc[valid[metric_col].idxmin()]
    return {"worst_group": worst_row[group_col], "worst_value": float(worst_row[metric_col]), "reason": ""}
