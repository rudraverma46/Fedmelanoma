"""
summary_writer.py
=================
Builds and saves partition_summary.csv.

CHANGE vs original:
  create_partition_summary() now appends two extra columns:
    - target_malignant_pct   (from hospital dict, if present)
    - actual_malignant_pct   (computed from df)
"""

from pathlib import Path
import pandas as pd
from config import OUTPUT_DIR

# ── Label helpers (same keyword lists as malignant_range_adjuster) ──────────
_MALIGNANT_KW = ["malignant", "melanoma", "carcinoma", "basal cell", "squamous cell"]
_BENIGN_KW    = ["benign", "nevus", "naevus", "seborrheic keratosis",
                 "dermatofibroma", "angioma", "lentigo"]
_DIAG_COLS_PRIORITY = [
    "benign_malignant", "diagnosis_1", "diagnosis",
    "diagnosis_2", "diagnosis_3", "diagnosis_4", "diagnosis_5",
    "target", "label", "class",
]


def _find_label_cols(df: pd.DataFrame) -> list[str]:
    lower_map = {c.lower(): c for c in df.columns}
    return [lower_map[c.lower()] for c in _DIAG_COLS_PRIORITY if c.lower() in lower_map]


def _classify_row(row: pd.Series, label_cols: list[str]) -> str:
    for col in label_cols:
        val = row.get(col, None)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        s = str(val).strip().lower()
        if s in ("", "nan", "none"):
            continue
        if s == "malignant":
            return "malignant"
        if s == "benign":
            return "benign"
        if s == "indeterminate":
            return "indeterminate"
        if col.lower() in ("target", "label"):
            try:
                n = int(float(s))
                if n == 1: return "malignant"
                if n == 0: return "benign"
            except ValueError:
                pass
        if any(k in s for k in _MALIGNANT_KW):
            return "malignant"
        if any(k in s for k in _BENIGN_KW):
            return "benign"
    return "indeterminate"


def _compute_mal_pct(sub: pd.DataFrame) -> float:
    """Return malignant percentage for a hospital subset."""
    if len(sub) == 0:
        return 0.0
    label_cols = _find_label_cols(sub)
    if not label_cols:
        return 0.0
    labels = sub.apply(lambda r: _classify_row(r, label_cols), axis=1)
    mal = int((labels == "malignant").sum())
    return mal / len(sub) * 100.0


# ────────────────────────────────────────────────────────────────────────────


def create_partition_summary(df: pd.DataFrame,
                             hospitals: list[dict],
                             cols: dict,
                             copied_counts: dict,
                             missing_counts: dict) -> pd.DataFrame:
    rows = []

    # Stable label sets across all hospitals
    fitz_labels    = sorted(df[cols["fitzpatrick"]].astype(str).unique()) if cols["fitzpatrick"] else []
    img_labels     = sorted(df[cols["image_type"]].astype(str).unique())  if cols["image_type"]  else []
    confirm_labels = sorted(df[cols["confirm"]].astype(str).unique())     if cols["confirm"]     else []

    for h in hospitals:
        sub = df[df["assigned_hospital"] == h["name"]]
        row = {
            "hospital_name":           h["name"],
            "hospital_type":           h["type"],
            "total_samples":           len(sub),
            "copied_images":           copied_counts.get(h["name"], 0),
            "missing_images":          missing_counts.get(h["name"], 0),
            "w_k_phi":                 h["w_phi"],
            "w_k_tau":                 h["w_tau"],
            "w_k_lambda":              h["w_lambda"],
            # ── NEW columns ────────────────────────────────────────────────
            "target_malignant_pct":    h.get("target_malignant_pct", ""),
            "actual_malignant_pct":    round(_compute_mal_pct(sub), 2),
            # ───────────────────────────────────────────────────────────────
        }

        if cols["fitzpatrick"]:
            vc = sub[cols["fitzpatrick"]].astype(str).value_counts()
            for lab in fitz_labels:
                row[f"fitz__{lab}"] = int(vc.get(lab, 0))
        if cols["image_type"]:
            vc = sub[cols["image_type"]].astype(str).value_counts()
            for lab in img_labels:
                row[f"imgtype__{lab}"] = int(vc.get(lab, 0))
        if cols["confirm"]:
            vc = sub[cols["confirm"]].astype(str).value_counts()
            for lab in confirm_labels:
                row[f"confirm__{lab}"] = int(vc.get(lab, 0))

        rows.append(row)

    return pd.DataFrame(rows)


def save_partition_summary(summary_df: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> Path:
    out = output_dir / "partition_summary.csv"
    summary_df.to_csv(out, index=False)
    print(f"[summary_writer] partition_summary.csv -> {out}")
    return out
