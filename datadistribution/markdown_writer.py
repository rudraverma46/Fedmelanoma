"""
markdown_writer.py
==================
Writes hospital_description.md per hospital and the global
process_explanation.md.

CHANGE vs original:
  write_hospital_description_md() now shows:
    - Target malignant %  (from hospital dict key 'target_malignant_pct')
    - Actual malignant %  (computed from df_hosp)
  write_process_explanation_md() now shows a malignant-range summary table.
"""

from pathlib import Path
import pandas as pd
from config import OUTPUT_DIR, TAU_SOFT, RANDOM_SEED

# ── Label helpers (same keyword lists as malignant_range_adjuster) ──────────
_MALIGNANT_KW = ["malignant", "melanoma", "carcinoma", "basal cell", "squamous cell"]
_BENIGN_KW    = ["benign", "nevus", "naevus", "seborrheic keratosis",
                 "dermatofibroma", "angioma", "lentigo"]
_DIAG_COLS_PRIORITY = [
    "benign_malignant", "diagnosis_1", "diagnosis",
    "diagnosis_2", "diagnosis_3", "diagnosis_4", "diagnosis_5",
    "target", "label", "class",
]


def _detect_label_columns(df: pd.DataFrame) -> list[str]:
    lower_map = {c.lower(): c for c in df.columns}
    return [lower_map[c.lower()] for c in _DIAG_COLS_PRIORITY if c.lower() in lower_map]


def _classify_benign_malignant(row: pd.Series, label_cols: list[str]) -> str:
    for col in label_cols:
        val = row.get(col, "")
        if pd.isna(val):
            continue
        s = str(val).lower().strip()
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
    return "unknown"


def _value_counts_md(series: pd.Series) -> str:
    if series is None or len(series) == 0:
        return "  - (no data)\n"
    vc = series.astype(str).value_counts()
    if vc.empty:
        return "  - (no data)\n"
    return "\n".join([f"  - **{k}**: {v}" for k, v in vc.items()]) + "\n"


def _benign_malignant_md(df_hosp: pd.DataFrame) -> str:
    label_cols = _detect_label_columns(df_hosp)
    if not label_cols:
        return (
            "Label columns used: **No suitable label/diagnosis column found**\n\n"
            "| Class | Number of Images | Percentage |\n"
            "|-------|------------------|------------|\n"
            f"| Unknown | {len(df_hosp)} | 100.00% |\n"
        )

    labels = df_hosp.apply(lambda row: _classify_benign_malignant(row, label_cols), axis=1)
    counts = labels.value_counts().to_dict()
    total  = len(df_hosp)

    benign      = int(counts.get("benign", 0))
    malignant   = int(counts.get("malignant", 0))
    indet       = int(counts.get("indeterminate", 0))
    unknown     = int(counts.get("unknown", 0))

    benign_pct    = (benign    / total * 100) if total else 0
    malignant_pct = (malignant / total * 100) if total else 0
    indet_pct     = (indet     / total * 100) if total else 0
    unknown_pct   = (unknown   / total * 100) if total else 0

    return (
        f"Label columns used: **{', '.join(label_cols)}**\n\n"
        "| Class | Number of Images | Percentage |\n"
        "|-------|------------------|------------|\n"
        f"| Benign | {benign} | {benign_pct:.2f}% |\n"
        f"| Malignant | {malignant} | {malignant_pct:.2f}% |\n"
        f"| Indeterminate | {indet} | {indet_pct:.2f}% |\n"
        f"| Unknown / not detected | {unknown} | {unknown_pct:.2f}% |\n"
    )


def _compute_actual_malignant_pct(df_hosp: pd.DataFrame) -> float:
    """Return actual malignant % for this hospital subset."""
    if len(df_hosp) == 0:
        return 0.0
    label_cols = _detect_label_columns(df_hosp)
    if not label_cols:
        return 0.0
    labels = df_hosp.apply(lambda r: _classify_benign_malignant(r, label_cols), axis=1)
    mal = int((labels == "malignant").sum())
    return mal / len(df_hosp) * 100.0


def _wk_reason_md(hospital: dict) -> str:
    return f"""
The vector was generated from the hospital's clinical archetype and the dataset distribution.

| Component | Direction | Value | Why this value makes sense |
|-----------|-----------|-------|----------------------------|
| w_k_phi | {hospital['directions']['phi']} | {hospital['w_phi']:.4f} | Controls preference for higher Fitzpatrick skin type score. `high` means more preference for darker skin types, `low` means less preference, and `neutral` means balanced. |
| w_k_tau | {hospital['directions']['tau']} | {hospital['w_tau']:.4f} | Controls preference for higher image-type score. Higher values favor dermoscopic/high-quality images. Lower values fit clinical, mobile, or low-resource settings. |
| w_k_lambda | {hospital['directions']['lambda']} | {hospital['w_lambda']:.4f} | Controls preference for stronger diagnosis confirmation. Higher values favor histopathology/biopsy-confirmed samples. Lower values fit screening or clinical-only settings. |

Clinical reason for this hospital:

**{hospital['rationale']}**

Generation logic:

1. The dataset mean values of `f_phi`, `f_tau`, and `f_lambda` were calculated.
2. This hospital was given direction tokens: `high`, `low`, or `neutral`.
3. Direction `high` pushes the value above the dataset mean.
4. Direction `low` pushes the value below the dataset mean.
5. Direction `neutral` keeps the value close to the dataset average.
6. A small random jitter was added so that hospitals do not become identical.
"""


def write_hospital_description_md(hospital: dict,
                                  df_hosp: pd.DataFrame,
                                  cols: dict,
                                  copied: int,
                                  missing: int,
                                  output_dir: Path = OUTPUT_DIR) -> Path:
    out = output_dir / hospital["name"] / "hospital_description.md"

    fitz_md    = _value_counts_md(df_hosp[cols["fitzpatrick"]]) if cols["fitzpatrick"] else "  - (column missing)\n"
    img_md     = _value_counts_md(df_hosp[cols["image_type"]])  if cols["image_type"]  else "  - (column missing)\n"
    confirm_md = _value_counts_md(df_hosp[cols["confirm"]])     if cols["confirm"]     else "  - (column missing)\n"

    benign_malignant_section = _benign_malignant_md(df_hosp)
    wk_reason_section = _wk_reason_md(hospital)

    # ── NEW: target and actual malignant % ─────────────────────────────────
    target_pct = hospital.get("target_malignant_pct", "N/A")
    actual_pct = round(_compute_actual_malignant_pct(df_hosp), 2)
    if isinstance(target_pct, float):
        target_pct_str = f"{target_pct:.2f}%"
    else:
        target_pct_str = str(target_pct)
    # ───────────────────────────────────────────────────────────────────────

    md = f"""# {hospital['name']}

## 1. Hospital Name
**{hospital['name']}**

## 2. Hospital Type
{hospital['type']}

## 3. Clinical Role
{hospital['clinical_role']}

## 4. Expected Metadata Preference
{hospital['expected_pattern']}

Direction tokens: phi=`{hospital['directions']['phi']}`, tau=`{hospital['directions']['tau']}`, lambda=`{hospital['directions']['lambda']}`.

## 5. Final Generated w_k Vector
| Component | Value |
|-----------|-------|
| w_k_phi    | {hospital['w_phi']:.4f} |
| w_k_tau    | {hospital['w_tau']:.4f} |
| w_k_lambda | {hospital['w_lambda']:.4f} |

## 5.1 Why These w_k Values Were Selected
{wk_reason_section}

## 6. Explanation of w_k_phi (Fitzpatrick preference)
Direction `{hospital['directions']['phi']}`: a higher value means this hospital prefers samples with
higher Fitzpatrick rank (darker skin types). The numeric value was derived from the dataset
mean Fitzpatrick score plus a direction-specific offset and small jitter for heterogeneity.

## 7. Explanation of w_k_tau (Image type preference)
Direction `{hospital['directions']['tau']}`: a higher value means this hospital prefers higher-quality
imaging (e.g., dermoscopic). Lower values reflect clinical/mobile-image-heavy facilities.

## 8. Explanation of w_k_lambda (Confirmation type preference)
Direction `{hospital['directions']['lambda']}`: a higher value means stronger preference for high-confidence
diagnostic confirmation (Tier 1 = histopathology). Lower values reflect screening / clinical-only contexts.

## 9. Number of Assigned Samples
**{len(df_hosp)}** metadata rows assigned.

## 10. Image Count
Successfully copied images: **{copied}**

## 11. Missing Image Count
Missing images at this hospital: **{missing}**

## 11.1 Benign / Malignant Distribution
{benign_malignant_section}

## 11.2 Malignant Percentage Summary
| Metric | Value |
|--------|-------|
| Target Malignant % | {target_pct_str} |
| Actual Malignant % | {actual_pct:.2f}% |

## 12. Fitzpatrick Skin Type Distribution
{fitz_md}

## 13. Image Type Distribution
{img_md}

## 14. Diagnosis Confirmation Type Distribution
{confirm_md}

## 15. Why This Hospital Received These Samples
{hospital['rationale']}

The probabilistic softmax assignment (with tau_soft = {TAU_SOFT}) prefers samples whose
feature vector f(m_i) aligns with this hospital's w_k vector, but allows controlled randomness
so that not every sample goes to the single best-matching hospital. This produces a realistic
Non-IID partition mirroring true federated hospital data.

w_k generation note: {hospital['wk_explanation']}
"""
    out.write_text(md, encoding="utf-8")
    return out


def write_process_explanation_md(hospitals: list[dict],
                                 cols: dict,
                                 distributions: dict,
                                 output_dir: Path = OUTPUT_DIR) -> Path:
    out = output_dir / "process_explanation.md"

    # Hospital preference matrix W
    rows = ["| Hospital | Type | w_phi | w_tau | w_lambda |",
            "|----------|------|-------|-------|----------|"]
    for h in hospitals:
        rows.append(f"| {h['name']} | {h['type']} | {h['w_phi']:.4f} | {h['w_tau']:.4f} | {h['w_lambda']:.4f} |")
    W_table = "\n".join(rows)

    # ── NEW: malignant-range summary table ─────────────────────────────────
    mal_rows = ["| Hospital | Target Malignant % | Actual Malignant % |",
                "|----------|-------------------|-------------------|"]
    for h in hospitals:
        t = h.get("target_malignant_pct", "N/A")
        a = h.get("actual_malignant_pct_after", "N/A")
        t_str = f"{t:.2f}%" if isinstance(t, float) else str(t)
        a_str = f"{a:.2f}%" if isinstance(a, float) else str(a)
        mal_rows.append(f"| {h['name']} | {t_str} | {a_str} |")
    mal_table = "\n".join(mal_rows)
    # ───────────────────────────────────────────────────────────────────────

    md = f"""# Process Explanation — Non-IID Federated Hospital Partitioning

## 1. What Non-IID Partitioning Means
Non-IID (non-independent and identically distributed) partitioning means each hospital
receives a *different* statistical distribution of data — mirroring real-world federated
medical data, where hospitals differ in patient populations, equipment, and confirmation methods.

## 2. Why Hospital-Wise Division Is Useful in Federated Learning
Federated learning trains a global model across multiple silos *without* sharing raw data.
Realistic Non-IID hospital partitions help us evaluate algorithms (FedAvg, FedProx, etc.)
under heterogeneity that resembles real clinical deployment.

## 3. Metadata Columns Used
- Fitzpatrick column: `{cols['fitzpatrick']}`
- Image type column: `{cols['image_type']}`
- Confirmation type column: `{cols['confirm']}`
- Image ID column: `{cols['id']}`

## 4. What f(m_i) Means
For each sample i with metadata m_i:

f(m_i) = [f_phi, f_tau, f_lambda]

Only three features are used. f(m_i) is used **only** for hospital assignment, not as a
training feature (unless you later choose to).

## 5. Formula for f_phi (Fitzpatrick)
Let x ∈ {{1..6}} be the Fitzpatrick rank.

  **f_phi = x / 6**

I=1/6, II=2/6, III=3/6, IV=4/6, V=5/6, VI=6/6. Missing → 0.5.

## 6. Formula for f_lambda (Confirmation)
With T(c) ∈ {{1,2,3,4}} (1 = strongest):

  **f_lambda = 1 − (T(c) − 1) / (T_max − 1)**, T_max = 4

Tier 1 (histopath/biopsy/excision/pathology) = 1.0
Tier 2 (confocal/expert consensus)            = 0.6667
Tier 3 (serial imaging/follow-up/clinical)    = 0.3333
Tier 4 (single contributor/unknown)           ≈ 0.1 (small baseline). Missing → 0.5.

## 7. General Scoring Rule for f_tau (Image type)
- dermoscopic / dermoscopy  → high (≈ 0.9)
- clinical close-up         → medium-high (≈ 0.7)
- clinical / overview       → medium-low (≈ 0.4)
- mobile / smartphone       → low (≈ 0.3)
- missing / unknown         → 0.5

## 8. What w_k Means
w_k = [w_k_phi, w_k_tau, w_k_lambda] is each hospital's preference vector,
indicating how strongly it prefers samples scoring high in each feature dimension.

## 9. How w_k Was Generated
1. Compute dataset means of f_phi, f_tau, f_lambda.
2. Assign each hospital archetype a *direction* per feature: high / low / neutral.
3. Map each direction + dataset mean to a base value, e.g.:
   - high   → ≈ 0.6 + 0.5 · mean (clipped to [0.75, 0.95])
   - low    → ≈ 0.4 − 0.5 · mean (clipped to [0.05, 0.30])
   - neutral→ ≈ 0.4 + 0.2 · mean (near 0.5)
4. Add small jitter (±0.05) for heterogeneity.
5. Optionally nudge values using blueprint text cues
   (e.g., "dermoscopy", "diversity", "histopathology").

Dataset means observed:
- mean f_phi    = {distributions.get('mean_f_phi', float('nan')):.4f}
- mean f_tau    = {distributions.get('mean_f_tau', float('nan')):.4f}
- mean f_lambda = {distributions.get('mean_f_lambda', float('nan')):.4f}

## 10. Hospital Preference Matrix W
{W_table}

## 11. Formula for score_ik
  **score_ik = w_k_phi · f_phi + w_k_tau · f_tau + w_k_lambda · f_lambda**

## 12. Softmax Probability
  **P(k | m_i) = exp(score_ik / tau_soft) / Σ_j exp(score_ij / tau_soft)**

## 13. Meaning of tau_soft
tau_soft = {TAU_SOFT}. Smaller → sharper, more Non-IID. Larger → softer, more IID-like.

## 14. Why Probabilistic Assignment (not argmax)
Pure argmax would deterministically funnel every matching sample to one hospital and
collapse natural variation. Probabilistic sampling preserves realistic spread while
still respecting hospital preferences.

## 15. How Images Were Copied
We use the detected ID column (`{cols['id']}`) and try common extensions
(.jpg, .jpeg, .png) inside the input directory. Files are **copied** (not moved) — originals
remain untouched.

## 16. How Each Hospital metadata.csv Was Created
After assignment, the master DataFrame is filtered by `assigned_hospital`. Each subset
(including `f_phi`, `f_tau`, `f_lambda`, `f_mi_vector`, `assigned_hospital_id`,
`assigned_hospital`) is written to that hospital's `metadata.csv`.

## 17. Malignant-Range Adjustment
After the original softmax assignment, a post-processing step redistributes malignant
and benign samples between hospitals to create a clinically realistic heterogeneous
malignant-case distribution (approximately 5 % – 70 % range).

The original f(m_i), w_k, and softmax logic are NOT changed.
Only `assigned_hospital` and `assigned_hospital_id` are updated for affected rows.

{mal_table}

## 18. How This Output Can Be Used in Federated Learning
- Treat each hospital folder as one federated client.
- Train a local model per client on its `images/` + `metadata.csv`.
- Aggregate weights with FedAvg/FedProx, evaluate global model on held-out data.
- Because partitions are Non-IID (both feature-wise and label-distribution-wise),
  this benchmarks realistic FL behaviour under label imbalance.

Random seed used: {RANDOM_SEED}.
"""
    out.write_text(md, encoding="utf-8")
    print(f"[markdown_writer] process_explanation.md -> {out}")
    return out
