"""
feature_scoring.py
==================
Implements f(m_i) = [f_phi, f_tau, f_lambda] using ONLY:
- fitzpatrick_skin_type
- image_type
- diagnosis_confirm_type
"""

import re
import pandas as pd

# ---- Fitzpatrick mapping (Roman numeral / number / "type X") ----
FITZ_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6}


def parse_fitzpatrick(value) -> int | None:
    """Return integer rank 1..6 or None if unknown/missing."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip().lower()
    if s in ("", "nan", "none", "unknown", "na"):
        return None

    # numeric like "3" or "3.0"
    try:
        n = int(float(s))
        if 1 <= n <= 6:
            return n
    except ValueError:
        pass

    # remove "type", "fitzpatrick", spaces
    s_clean = re.sub(r"(type|fitzpatrick|skin|\s+)", "", s)
    if s_clean in FITZ_ROMAN:
        return FITZ_ROMAN[s_clean]

    # last resort: search roman tokens
    for token in ("vi", "v", "iv", "iii", "ii", "i"):
        if re.search(rf"\b{token}\b", s):
            return FITZ_ROMAN[token]
    return None


def score_fitzpatrick(value) -> float:
    """f_phi = x / 6 ; missing -> 0.5 neutral."""
    rank = parse_fitzpatrick(value)
    if rank is None:
        return 0.5
    return rank / 6.0


# ---- Confirmation tier mapping ----
TIER1_KEYS = ["histopathology", "histo", "biopsy", "excision", "pathology"]
TIER2_KEYS = ["confocal", "consensus", "expert consensus"]
TIER3_KEYS = ["serial imaging", "no change", "follow", "clinical"]
TIER4_KEYS = ["single contributor", "single image", "unknown", ""]

T_MAX = 4


def _tier_for_confirm(value: str) -> int:
    s = str(value).strip().lower() if value is not None else ""
    if s in ("nan", "none", "na"):
        return 4
    if any(k in s for k in TIER1_KEYS):
        return 1
    if any(k in s for k in TIER2_KEYS):
        return 2
    if any(k in s for k in TIER3_KEYS):
        return 3
    return 4


def score_confirmation_type(value) -> float:
    """f_lambda = 1 - (T - 1)/(T_max - 1) ; missing -> 0.5 neutral."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.5
    s = str(value).strip()
    if s.lower() in ("", "nan", "none", "unknown", "na"):
        return 0.5
    t = _tier_for_confirm(s)
    score = 1.0 - (t - 1) / (T_MAX - 1)
    if score <= 0.0:
        score = 0.1  # small baseline so Tier 4 isn't a perfect zero
    return float(score)


# ---- Image type tier mapping ----
IMG_TIER1_KEYS = ["dermoscopic", "dermoscopy", "dermoscop"]
IMG_TIER2_KEYS = ["close-up", "closeup", "close"]
IMG_TIER3_KEYS = ["clinical", "overview"]
IMG_TIER4_KEYS = ["mobile", "phone", "smartphone", "unknown", ""]

IMG_T_MAX = 4


def _tier_for_image_type(value: str) -> int:
    s = str(value).strip().lower() if value is not None else ""

    if s in ("", "nan", "none", "unknown", "na"):
        return 4

    if any(k in s for k in IMG_TIER1_KEYS):
        return 1

    if any(k in s for k in IMG_TIER2_KEYS):
        return 2

    if any(k in s for k in IMG_TIER3_KEYS):
        return 3

    if any(k in s for k in IMG_TIER4_KEYS):
        return 4

    return 4


def score_image_type(value, mapping: dict = None) -> float:
    """f_tau = 1 - (T - 1)/(T_max - 1) ; missing -> 0.5 neutral."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.5

    s = str(value).strip()
    if s.lower() in ("", "nan", "none", "unknown", "na"):
        return 0.5

    t = _tier_for_image_type(s)
    score = 1.0 - (t - 1) / (IMG_T_MAX - 1)

    if score <= 0.0:
        score = 0.1  # small baseline so Tier 4 isn't a perfect zero

    return float(score)


def build_sample_vector(row, cols: dict, image_type_map: dict = None) -> tuple[float, float, float]:
    """Return (f_phi, f_tau, f_lambda) for one metadata row."""
    fitz_val    = row[cols["fitzpatrick"]] if cols["fitzpatrick"] else None
    img_val     = row[cols["image_type"]]  if cols["image_type"]  else None
    confirm_val = row[cols["confirm"]]     if cols["confirm"]     else None

    f_phi    = score_fitzpatrick(fitz_val)
    f_tau    = score_image_type(img_val, image_type_map)
    f_lambda = score_confirmation_type(confirm_val)

    return f_phi, f_tau, f_lambda