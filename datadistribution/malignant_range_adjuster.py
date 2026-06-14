"""
malignant_range_adjuster.py
===========================
Post-assignment adjuster that redistributes malignant / benign samples
across hospitals so that hospital-wise malignant percentages span a wide
range (approximately 5 % – 70 %) instead of the narrow default range.

CALL SITE  (in main.py, after assign_hospitals() and before output writing):
    from malignant_range_adjuster import adjust_malignant_range
    if ENABLE_MALIGNANT_RANGE_ADJUSTMENT:
        df = adjust_malignant_range(df, hospitals, OUTPUT_DIR)

Nothing else in the existing pipeline needs to know about this file.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

# ── Config import (only the flag + target dict, both may be absent) ────────
try:
    from config import (
        ENABLE_MALIGNANT_RANGE_ADJUSTMENT,
        TARGET_MALIGNANT_PERCENTAGES,
        MALIGNANT_ADJUSTMENT_STRENGTH,
        OUTPUT_DIR,
    )
except ImportError:
    ENABLE_MALIGNANT_RANGE_ADJUSTMENT = True
    TARGET_MALIGNANT_PERCENTAGES      = {}
    MALIGNANT_ADJUSTMENT_STRENGTH     = 1.0
    from config import OUTPUT_DIR


# ══════════════════════════════════════════════════════════════════════════════
# 1.  LABEL DETECTION
# ══════════════════════════════════════════════════════════════════════════════

_MALIGNANT_KW = [
    "malignant", "melanoma", "carcinoma",
    "basal cell", "squamous cell",
]
_BENIGN_KW = [
    "benign", "nevus", "naevus",
    "seborrheic keratosis", "dermatofibroma",
    "angioma", "lentigo",
]

# Columns tried in priority order
_DIAG_COLS_PRIORITY = [
    "benign_malignant",          # direct binary label (best)
    "diagnosis_1",
    "diagnosis",
    "diagnosis_2",
    "diagnosis_3",
    "diagnosis_4",
    "diagnosis_5",
    "target",
    "label",
    "class",
]


def _find_label_cols(df: pd.DataFrame) -> list[str]:
    lower_map = {c.lower(): c for c in df.columns}
    found = []
    for cand in _DIAG_COLS_PRIORITY:
        if cand.lower() in lower_map:
            found.append(lower_map[cand.lower()])
    return found


def _classify_row(row: pd.Series, label_cols: list[str]) -> str:
    """
    Returns 'malignant', 'benign', or 'indeterminate'.

    Priority:
      1. If benign_malignant / diagnosis_1 column exists, parse it directly.
      2. Fall back to keyword scan across remaining diag columns.
    """
    for col in label_cols:
        val = row.get(col, None)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        s = str(val).strip().lower()
        if s in ("", "nan", "none"):
            continue

        # Direct binary labels
        if s == "malignant":
            return "malignant"
        if s == "benign":
            return "benign"
        if s == "indeterminate":
            return "indeterminate"

        # Numeric target (ISIC convention: 1 = malignant, 0 = benign)
        if col.lower() in ("target", "label"):
            try:
                n = int(float(s))
                if n == 1:
                    return "malignant"
                if n == 0:
                    return "benign"
            except ValueError:
                pass

        # Keyword scan
        if any(k in s for k in _MALIGNANT_KW):
            return "malignant"
        if any(k in s for k in _BENIGN_KW):
            return "benign"

    return "indeterminate"


def add_label_column(df: pd.DataFrame) -> pd.DataFrame:
    """Adds '_adj_label' column: 'malignant' | 'benign' | 'indeterminate'."""
    label_cols = _find_label_cols(df)
    if not label_cols:
        print("[adjuster] WARNING: no diagnosis column found – all samples treated as indeterminate.")
        df = df.copy()
        df["_adj_label"] = "indeterminate"
        return df

    df = df.copy()
    df["_adj_label"] = df.apply(lambda r: _classify_row(r, label_cols), axis=1)
    counts = df["_adj_label"].value_counts().to_dict()
    print(f"[adjuster] Label counts  →  {counts}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2.  TARGET PERCENTAGE MAPPING
# ══════════════════════════════════════════════════════════════════════════════

# Hospital-type → malignant target (%).
# Lower-risk / screening hospitals get small targets;
# specialist / oncology hospitals get large targets.
_TYPE_TARGET_MAP: dict[str, float] = {
    # ---- low-risk (screening / outreach / tele) ----
    "public health screening camp":       5.0,
    "teledermatology screening network":  8.0,
    "community outreach clinic":         12.0,
    "rural community health center":     18.0,

    # ---- mid-risk (general / regional) ----
    "general multispecialty hospital":   25.0,
    "regional mixed hospital":           35.0,
    "urban dermatology research clinic": 45.0,
    "academic dermoscopy unit":          55.0,

    # ---- high-risk (oncology / biopsy / specialist) ----
    "oncology biopsy center":            65.0,
    "tertiary melanoma institute":       70.0,
}

# Named-hospital overrides (use exact names from hospital_profiles.py)
_NAME_TARGET_MAP: dict[str, float] = {
    "GoldenSun_Public_Health_Screening_Camp":       5.0,
    "TeleDerm_National_Screening_Network":          8.0,
    "SunRay_Community_Outreach_Clinic":            12.0,
    "Greenfield_Rural_Community_Health_Center":    18.0,
    "CityGeneral_Multispecialty_Hospital":         25.0,
    "Lakeside_Regional_Mixed_Hospital":            35.0,
    "MetroSkin_Dermatology_Research_Clinic":       45.0,
    "Westbridge_Academic_Dermoscopy_Unit":         55.0,
    "Pinewood_Oncology_Biopsy_Center":             65.0,
    "NorthCare_Tertiary_Melanoma_Institute":       70.0,
}


def _resolve_targets(hospitals: list[dict]) -> dict[str, float]:
    """
    Build {hospital_name: target_%} by merging (in priority order):
      1. config.TARGET_MALIGNANT_PERCENTAGES  (user-defined, highest priority)
      2. _NAME_TARGET_MAP                     (hard-coded named overrides)
      3. _TYPE_TARGET_MAP                     (type-based fallback)
      4. 30 %                                 (global default)
    """
    result: dict[str, float] = {}
    for h in hospitals:
        name = h["name"]
        htype = h.get("type", "").lower().strip()

        # Priority 1: config override
        if name in TARGET_MALIGNANT_PERCENTAGES:
            result[name] = float(TARGET_MALIGNANT_PERCENTAGES[name])
            continue

        # Priority 2: named map
        if name in _NAME_TARGET_MAP:
            result[name] = _NAME_TARGET_MAP[name]
            continue

        # Priority 3: type map
        matched = False
        for type_key, pct in _TYPE_TARGET_MAP.items():
            if type_key in htype:
                result[name] = pct
                matched = True
                break

        if not matched:
            result[name] = 30.0   # safe default

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 3.  BEFORE-STATE SNAPSHOT
# ══════════════════════════════════════════════════════════════════════════════

def _snapshot(df: pd.DataFrame, hospitals: list[dict]) -> dict[str, dict]:
    snap: dict[str, dict] = {}
    for h in hospitals:
        sub = df[df["assigned_hospital"] == h["name"]]
        lbl = sub["_adj_label"]
        b = int((lbl == "benign").sum())
        m = int((lbl == "malignant").sum())
        i = int((lbl == "indeterminate").sum())
        total = len(sub)
        snap[h["name"]] = {
            "benign": b,
            "malignant": m,
            "indeterminate": i,
            "total": total,
            "mal_pct": (m / total * 100) if total else 0.0,
        }
    return snap


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CORE ADJUSTMENT LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def _do_adjustment(
    df: pd.DataFrame,
    hospitals: list[dict],
    targets: dict[str, float],
    strength: float,
) -> pd.DataFrame:
    """
    Iteratively move malignant samples from hospitals above their target
    to hospitals below their target (and mirror benign moves in the opposite
    direction to keep hospital sizes stable).

    Steps
    -----
    1. Build pools: for each hospital collect indices by label.
    2. Sort hospitals: lowest-target first.
    3. For each hospital that is *under* its target, request malignant
       samples from hospitals that are *over* their target.
    4. Swap: give one malignant from donor, take one benign back → both
       hospitals shift toward their targets simultaneously.
    5. Repeat until no more swaps are possible or all hospitals are close
       enough to their targets.

    The `strength` parameter (0–1) scales how many moves are attempted.
    strength = 1.0 → try to reach the exact target.
    strength < 1.0 → only partially adjust (preserves more of the
                     original softmax assignment).
    """
    df = df.copy()

    # Build mutable pools: {hospital_name: {'malignant': [idx,...], 'benign': [...], 'indeterminate': [...]}}
    pools: dict[str, dict[str, list[int]]] = {}
    for h in hospitals:
        name = h["name"]
        sub = df[df["assigned_hospital"] == name]
        pools[name] = {
            "malignant":     list(sub[sub["_adj_label"] == "malignant"].index),
            "benign":        list(sub[sub["_adj_label"] == "benign"].index),
            "indeterminate": list(sub[sub["_adj_label"] == "indeterminate"].index),
        }

    def _mal_pct(name: str) -> float:
        m = len(pools[name]["malignant"])
        total = m + len(pools[name]["benign"]) + len(pools[name]["indeterminate"])
        return (m / total * 100) if total else 0.0

    def _need_malignant(name: str) -> int:
        """How many malignant samples we still need to add."""
        pct = _mal_pct(name)
        target = targets[name]
        if pct >= target:
            return 0
        total = (len(pools[name]["malignant"])
                 + len(pools[name]["benign"])
                 + len(pools[name]["indeterminate"]))
        needed_abs = int(round(target / 100.0 * total)) - len(pools[name]["malignant"])
        return max(0, int(needed_abs * strength))

    def _can_give_malignant(name: str) -> int:
        """How many malignant samples this hospital can donate."""
        pct = _mal_pct(name)
        target = targets[name]
        if pct <= target:
            return 0
        total = (len(pools[name]["malignant"])
                 + len(pools[name]["benign"])
                 + len(pools[name]["indeterminate"]))
        excess_abs = len(pools[name]["malignant"]) - int(round(target / 100.0 * total))
        return max(0, int(excess_abs * strength))

    # Sort by target ascending so we serve the most needy first
    hospital_names_asc = sorted(hospitals, key=lambda h: targets[h["name"]])

    max_passes = 20
    total_swaps = 0

    for _pass in range(max_passes):
        swaps_this_pass = 0

        for receiver in hospital_names_asc:
            recv_name = receiver["name"]
            needed = _need_malignant(recv_name)
            if needed == 0:
                continue

            for donor in reversed(hospital_names_asc):   # highest-target donors first
                don_name = donor["name"]
                if don_name == recv_name:
                    continue
                can_give = _can_give_malignant(don_name)
                if can_give == 0:
                    continue

                to_move = min(needed, can_give, len(pools[don_name]["malignant"]))
                if to_move == 0:
                    continue

                # ---- SWAP LOOP ----
                # Each iteration: move 1 malignant donor→receiver, 1 benign receiver→donor
                for _ in range(to_move):
                    if not pools[don_name]["malignant"]:
                        break
                    if not pools[recv_name]["benign"]:
                        # No benign to give back → one-way move only if sizes allow
                        # (try indeterminate as substitute)
                        if not pools[recv_name]["indeterminate"]:
                            break   # can't balance
                        # Move malignant to receiver, move indeterminate to donor
                        mal_idx = pools[don_name]["malignant"].pop()
                        ind_idx = pools[recv_name]["indeterminate"].pop()
                        df.at[mal_idx, "assigned_hospital"]    = recv_name
                        df.at[mal_idx, "assigned_hospital_id"] = receiver["id"]
                        pools[recv_name]["malignant"].append(mal_idx)

                        df.at[ind_idx, "assigned_hospital"]    = don_name
                        df.at[ind_idx, "assigned_hospital_id"] = donor["id"]
                        pools[don_name]["indeterminate"].append(ind_idx)
                        swaps_this_pass += 1
                        total_swaps += 1
                    else:
                        # Ideal swap: malignant donor→receiver, benign receiver→donor
                        mal_idx = pools[don_name]["malignant"].pop()
                        ben_idx = pools[recv_name]["benign"].pop()
                        df.at[mal_idx, "assigned_hospital"]    = recv_name
                        df.at[mal_idx, "assigned_hospital_id"] = receiver["id"]
                        pools[recv_name]["malignant"].append(mal_idx)

                        df.at[ben_idx, "assigned_hospital"]    = don_name
                        df.at[ben_idx, "assigned_hospital_id"] = donor["id"]
                        pools[don_name]["benign"].append(ben_idx)
                        swaps_this_pass += 1
                        total_swaps += 1

                # recompute need after moves
                needed = _need_malignant(recv_name)
                if needed == 0:
                    break

        if swaps_this_pass == 0:
            break

    print(f"[adjuster] Adjustment complete. Total swaps: {total_swaps}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5.  SUMMARY CSV
# ══════════════════════════════════════════════════════════════════════════════

def _save_adjustment_summary(
    hospitals: list[dict],
    targets: dict[str, float],
    before: dict[str, dict],
    after: dict[str, dict],
    output_dir: Path,
) -> Path:
    rows = []
    for h in hospitals:
        name = h["name"]
        b_snap = before[name]
        a_snap = after[name]
        rows.append({
            "hospital_name":                       name,
            "hospital_type":                       h.get("type", ""),
            "target_malignant_percentage":         round(targets[name], 2),
            "actual_malignant_percentage_before":  round(b_snap["mal_pct"], 2),
            "actual_malignant_percentage_after":   round(a_snap["mal_pct"], 2),
            "benign_count_before":                 b_snap["benign"],
            "malignant_count_before":              b_snap["malignant"],
            "indeterminate_count_before":          b_snap["indeterminate"],
            "benign_count_after":                  a_snap["benign"],
            "malignant_count_after":               a_snap["malignant"],
            "indeterminate_count_after":           a_snap["indeterminate"],
            "total_samples_before":                b_snap["total"],
            "total_samples_after":                 a_snap["total"],
            "difference_from_target_after":        round(a_snap["mal_pct"] - targets[name], 2),
        })
    out_path = output_dir / "malignant_range_adjustment_summary.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[adjuster] Adjustment summary → {out_path}")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# 6.  TERMINAL TABLE
# ══════════════════════════════════════════════════════════════════════════════

def _print_table(
    hospitals: list[dict],
    targets: dict[str, float],
    before: dict[str, dict],
    after: dict[str, dict],
) -> None:
    col_w = 45
    print("\n" + "=" * 130)
    print("MALIGNANT RANGE ADJUSTMENT RESULTS")
    print("=" * 130)
    header = (
        f"{'Hospital':<{col_w}} {'Target%':>8} {'Before%':>8} {'After%':>8}"
        f"  {'Benign':>7} {'Malig':>7} {'Indet':>7} {'Total':>7}"
    )
    print(header)
    print("-" * 130)
    for h in hospitals:
        name = h["name"]
        a = after[name]
        print(
            f"{name:<{col_w}} "
            f"{targets[name]:>8.2f} "
            f"{before[name]['mal_pct']:>8.2f} "
            f"{a['mal_pct']:>8.2f}  "
            f"{a['benign']:>7} "
            f"{a['malignant']:>7} "
            f"{a['indeterminate']:>7} "
            f"{a['total']:>7}"
        )
    print("-" * 130)

    all_before = [before[h["name"]]["mal_pct"] for h in hospitals]
    all_after  = [after[h["name"]]["mal_pct"]  for h in hospitals]
    print(f"\n  Before → min: {min(all_before):.2f}%  max: {max(all_before):.2f}%")
    print(f"  After  → min: {min(all_after):.2f}%   max: {max(all_after):.2f}%")

    if min(all_after) > 6.0 or max(all_after) < 65.0:
        print(
            "\n  NOTE: Exact 5% – 70% range could not be fully achieved.\n"
            "  Reason: the dataset may not contain enough malignant samples\n"
            "  to reach 70% in any single hospital, or not enough benign\n"
            "  samples to dilute a hospital to 5%. The closest possible\n"
            "  result was generated."
        )
    print("=" * 130 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def adjust_malignant_range(
    df: pd.DataFrame,
    hospitals: list[dict],
    output_dir: Path,
) -> pd.DataFrame:
    """
    Main entry point called from main.py.

    Parameters
    ----------
    df         : DataFrame after assign_hospitals()
    hospitals  : list of hospital dicts (same object used everywhere)
    output_dir : root output directory (for saving the summary CSV)

    Returns
    -------
    Adjusted DataFrame with the same schema as the input.
    The only changed columns are 'assigned_hospital' and 'assigned_hospital_id'.
    A temporary '_adj_label' helper column is removed before returning.
    """
    print("\n[adjuster] ── Starting malignant-range adjustment ──")

    # 1. Add label column
    df = add_label_column(df)

    # 2. Resolve targets
    targets = _resolve_targets(hospitals)
    print("[adjuster] Resolved targets:")
    for name, pct in targets.items():
        print(f"  {name:<50} → {pct:.1f}%")

    # 3. Snapshot before
    before = _snapshot(df, hospitals)

    # 4. Adjust
    strength = float(MALIGNANT_ADJUSTMENT_STRENGTH)
    df = _do_adjustment(df, hospitals, targets, strength)

    # 5. Snapshot after
    after = _snapshot(df, hospitals)

    # 6. Print table
    _print_table(hospitals, targets, before, after)

    # 7. Save summary CSV
    _save_adjustment_summary(hospitals, targets, before, after, output_dir)

    # 8. Clean up helper column
    df = df.drop(columns=["_adj_label"], errors="ignore")

    print("[adjuster] ── Adjustment done ──\n")
    return df
