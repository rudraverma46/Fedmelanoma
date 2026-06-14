"""
wk_generator.py
===============
Generates each hospital's preference vector w_k = [w_phi, w_tau, w_lambda]
using:
  1. dataset distribution of the three features
  2. blueprint logic (textual cues, optional)
  3. hospital archetype direction tokens

Output: list of dicts with numeric w_k in [0, 1] per hospital.
"""

import numpy as np
import pandas as pd


def analyze_metadata_distributions(df: pd.DataFrame, cols: dict, image_type_map: dict) -> dict:
    """Return summary statistics of the three features across the dataset."""
    stats = {}

    # Mean of f_phi (already in df as 'f_phi')
    if "f_phi" in df.columns:
        stats["mean_f_phi"]    = float(df["f_phi"].mean())
        stats["std_f_phi"]     = float(df["f_phi"].std() or 0.0)
    if "f_tau" in df.columns:
        stats["mean_f_tau"]    = float(df["f_tau"].mean())
        stats["std_f_tau"]     = float(df["f_tau"].std() or 0.0)
    if "f_lambda" in df.columns:
        stats["mean_f_lambda"] = float(df["f_lambda"].mean())
        stats["std_f_lambda"]  = float(df["f_lambda"].std() or 0.0)

    if cols["fitzpatrick"]:
        stats["fitz_value_counts"] = df[cols["fitzpatrick"]].astype(str).value_counts().to_dict()
    if cols["image_type"]:
        stats["image_type_value_counts"] = df[cols["image_type"]].astype(str).value_counts().to_dict()
    if cols["confirm"]:
        stats["confirm_value_counts"] = df[cols["confirm"]].astype(str).value_counts().to_dict()

    return stats


def _direction_to_value(direction: str, dataset_mean: float, rng: np.random.Generator) -> float:
    """
    Map a direction token + dataset mean -> scalar in [0,1].

    Logic:
      "high"    -> push above dataset mean (target ~0.85), nudged by mean.
      "low"     -> push below dataset mean (target ~0.20).
      "neutral" -> stay near dataset mean (or 0.5 if missing).
    Then add small jitter to ensure heterogeneity.
    """
    if dataset_mean is None or np.isnan(dataset_mean):
        dataset_mean = 0.5

    if direction == "high":
        base = max(0.75, min(0.95, 0.6 + 0.5 * dataset_mean))
    elif direction == "low":
        base = min(0.30, max(0.05, 0.4 - 0.5 * dataset_mean))
    else:  # neutral
        base = 0.4 + 0.2 * dataset_mean  # near 0.5

    jitter = rng.uniform(-0.05, 0.05)
    return float(np.clip(base + jitter, 0.05, 0.95))


def generate_wk_values(archetypes: list[dict],
                       distributions: dict,
                       blueprint_text: str | None,
                       seed: int) -> list[dict]:
    """
    Produce numeric w_k for each hospital. Returns list aligned with archetypes,
    each item: {'name', 'type', 'w_phi', 'w_tau', 'w_lambda', 'wk_explanation'}.
    """
    rng = np.random.default_rng(seed)

    mean_phi    = distributions.get("mean_f_phi", 0.5)
    mean_tau    = distributions.get("mean_f_tau", 0.5)
    mean_lambda = distributions.get("mean_f_lambda", 0.5)

    # Optional blueprint nudges (text-cue based)
    bp_lower = (blueprint_text or "").lower()
    bp_dermoscopy_emphasis = "dermoscop" in bp_lower
    bp_skin_diversity      = "diversity" in bp_lower or "fitzpatrick" in bp_lower
    bp_pathology_emphasis  = "histopath" in bp_lower or "biopsy" in bp_lower

    out = []
    for arch in archetypes:
        d = arch["directions"]

        w_phi    = _direction_to_value(d["phi"],    mean_phi,    rng)
        w_tau    = _direction_to_value(d["tau"],    mean_tau,    rng)
        w_lambda = _direction_to_value(d["lambda"], mean_lambda, rng)

        # Apply small blueprint nudges
        if bp_dermoscopy_emphasis and d["tau"] == "high":
            w_tau = float(np.clip(w_tau + 0.03, 0.05, 0.95))
        if bp_skin_diversity and d["phi"] == "high":
            w_phi = float(np.clip(w_phi + 0.03, 0.05, 0.95))
        if bp_pathology_emphasis and d["lambda"] == "high":
            w_lambda = float(np.clip(w_lambda + 0.03, 0.05, 0.95))

        explanation = (
            f"Archetype '{arch['type']}' has direction "
            f"phi={d['phi']}, tau={d['tau']}, lambda={d['lambda']}. "
            f"Dataset means (phi={mean_phi:.3f}, tau={mean_tau:.3f}, lambda={mean_lambda:.3f}) "
            f"were used to scale each direction, with small jitter for heterogeneity. "
            f"Blueprint cues applied: dermoscopy={bp_dermoscopy_emphasis}, "
            f"diversity={bp_skin_diversity}, pathology={bp_pathology_emphasis}."
        )

        out.append({
            "id": arch["id"],
            "name": arch["name"],
            "type": arch["type"],
            "clinical_role": arch["clinical_role"],
            "expected_pattern": arch["expected_pattern"],
            "rationale": arch["rationale"],
            "directions": d,
            "w_phi":    w_phi,
            "w_tau":    w_tau,
            "w_lambda": w_lambda,
            "wk_explanation": explanation,
        })

    return out


def validate_wk_values(hospitals: list[dict]) -> None:
    """Sanity-check ranges and ensure heterogeneity."""
    arr = np.array([[h["w_phi"], h["w_tau"], h["w_lambda"]] for h in hospitals])
    assert arr.min() >= 0.0 and arr.max() <= 1.0, "w_k values out of [0,1]!"
    var = arr.var(axis=0).sum()
    if var < 1e-3:
        print("[wk_generator] WARNING: very low heterogeneity across hospitals.")
    else:
        print(f"[wk_generator] w_k validated. Heterogeneity (sum of variances) = {var:.4f}")