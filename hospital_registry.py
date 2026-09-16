"""
hospital_registry.py
=====================
Phase 4 — Adapter: dynamic hospital discovery for the Flower runtime.

Neither server/server.py's server_fn nor client/client.py's client_fn may
hardcode the hospital list (train.py and evaluate.py may -- they are
frozen, already-verified centralized scripts; this module exists
specifically so the *new* Flower runtime code does not need to). This
module enumerates hospital_partitions/ at call time and, per hospital,
derives exactly the inputs server/strategy.py's HospitalConfig needs:

    n_samples          -- row count of the hospital's local training CSV
    fitzpatrick_dist   -- P(skin_type)   over client.dataset's 6 categories
    modality_dist      -- P(image_type)  over client.dataset's 3 categories
    confirmation_dist  -- P(confirm_type) over client.dataset's 4 categories
    tier / latency_*   -- straggler-simulation parameters (see note below)

WHY distributions are computed from metadata_train.csv directly, not from
the documented *_distribution.csv files:
  The expected per-hospital file set (see task spec / blueprint) includes
  fitzpatrick_skin_type_distribution.csv and
  diagnosis_confirm_type_distribution.csv, but NO equivalent file for
  imaging modality. Since modality must be derived from metadata.csv
  regardless, all three axes are derived the same way for consistency, and
  by reusing MelanomaDataset's own column names and category orderings
  (imported, not duplicated) this stays byte-for-byte aligned with how
  client/dataset.py encodes the same columns for the model's Metadata MLP.

WHY tier / latency are assigned here rather than read from the dataset:
  Straggler tier and compute latency (blueprint Section 4.2.2) are a
  network/compute-simulation property, not a patient-data attribute --
  no file in the documented dataset layout encodes it. This module assigns
  it deterministically (round-robin over the blueprint's three canonical
  tiers, in sorted hospital-name order) so registry construction is
  reproducible without inventing per-hospital hardware profiles. This is a
  documented runtime assumption, not a research claim.

This module has ONLY read-only, best-effort filesystem dependencies. If
hospital_partitions/ does not exist, discover_hospitals() and
build_hospital_registry() return empty results rather than raising --
callers (server_fn / client_fn) decide whether an empty registry is fatal.
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

# Read-only import: reuses MelanomaDataset's own category lists/column
# names instead of duplicating them, so this module can never drift out of
# sync with how client/dataset.py encodes the same three metadata axes.
from client.dataset import MelanomaDataset

DEFAULT_PARTITIONS_ROOT: str = "hospital_partitions"

# Canonical straggler tiers + latency parameters, Blueprint Section 4.2.2.
# Cycled deterministically across discovered hospitals (see module
# docstring) -- not derived from any dataset file.
_TIER_CYCLE: Tuple[str, ...] = ("reliable", "moderate", "high_straggler")
_TIER_LATENCY: Dict[str, Tuple[float, float]] = {
    "reliable":       (1.39, 0.30),
    "moderate":       (2.77, 0.50),
    "high_straggler": (4.09, 0.80),
}


# ===========================================================================
# 1. Hospital discovery
# ===========================================================================

def discover_hospitals(partitions_root: str = DEFAULT_PARTITIONS_ROOT) -> List[str]:
    """
    Enumerate hospital partition folders under *partitions_root*.

    Returns a sorted list of directory names (sorting makes discovery
    deterministic across the server process and every client process, so
    a given hospital consistently maps to the same integer index/partition
    id everywhere). Returns [] if the directory does not exist -- this is
    the expected state before the dataset is added, not an error.
    """
    if not os.path.isdir(partitions_root):
        return []
    return sorted(
        name
        for name in os.listdir(partitions_root)
        if os.path.isdir(os.path.join(partitions_root, name))
    )


# ===========================================================================
# 2. Per-hospital distribution extraction
# ===========================================================================

def _category_distribution(series: pd.Series, categories: Sequence[str]) -> np.ndarray:
    """Normalised histogram of *series* over the fixed *categories* order."""
    counts = series.astype(str).str.strip().value_counts()
    raw = np.array([float(counts.get(c, 0)) for c in categories], dtype=float)
    total = raw.sum()
    if total <= 0:
        # No rows matched any known category -- fall back to uniform rather
        # than dividing by zero; this hospital contributes no diversity
        # signal beyond "unknown", which max-normalisation handles safely.
        return np.full(len(categories), 1.0 / len(categories))
    return raw / total


def _hospital_metadata_path(hospital: str, partitions_root: str) -> str:
    """Prefer the training split; fall back to the combined metadata.csv."""
    base = os.path.join(partitions_root, hospital)
    for name in ("metadata_train.csv", "metadata.csv"):
        candidate = os.path.join(base, name)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"No metadata_train.csv or metadata.csv found for hospital "
        f"'{hospital}' under {base}"
    )


def build_hospital_record(
    hospital: str,
    index: int,
    partitions_root: str = DEFAULT_PARTITIONS_ROOT,
) -> Dict[str, object]:
    """
    Build one hospital's registry record: everything
    server.strategy.HospitalConfig needs, as a plain dict (kept
    dependency-free from server/ so this module stays importable, and
    cheap to import, from client/client.py as well).

    Returns a dict with keys: n_samples, tier, latency_mu, latency_sigma,
    fitzpatrick_dist, modality_dist, confirmation_dist (the last three are
    plain Python lists of float, summing to 1.0).
    """
    df = pd.read_csv(_hospital_metadata_path(hospital, partitions_root))

    fitz = _category_distribution(df["fitzpatrick_skin_type"], MelanomaDataset._SKIN_TYPES)
    modality = _category_distribution(df["image_type"], MelanomaDataset._IMAGE_TYPES)
    confirm = _category_distribution(df["diagnosis_confirm_type"], MelanomaDataset._CONFIRM_TYPES)

    tier = _TIER_CYCLE[index % len(_TIER_CYCLE)]
    latency_mu, latency_sigma = _TIER_LATENCY[tier]

    return {
        "n_samples": int(len(df)),
        "tier": tier,
        "latency_mu": latency_mu,
        "latency_sigma": latency_sigma,
        "fitzpatrick_dist": fitz.tolist(),
        "modality_dist": modality.tolist(),
        "confirmation_dist": confirm.tolist(),
    }


# ===========================================================================
# 3. Full registry (all hospitals + global reference distributions)
# ===========================================================================

def build_hospital_registry(
    partitions_root: str = DEFAULT_PARTITIONS_ROOT,
) -> Tuple[List[str], List[Dict[str, object]], List[float], List[float], List[float]]:
    """
    Discover hospitals and build every input server.strategy.FedMelanomaStrategy
    needs at construction time.

    Returns
    -------
    (hospital_names, hospital_records, global_fitzpatrick, global_modality, global_confirmation)

    hospital_records[i] corresponds to hospital_names[i]; that same index
    is the selector.py client_id FedMelanomaStrategy assigns it (see
    strategy.py's HospitalConfig enumeration), which client_fn's
    partition-id resolution (client/client.py) relies on -- both sides
    call discover_hospitals() against the same root, so the mapping is
    guaranteed consistent without any coordination beyond sorted order.

    global_* are the n_samples-weighted average of each hospital's local
    distribution -- the reference centroid emd.py's EarthMoverDistance
    compares each hospital against (blueprint Phase 6: "global centroid").

    Returns ([], [], [], [], []) when hospital_partitions/ is absent or
    empty -- callers decide whether that is fatal for their purpose.
    """
    hospitals = discover_hospitals(partitions_root)
    if not hospitals:
        return [], [], [], [], []

    records: List[Dict[str, object]] = []
    fitz_rows: List[np.ndarray] = []
    mod_rows: List[np.ndarray] = []
    conf_rows: List[np.ndarray] = []
    weights: List[float] = []

    for idx, hospital in enumerate(hospitals):
        record = build_hospital_record(hospital, idx, partitions_root)
        records.append(record)
        fitz_rows.append(np.array(record["fitzpatrick_dist"], dtype=float))
        mod_rows.append(np.array(record["modality_dist"], dtype=float))
        conf_rows.append(np.array(record["confirmation_dist"], dtype=float))
        weights.append(float(record["n_samples"]))

    weights_arr = np.array(weights, dtype=float)
    total_weight = weights_arr.sum()
    weights_arr = (
        weights_arr / total_weight
        if total_weight > 0
        else np.full(len(hospitals), 1.0 / len(hospitals))
    )

    global_fitz = np.average(np.stack(fitz_rows), axis=0, weights=weights_arr)
    global_mod = np.average(np.stack(mod_rows), axis=0, weights=weights_arr)
    global_conf = np.average(np.stack(conf_rows), axis=0, weights=weights_arr)

    return hospitals, records, global_fitz.tolist(), global_mod.tolist(), global_conf.tolist()
