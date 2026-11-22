"""
hospital_registry.py
=====================
Adapter: dynamic hospital discovery for the Flower runtime.

Enumerates hospital_partitions/ at call time (sorted folder names, so every process maps a hospital to the same integer
index) and derives, per hospital, what the federation needs:

    n_samples          -- row count of the hospital's local training CSV
    tier / latency_*   -- straggler-environment parameters, assigned round-robin over (reliable, moderate,
                          high_straggler) in sorted hospital-name order (4/3/3 hospitals)

The metadata histograms and global reference distributions of the Novelty reference are NOT computed here: nothing in
this experiment (partitioning, selection, training, evaluation) reads them. The metadata the ResNet-50 model uses is
encoded by client/dataset.py (13-dimensional one-hot) and is unaffected.

If hospital_partitions/ does not exist, discover_hospitals() and build_hospital_registry() return empty results.
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import pandas as pd

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
# 2. Per-hospital record
# ===========================================================================

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
    """One hospital's record: n_samples, tier, latency_mu, latency_sigma (a plain dict)."""
    df = pd.read_csv(_hospital_metadata_path(hospital, partitions_root))
    tier = _TIER_CYCLE[index % len(_TIER_CYCLE)]
    latency_mu, latency_sigma = _TIER_LATENCY[tier]
    return {"n_samples": int(len(df)), "tier": tier, "latency_mu": latency_mu, "latency_sigma": latency_sigma}


# ===========================================================================
# 3. Full registry
# ===========================================================================

def build_hospital_registry(
    partitions_root: str = DEFAULT_PARTITIONS_ROOT,
) -> Tuple[List[str], List[Dict[str, object]]]:
    """Returns (hospital_names, hospital_records); records[i] belongs to names[i], and i is the client_id.
    client_fn resolves its partition from the same sorted order. ([], []) when hospital_partitions/ is absent."""
    hospitals = discover_hospitals(partitions_root)
    return hospitals, [build_hospital_record(h, i, partitions_root) for i, h in enumerate(hospitals)]
