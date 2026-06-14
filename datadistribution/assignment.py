"""
assignment.py
=============
Computes score_ik = w_k . f(m_i), applies softmax with tau_soft,
and probabilistically assigns each sample to a hospital.
Includes optional capacity control to avoid extreme imbalance.
"""

import numpy as np
import pandas as pd


def softmax(x: np.ndarray, tau: float = 1.0, axis: int = -1) -> np.ndarray:
    z = x / tau
    z = z - z.max(axis=axis, keepdims=True)  # stability
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def calculate_scores(F: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    F: (N, 3) sample features
    W: (K, 3) hospital weights
    Returns (N, K) dot-product scores.
    """
    return F @ W.T


def assign_hospitals(df: pd.DataFrame,
                     hospitals: list[dict],
                     tau_soft: float,
                     seed: int,
                     max_capacity_fraction: float = 0.25) -> pd.DataFrame:
    """
    Assign each row in df to a hospital via softmax sampling.
    Adds 'assigned_hospital_id' and 'assigned_hospital' columns.
    """
    rng = np.random.default_rng(seed)

    F = df[["f_phi", "f_tau", "f_lambda"]].to_numpy(dtype=float)
    W = np.array([[h["w_phi"], h["w_tau"], h["w_lambda"]] for h in hospitals], dtype=float)
    K = len(hospitals)
    N = len(df)

    scores = calculate_scores(F, W)         # (N, K)
    probs  = softmax(scores, tau=tau_soft)  # (N, K)

    # Capacity control
    max_capacity = int(np.ceil(max_capacity_fraction * N))
    counts = np.zeros(K, dtype=int)

    # Process in randomized order for fairness
    order = rng.permutation(N)
    assignments = np.full(N, -1, dtype=int)

    for idx in order:
        p = probs[idx].copy()
        # Zero-out hospitals that are full
        full_mask = counts >= max_capacity
        if full_mask.all():
            # All full (can happen if N tightly limited): allow overflow to argmax
            choice = int(np.argmax(probs[idx]))
        else:
            p[full_mask] = 0.0
            s = p.sum()
            if s <= 0:
                choice = int(rng.integers(0, K))
            else:
                p = p / s
                choice = int(rng.choice(K, p=p))
        assignments[idx] = choice
        counts[choice] += 1

    df = df.copy()
    df["assigned_hospital_id"] = assignments
    name_map = {h["id"]: h["name"] for h in hospitals}
    df["assigned_hospital"] = df["assigned_hospital_id"].map(name_map)
    print(f"[assignment] Done. Per-hospital counts: {counts.tolist()}")
    return df