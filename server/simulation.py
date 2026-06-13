"""
simulation.py  ·  FedMelanoma v4 — Full Federated Learning Simulation
======================================================================
Research-grade simulation of the FedMelanoma v4 client selection and straggler
management pipeline. Executes without real hospital infrastructure by generating
K=10 clinically heterogeneous client datasets from preprocessed_data.csv via
the Non-IID softmax-temperature partitioner (Section 3.1), then running T=50
federated rounds with the canonical Boltzmann selector + async straggler handler.

Pipeline
--------
  Step 1  Load preprocessed_data.csv → Non-IID partition into K=10 clients
          via softmax-temperature assignment (Section 3.1.3, τ_soft=0.5)
  Step 2  Compute OT-based diversity scores D_i via Wasserstein distances
          across three metadata axes (Section 3.2)
  Step 3  Instantiate Client objects with tier-specific straggler parameters
          (Section 4.2.2: reliable/moderate/high_straggler)
  Step 4  Run T=50 rounds:
            a. Compute scores + Boltzmann probabilities
            b. Select C=6 clients (Boltzmann, without replacement)
            c. Simulate responses (log-normal latency + Beta completion)
            d. manage_window → on-time / buffered classification
            e. Compute aggregation weights c_i (Section 3.10.2)
            f. Update R_i + N_eff for all selected clients
            g. (parallel) greedy selection → ablation counts
  Step 5  Generate 6 diagnostic plots saved to fedmelanoma_simulation.png

Diagnostic plots
----------------
  ① Selection frequency per client (bar, coloured by tier)
  ② Score evolution over rounds (per-client line chart)
  ③ Diversity D_i vs. mean Boltzmann selection probability (scatter)
  ④ N_eff exposure decay trajectory (N_eff vs score parametric paths)
  ⑤ Staleness histogram (s=0,1,2,3, failed bucket)
  ⑥ Boltzmann vs. greedy selection frequency (grouped bars, ablation)

Usage
-----
    python simulation.py                        # uses preprocessed_data.csv in cwd
    python simulation.py --csv /path/to/data    # specify CSV path
    python simulation.py --rounds 100 --clients 20 --seed 7

Dependencies: numpy, pandas, scipy, matplotlib
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

warnings.filterwarnings("ignore")

# Local modules (same package directory)
from selector import Client, Selector
from straggler import StragglerManager, ClientResponse

# ---------------------------------------------------------------------------
# Canonical hyperparameters (Section 3, FedMelanoma v4 blueprint)
# ---------------------------------------------------------------------------
GAMMA   = 1.0   # Diversity amplification weight γ
LAMBDA_ = 0.05  # Exposure attenuation rate λ  (diversity bonus half-life at N_eff=20)
TAU     = 0.20  # Boltzmann temperature τ        (strong exploitation regime)
RHO     = 0.85  # EMA decay ρ for R_i            (half-life ≈ 4.27 rounds)
DELTA   = 0.90  # Staleness discount δ            (unified across R_i, N_eff, aggregation)

# Default simulation configuration
N_CLIENTS_DEFAULT = 10    # K = 10 hospital nodes
N_SELECT_DEFAULT  = 6     # C = 6 clients per round (Section 3.9.2)
N_ROUNDS_DEFAULT  = 50    # T = 50 training rounds
RANDOM_SEED       = 42

# ---------------------------------------------------------------------------
# Straggler tier definitions (Section 4.2.2)
# ---------------------------------------------------------------------------
# Maps client index range → (latency_mu, latency_sigma, straggler_prob, tier_name)
TIER_CONFIG = {
    "reliable": {
        "indices":       [0, 1, 2, 3, 4],        # C0–C4 (Tertiary Oncology, General Derm)
        "latency_mu":    1.39,                    # LogN median ≈ exp(1.39) ≈ 4.0 s
        "latency_sigma": 0.30,
        "straggler_prob":0.05,
        "description":   "Tertiary Oncology / General Dermatology",
        "color":         "#2196F3",               # Blue
    },
    "moderate": {
        "indices":       [5, 6, 7],               # C5–C7 (Primary Care, Community Clinic)
        "latency_mu":    2.77,                    # LogN median ≈ exp(2.77) ≈ 16 s
        "latency_sigma": 0.50,
        "straggler_prob":0.30,
        "description":   "Primary Care / Community Clinic",
        "color":         "#FF9800",               # Orange
    },
    "high_straggler": {
        "indices":       [8, 9],                  # C8–C9 (Telemedicine, Rural Clinic)
        "latency_mu":    4.09,                    # LogN median ≈ exp(4.09) ≈ 60 s
        "latency_sigma": 0.80,
        "straggler_prob":0.60,
        "description":   "Telemedicine / Rural Clinic",
        "color":         "#F44336",               # Red
    },
}


def _get_tier(k: int, n_clients: int) -> str:
    """Map client index k to tier name, scaling tier boundaries with n_clients."""
    reliable_end       = int(round(n_clients * 0.50))  # first 50% → reliable
    moderate_end       = int(round(n_clients * 0.80))  # next 30% → moderate
    if k < reliable_end:
        return "reliable"
    elif k < moderate_end:
        return "moderate"
    else:
        return "high_straggler"


# ===========================================================================
# STEP 1: Non-IID data partitioning  (Section 3.1)
# ===========================================================================

def load_and_partition_data(
    csv_path:  str,
    n_clients: int,
    tau_soft:  float = 0.5,
    rng:       Optional[np.random.Generator] = None,
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    """
    Load preprocessed_data.csv and produce K Non-IID partitions via the
    softmax-temperature assignment mechanism (Section 3.1.3).

    Each sample is assigned to the hospital most compatible with its metadata:

        P(k | m_i) = softmax( W_k · f(m_i) / τ_soft )

    where:
      f(m_i) ∈ R^3   : affinity encoding of skin_type, image_type, confirm_type
      W ∈ R^{K×3}    : facility profile matrix (hospital metadata affinities)
      τ_soft = 0.5   : temperature controlling Non-IID skew

    WHY τ_soft = 0.5:
      At τ_soft=0.5, the assignment is heavily skewed — most samples go to the
      most-compatible hospital, producing the clinically realistic 15–17× spread
      in per-client malignancy rates (Section 3.1.3). Higher τ would produce
      more uniform distribution (less Non-IID); lower τ → deterministic assignment.

    The affinity encoding f(m_i) maps categorical metadata to scalar values
    reflecting clinical relevance gradient (Section 3.1.2):
      Fitzpatrick:   Type I → 0.10 (fair) ... Type VI → 1.00 (dark)
      Modality:      dermoscopic=0.95, clinical close-up=0.60, overview=0.30
      Confirmation:  histopathology=1.00 ... single contributor=0.75

    Parameters
    ----------
    csv_path  : Path to preprocessed_data.csv.
    n_clients : K (default 10).
    tau_soft  : Softmax temperature (default 0.5).
    rng       : numpy Generator.

    Returns
    -------
    (global_df, partitions) — global DataFrame and list of K per-client DataFrames.
    """
    rng = rng or np.random.default_rng(RANDOM_SEED)

    df = pd.read_csv(csv_path)
    print(f"\n[Data] Loaded {len(df):,} records from: {csv_path}")
    print(
        f"[Data] Malignancy: {df['label'].sum():,} / {len(df):,} "
        f"({100*df['label'].mean():.2f}%)"
    )
    print(f"[Data] Columns: {list(df.columns)}")

    # --- Affinity encoding f(m_i) → R^3 (Section 3.1.2) ---
    SKIN_AFF = {"I": 0.10, "II": 0.20, "III": 0.40, "IV": 0.60, "V": 0.80, "VI": 1.00}
    MOD_AFF  = {
        "dermoscopic":         0.95,
        "clinical: close-up":  0.60,
        "clinical: overview":  0.30,
    }
    CONF_AFF = {
        "histopathology":                         1.00,
        "serial imaging showing no change":       0.90,
        "single image expert consensus":          0.85,
        "single contributor clinical assessment": 0.75,
    }

    f_skin = df["skin_type"].map(SKIN_AFF).fillna(0.50).values   # (N,)
    f_mod  = df["image_type"].map(MOD_AFF).fillna(0.60).values   # (N,)
    f_conf = df["confirm_type"].map(CONF_AFF).fillna(0.80).values # (N,)
    F      = np.stack([f_skin, f_mod, f_conf], axis=1)            # (N, 3)

    # --- Facility profile matrix W ∈ R^{K×3} ---
    # Row k: [skin_affinity, modality_affinity, confirmation_affinity]
    # Low k → tertiary oncology: dermoscopic, histopath-confirmed, fair skin
    # High k → telemedicine:     clinical overview, single contributor, dark skin
    # Scaled to n_clients via linear interpolation between the two extremes.
    low_end  = np.array([0.10, 0.95, 1.00])   # Tertiary oncology profile
    high_end = np.array([1.00, 0.20, 0.50])   # Telemedicine profile
    alphas   = np.linspace(0.0, 1.0, n_clients)
    W        = np.array([(1 - a) * low_end + a * high_end for a in alphas])  # (K, 3)

    # --- Softmax-temperature assignment ---
    logits = (F @ W.T) / tau_soft    # (N, K)
    logits -= logits.max(axis=1, keepdims=True)  # Numerical stability
    probs   = np.exp(logits)
    probs  /= probs.sum(axis=1, keepdims=True)   # (N, K), each row sums to 1

    # Stochastic assignment: each sample drawn from its K-dimensional Boltzmann
    assignments = np.array([
        rng.choice(n_clients, p=probs[i])
        for i in range(len(df))
    ])

    partitions = [df[assignments == k].copy().reset_index(drop=True) for k in range(n_clients)]

    # --- Report partition statistics ---
    print(f"\n[Partition] Non-IID partition statistics (τ_soft={tau_soft}):")
    hdr = f"  {'ID':<6} {'N':>6} {'Mal%':>7}  {'SkinTypes':<28} {'Modalities'}"
    print(hdr)
    print("  " + "-" * 80)
    for k, part in enumerate(partitions):
        if len(part) == 0:
            print(f"  C{k:<5} {'0':>6} {'—':>7}  {'(empty)':}")
            continue
        mal_rate   = 100.0 * part["label"].mean()
        skin_types = ", ".join(sorted(part["skin_type"].unique()))
        mods       = ", ".join(sorted(part["image_type"].unique()))
        tier       = _get_tier(k, n_clients)
        print(
            f"  C{k:<5} {len(part):>6} {mal_rate:>6.2f}%  "
            f"{skin_types:<28} {mods}  [{tier}]"
        )

    return df, partitions


# ===========================================================================
# STEP 2: OT-based diversity scores  (Section 3.2)
# ===========================================================================

def compute_diversity_scores(
    partitions: List[pd.DataFrame],
    global_df:  pd.DataFrame,
) -> np.ndarray:
    """
    Compute static diversity priors D_i ∈ [0, 1] for each client via
    Wasserstein (Earth Mover's) distances over three metadata axes.

    WHY Optimal Transport (not KL divergence):
      KL(P‖Q) is undefined when Q(c)=0 for any category c where P(c)>0.
      This regularly occurs when comparing hospital distributions over rare
      metadata categories (e.g., a client with zero clinical:overview samples).
      OT (Wasserstein-1 / EMD) is always well-defined between discrete
      distributions and respects semantic geometry: the cost of transporting
      mass from "dermoscopic" to "clinical:overview" is greater than from
      "dermoscopic" to "clinical:close-up", which KL ignores entirely.

    Implementation: 1D Wasserstein distance on ordinal affinity encodings.
    For each axis, we treat each sample's affinity value as a point on R^1
    and compute the exact 1D Wasserstein distance (closed-form via sorted CDF
    comparison, implemented in scipy.stats.wasserstein_distance).

    Diversity score formula:
        raw_D_i = 0.5·W1_skin + 0.3·W1_mod + 0.2·W1_conf
        D_i     = raw_D_i / max_k(raw_D_k)   [normalise to [0, 1]]

    Axis weights (0.5 / 0.3 / 0.2):
      Skin type receives highest weight (0.5) for equity reasons: Types V–VI
      have 0.00% malignancy in the dataset due to systematic underrepresentation.
      Clients holding these phenotypes are the most important to preserve in the
      global model. Their high D_i score ensures the selector actively seeks
      them out in early training rounds.

    D_i = 0: client's distribution matches global centroid (no novel signal)
    D_i = 1: maximally different from global centroid (most novel signal)
    """
    SKIN_AFF = {"I": 0.10, "II": 0.20, "III": 0.40, "IV": 0.60, "V": 0.80, "VI": 1.00}
    MOD_AFF  = {
        "dermoscopic":         0.95,
        "clinical: close-up":  0.60,
        "clinical: overview":  0.30,
    }
    CONF_AFF = {
        "histopathology":                         1.00,
        "serial imaging showing no change":       0.90,
        "single image expert consensus":          0.85,
        "single contributor clinical assessment": 0.75,
    }

    def axis_values(df: pd.DataFrame, col: str, aff: dict) -> np.ndarray:
        vals = df[col].map(aff).dropna().values
        return vals if len(vals) > 0 else np.array([0.5])

    # Global reference distributions (centroid)
    g_skin = axis_values(global_df, "skin_type", SKIN_AFF)
    g_mod  = axis_values(global_df, "image_type", MOD_AFF)
    g_conf = axis_values(global_df, "confirm_type", CONF_AFF)

    raw = np.zeros(len(partitions))
    for k, part in enumerate(partitions):
        if len(part) < 2:
            raw[k] = 0.0
            continue

        c_skin = axis_values(part, "skin_type", SKIN_AFF)
        c_mod  = axis_values(part, "image_type", MOD_AFF)
        c_conf = axis_values(part, "confirm_type", CONF_AFF)

        # 1D Wasserstein distances per axis (scipy closed-form)
        w1_skin = wasserstein_distance(c_skin, g_skin)
        w1_mod  = wasserstein_distance(c_mod,  g_mod)
        w1_conf = wasserstein_distance(c_conf, g_conf)

        # Weighted combination: skin type receives 50% weight for equity
        raw[k] = 0.5 * w1_skin + 0.3 * w1_mod + 0.2 * w1_conf

    # Normalise to [0, 1]
    D = raw / raw.max() if raw.max() > 1e-10 else raw

    print("\n[Diversity] OT-based diversity scores D_i:")
    for k, d in enumerate(D):
        bar = "█" * int(d * 20)
        print(f"  C{k}: D_i = {d:.4f}  |{bar:<20}|")

    return D


# ===========================================================================
# STEP 3: Client instantiation
# ===========================================================================

def create_clients(
    partitions:       List[pd.DataFrame],
    diversity_scores: np.ndarray,
    n_clients:        int,
    rng:              np.random.Generator,
) -> List[Client]:
    """
    Instantiate Client objects with tier-specific parameters derived from
    the blueprint's straggler model (Section 4.2.2).

    Per-client parameter perturbation:
      Small Gaussian noise is added to latency_mu and latency_sigma to model
      per-hospital variability within each tier. This prevents all reliable
      clients from having identical response time distributions, which would
      produce unrealistically regular simulation outputs.

    Minimum n_samples = 50: prevents degenerate partition behaviour for empty
    clients that might arise from the softmax-temperature assignment.
    """
    clients = []

    print("\n[Clients] Initialised client roster:")
    print(
        f"  {'ID':<5} {'Tier':<18} {'N_samples':>10} "
        f"{'D_i':>8} {'LatMu':>7} {'LatSig':>7} {'StragP':>7}"
    )
    print("  " + "-" * 70)

    for k in range(n_clients):
        part = partitions[k] if k < len(partitions) else pd.DataFrame()
        tier = _get_tier(k, n_clients)
        cfg  = TIER_CONFIG[tier]

        # Per-client noise on latency parameters (±5% mu, ±2% sigma)
        lat_mu    = cfg["latency_mu"]    + rng.normal(0.0, 0.05)
        lat_sigma = cfg["latency_sigma"] + abs(rng.normal(0.0, 0.02))

        client = Client(
            client_id       = k,
            n_samples       = max(int(len(part)), 50),
            R               = 1.0,                   # R_i^(0) = 1.0 (Section 3.5)
            D               = float(diversity_scores[k]),
            N_eff           = 0.0,                   # No prior interactions
            latency_mu      = float(lat_mu),
            latency_sigma   = float(lat_sigma),
            straggler_prob  = cfg["straggler_prob"],
            tier            = tier,
        )
        clients.append(client)

        print(
            f"  C{k:<4} {tier:<18} {client.n_samples:>10,} "
            f"{client.D:>8.4f} {client.latency_mu:>7.3f} "
            f"{client.latency_sigma:>7.3f} {client.straggler_prob:>7.3f}"
        )

    return clients


# ===========================================================================
# STEP 4: Federated simulation
# ===========================================================================

class FedMelanomaSimulation:
    """
    Full FedMelanoma v4 federated round simulation.

    Executes T rounds of the five-stage lifecycle (Section 4.3.1):
      1. Selection   — Boltzmann score → C=6 clients without replacement
      2. Broadcast   — (no actual model; simulated via no-op)
      3. Local train — (simulated via response_time only)
      4. Collection  — manage_window classifies on-time vs buffered
      5. Aggregation — c_i weights computed; state updated (R_i, N_eff)

    Also runs a parallel greedy selection for the ablation plot (Plot 6).

    Metrics tracked per round
    -------------------------
    score_matrix        : (T, K) scores at start of each round
    n_eff_matrix        : (T, K) N_eff values at start of each round
    boltzmann_probs     : (T, K) Boltzmann probability vectors
    selection_entropy   : (T,)   Shannon entropy H(p) per round
    selected_per_round  : (T,)   list of selected client IDs
    selection_counts    : (K,)   total selections per client (Boltzmann)
    greedy_counts       : (K,)   total selections per client (Greedy)
    all_staleness       : flat list of all s_i values (finite + failed)
    contribution_weights: per-round dict {cid: c_i}
    on_time_per_round   : (T,) on-time arrivals per round
    straggler_per_round : (T,) buffered stragglers per round
    failed_per_round    : (T,) failed clients per round
    """

    def __init__(
        self,
        clients:       List[Client],
        selector:      Selector,
        straggler_mgr: StragglerManager,
        n_select:      int = N_SELECT_DEFAULT,
        n_rounds:      int = N_ROUNDS_DEFAULT,
        rng:           Optional[np.random.Generator] = None,
    ) -> None:
        self.clients       = clients
        self.selector      = selector
        self.straggler     = straggler_mgr
        self.n_select      = n_select
        self.n_rounds      = n_rounds
        self.rng           = rng or np.random.default_rng(RANDOM_SEED)
        self.K             = len(clients)

        # --- Metrics storage ---
        self.score_matrix        = np.zeros((n_rounds, self.K))
        self.n_eff_matrix        = np.zeros((n_rounds, self.K))
        self.boltzmann_probs_mat = np.zeros((n_rounds, self.K))
        self.selection_entropy:  List[float] = []
        self.selected_per_round: List[List[int]] = []
        self.selection_counts    = np.zeros(self.K, dtype=int)
        self.greedy_counts       = np.zeros(self.K, dtype=int)
        self.all_staleness:      List[float] = []
        self.contribution_weights: List[Dict[int, float]] = []
        self.on_time_per_round:  List[int] = []
        self.straggler_per_round:List[int] = []
        self.failed_per_round:   List[int] = []

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Execute all T federated rounds.

        Each round follows the blueprint lifecycle (Section 4.3.1):
          1. Compute scores + Boltzmann probs for all K clients
          2. Boltzmann selection (without replacement) → selected set S^t
          3. Simulate responses for each selected client
          4. manage_window → on-time A^t + buffered stragglers
          5. Compute aggregation weights c_i for A^t
          6. Update R_i + N_eff for all i ∈ S^t
          7. Greedy selection (parallel, no state mutation) → ablation counts
          8. Log all metrics and state

        Non-selected clients (i ∉ S^t):
          R_i unchanged, N_eff unchanged, history appended without update.
        """
        print(f"\n{'═'*68}")
        print(f"  FedMelanoma v4 Simulation")
        print(f"  K={self.K} clients | C={self.n_select} per round | T={self.n_rounds} rounds")
        print(f"  γ={GAMMA} | λ={LAMBDA_} | τ={TAU} | ρ={RHO} | δ={DELTA}")
        print(f"{'═'*68}\n")

        for t in range(self.n_rounds):
            print(f"\n{'─'*60}")
            print(f"  ROUND {t+1:>2}/{self.n_rounds}")
            print(f"{'─'*60}")

            # ── 1. Record current scores and N_eff ──────────────────────
            scores = np.array([self.selector.compute_score(c) for c in self.clients])
            self.score_matrix[t]  = scores
            self.n_eff_matrix[t]  = np.array([c.N_eff for c in self.clients])

            # ── 2. Compute full Boltzmann probability vector ─────────────
            probs = self.selector.boltzmann_probs(self.clients)
            self.boltzmann_probs_mat[t] = probs

            H = float(-np.sum(probs * np.log(probs + 1e-12)))   # Shannon entropy
            self.selection_entropy.append(H)

            # ── 3. Boltzmann selection ───────────────────────────────────
            selected = self.selector.boltzmann_selection(
                self.clients, self.n_select, rng=self.rng
            )
            sel_ids = [c.client_id for c in selected]
            self.selected_per_round.append(sel_ids)
            for cid in sel_ids:
                self.selection_counts[cid] += 1

            print(f"  → Boltzmann selected: {['C'+str(i) for i in sel_ids]}")

            # ── 4. Greedy selection (ablation, no state side-effects) ────
            greedy_sel = self.selector.greedy_selection(self.clients, self.n_select)
            for c in greedy_sel:
                self.greedy_counts[c.client_id] += 1

            # ── 5. Simulate responses ────────────────────────────────────
            responses: Dict[int, ClientResponse] = {}
            for client in selected:
                resp = self.straggler.simulate_response(client, current_round=t)
                responses[client.client_id] = resp

            # ── 6. Manage aggregation window ─────────────────────────────
            on_time, stragglers = self.straggler.manage_window(
                selected, responses, current_round=t, n_selected=self.n_select
            )
            self.on_time_per_round.append(len(on_time))
            self.straggler_per_round.append(len(stragglers))
            self.failed_per_round.append(
                self.n_select - len(on_time) - len(stragglers)
            )

            # ── 7. Compute aggregation weights c_i ───────────────────────
            agg_weights = self._compute_aggregation_weights(on_time)
            self.contribution_weights.append(agg_weights)

            if agg_weights:
                wt_str = "  ".join(
                    f"C{cid}={w:.4f}" for cid, w in sorted(agg_weights.items())
                )
                print(f"  → Aggregation weights: {wt_str}")
            else:
                print(f"  → Aggregation weights: (no on-time clients)")

            # ── 8. Update R_i + N_eff for ALL selected clients ───────────
            # The aggregation weight depends on staleness, but R_i and N_eff
            # are updated regardless of whether the client was on-time or
            # straggler — both outcomes reflect a real participation attempt.
            # Failure (s=∞) contracts R_i but leaves N_eff unchanged.
            #
            # Non-selected clients: no update (Section 4.3.2)
            selected_cid_set = set(sel_ids)
            for client in self.clients:
                if client.client_id in selected_cid_set:
                    resp = responses.get(client.client_id)
                    if resp is not None:
                        final_s = resp.staleness  # 0.0, 1.0, or inf (mutated by manage_window)
                        self.selector.update_client_state(
                            client, final_s, round_idx=t, was_selected=True
                        )
                        # Record staleness (use S_max+1 as proxy for inf/failure)
                        self.all_staleness.append(
                            final_s if not np.isinf(final_s)
                            else float(self.straggler.s_max + 1)
                        )
                    else:
                        # Selected but no response object (edge case)
                        self.selector.update_client_state(
                            client, float("inf"), round_idx=t, was_selected=True
                        )
                        self.all_staleness.append(float(self.straggler.s_max + 1))
                else:
                    # Non-selected: append history without modifying R or N_eff
                    self.selector.update_client_state(
                        client, 0.0, round_idx=t, was_selected=False
                    )

            # ── 9. Per-round state summary ───────────────────────────────
            print(f"  → Post-round state:")
            for c in self.clients:
                bd = self.selector.score_breakdown(c)
                selected_marker = "◀" if c.client_id in selected_cid_set else " "
                print(
                    f"    {selected_marker} C{c.client_id}: "
                    f"R={bd['R_i']:.3f}  D={bd['D_i']:.3f}  "
                    f"N_eff={bd['N_eff']:.2f}  "
                    f"U_eff={bd['U_eff']:.3f}  "
                    f"score={bd['score']:.4f}"
                )

        # ── Simulation complete ──────────────────────────────────────────
        print(f"\n{'═'*68}")
        print(f"  Simulation complete.")
        print(f"  Total Boltzmann selections: {self.selection_counts.sum()}")
        print(f"  Selection counts (Boltzmann): "
              f"{dict(enumerate(self.selection_counts))}")
        print(f"  Selection counts (Greedy):    "
              f"{dict(enumerate(self.greedy_counts))}")
        strag_mgr_summary = self.straggler.summary()
        if strag_mgr_summary:
            print(f"  Straggler summary: {strag_mgr_summary}")
        print(f"{'═'*68}\n")

    # ------------------------------------------------------------------
    # Aggregation weight computation  (Section 3.10.2)
    # ------------------------------------------------------------------

    def _compute_aggregation_weights(
        self,
        on_time: List[ClientResponse],
    ) -> Dict[int, float]:
        """
        Compute normalised aggregation weights c_i for the arrived client set A^t.

        Three-factor formula (Section 3.10.2):

            c̃_i = (n_i / n_A) · δ^{s_i} · (1 + γ · D_i)
            c_i  = c̃_i / Σ_j c̃_j

        Factor interpretation:
          n_i / n_A      : Size proportionality (FedAvg baseline — maximum likelihood
                           interpretation of weighted averaging). Clients with more
                           samples contribute proportionally more to the global model.
          δ^{s_i}        : Staleness discount. Outdated gradients contribute less.
                           On-time (s=0): δ^0=1.0 (full contribution).
                           Buffered (s=1): δ^1=0.9 (10% discount).
                           Stale (s=2): δ^2=0.81 (19% discount).
          (1 + γ·D_i)    : Diversity amplification. Clients with rare patient
                           phenotypes receive boosted aggregation weight, ensuring
                           their gradient information disproportionately influences
                           the global model update. Same γ=1.0 as selector score.

        Normalisation: Σ_i c_i = 1 guaranteed by division by total.
        If all on-time clients fail (empty list), returns empty dict.
        """
        if not on_time:
            return {}

        cid_map = {c.client_id: c for c in self.clients}

        # Compute n_A = total samples of all arrived clients
        n_A = sum(
            cid_map[r.client_id].n_samples
            for r in on_time
            if r.client_id in cid_map
        )
        if n_A == 0:
            return {}

        c_tilde: Dict[int, float] = {}
        for resp in on_time:
            cid = resp.client_id
            if cid not in cid_map:
                continue
            client = cid_map[cid]
            staleness_w = self.straggler.apply_staleness_weight(DELTA, resp.staleness)
            c_tilde[cid] = (
                (client.n_samples / n_A)
                * staleness_w
                * (1.0 + GAMMA * client.D)
            )

        total = sum(c_tilde.values())
        if total < 1e-12:
            return {}

        return {cid: float(v / total) for cid, v in c_tilde.items()}


# ===========================================================================
# STEP 5: Diagnostic plots
# ===========================================================================

def generate_plots(
    sim:         FedMelanomaSimulation,
    clients:     List[Client],
    output_path: str = "fedmelanoma_simulation.png",
) -> None:
    """
    Produce all 6 diagnostic plots described in the simulation specification.

    ①  Selection frequency per client
        Bar chart showing how many rounds each client was selected.
        Colour-coded by tier. Uniform baseline (1/K) shown as dashed line.
        Demonstrates Boltzmann's no-starvation property: even high-straggler
        clients (C8–C9) receive non-zero selection across T=50 rounds.

    ②  Score evolution over rounds
        Per-client score_i(t) trajectory. Shows the automatic exploration-to-
        exploitation transition: scores converge toward R_i as N_eff grows and
        the diversity bonus attenuates. High-straggler clients (dashed) have
        lower scores but non-zero selection probability.

    ③  Diversity D_i vs. mean selection probability
        Scatter plot testing the core hypothesis: clients with higher OT
        diversity scores should receive higher mean Boltzmann selection
        probability, at least in early rounds when N_eff is low.

    ④  N_eff exposure decay trajectory
        Parametric paths in (N_eff, score) space, one per client. Shows each
        client's score decreasing as N_eff grows (diversity bonus attenuates).
        Circle = start of training (round 1), star = end (round T).
        Annotated with theoretical asymptote: score → R_i as N_eff → ∞.

    ⑤  Staleness histogram
        Distribution of s_i values across all selected clients and all rounds.
        Bucketed as s=0 (on-time), s=1 (buffered), s=2,3 (multi-round stale),
        and s>S_max (failed/expired). Annotated with δ^s values.

    ⑥  Boltzmann vs. greedy selection frequency (ablation)
        Side-by-side bars comparing Boltzmann (production) and greedy top-K
        (ablation) selection frequencies. Greedy is expected to show near-zero
        frequency for high-straggler clients, while Boltzmann maintains non-zero
        frequency for all — visually demonstrating the starvation failure mode.
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    TIER_COL = {
        "reliable":       "#1565C0",   # Dark blue
        "moderate":       "#E65100",   # Dark orange
        "high_straggler": "#B71C1C",   # Dark red
    }
    TIER_LIGHT = {
        "reliable":       "#90CAF9",
        "moderate":       "#FFCC80",
        "high_straggler": "#EF9A9A",
    }

    K         = len(clients)
    T         = sim.n_rounds
    cids      = [f"C{c.client_id}" for c in clients]
    clr       = [TIER_COL[c.tier] for c in clients]
    total_sel = T * sim.n_select

    fig = plt.figure(figsize=(22, 20))
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle(
        "FedMelanoma v4 — Federated Client Selection & Straggler Handling Diagnostics",
        fontsize=15, fontweight="bold", y=0.99, color="#1A1A2E",
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.46, wspace=0.34)

    x_axis = np.arange(K)

    # ── Shared legend elements ───────────────────────────────────────────────
    tier_patches = [
        Patch(facecolor=TIER_COL["reliable"],       label="Reliable (C0–C4): Tertiary Oncology"),
        Patch(facecolor=TIER_COL["moderate"],        label="Moderate (C5–C7): Primary Care"),
        Patch(facecolor=TIER_COL["high_straggler"],  label="High-Straggler (C8–C9): Telemedicine"),
    ]

    # =======================================================================
    # Plot ①: Selection frequency per client
    # =======================================================================
    ax1 = fig.add_subplot(gs[0, 0])

    boltz_freq  = sim.selection_counts / total_sel
    uniform_bsln= 1.0 / K

    bars1 = ax1.bar(
        x_axis, boltz_freq,
        color=clr, edgecolor="white", linewidth=0.9, zorder=3, alpha=0.92,
    )

    # Uniform baseline
    ax1.axhline(
        uniform_bsln, color="#555555", ls="--", lw=1.6,
        label=f"Uniform baseline (1/{K}={uniform_bsln:.3f})", zorder=2,
    )

    # Annotate with raw count
    for bar, cnt in zip(bars1, sim.selection_counts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.002,
            str(cnt),
            ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#222222",
        )

    ax1.set_xticks(x_axis); ax1.set_xticklabels(cids, fontsize=9)
    ax1.set_xlabel("Client ID", fontsize=11)
    ax1.set_ylabel("Selection frequency (fraction of total slots)", fontsize=10)
    ax1.set_ylim(0, max(boltz_freq) * 1.22)
    ax1.set_title(
        "① Selection Frequency per Client\n"
        f"(Boltzmann τ={TAU} | C={sim.n_select}/round | T={T} rounds)",
        fontsize=11, fontweight="bold",
    )
    ax1.legend(
        handles=tier_patches + [Line2D([0],[0], color="#555", ls="--", lw=1.6,
                                        label=f"Uniform baseline")],
        fontsize=8, loc="upper right",
    )

    # =======================================================================
    # Plot ②: Score evolution over rounds
    # =======================================================================
    ax2 = fig.add_subplot(gs[0, 1])

    rounds = np.arange(1, T + 1)
    for k, client in enumerate(clients):
        lw  = 2.0 if client.tier == "reliable" else (1.6 if client.tier == "moderate" else 1.3)
        ls  = "-" if client.tier != "high_straggler" else "--"
        alpha = 0.90 if client.tier == "reliable" else 0.80

        ax2.plot(
            rounds, sim.score_matrix[:, k],
            color=clr[k], lw=lw, ls=ls, alpha=alpha,
            label=f"C{k}",
        )

    ax2.axhline(1.0, color="#999", ls=":", lw=1.2, alpha=0.7,
                label="score=R_i (N_eff→∞ asymptote)")
    ax2.axhline(2.0, color="#ccc", ls=":", lw=1.0, alpha=0.5,
                label="Max score = 2.0")

    ax2.set_xlim(1, T); ax2.set_ylim(0, 2.15)
    ax2.set_xlabel("Round t", fontsize=11)
    ax2.set_ylabel("score_i(t)", fontsize=11)
    ax2.set_title(
        "② Score Evolution Over Rounds\n"
        r"$\mathrm{score}_i(t) = R_i \cdot \left(1 + \gamma \frac{D_i}{1 + \lambda N_i^\mathrm{eff}}\right)$",
        fontsize=11, fontweight="bold",
    )
    ax2.legend(fontsize=7.5, loc="upper right", ncol=3)

    # Annotate exploration→exploitation transition
    ax2.annotate(
        "← Exploration\n   (diversity bonus active)",
        xy=(T * 0.1, 1.65), fontsize=8, color="#333", style="italic",
    )
    ax2.annotate(
        "Exploitation →\n(score → R_i)",
        xy=(T * 0.68, 1.25), fontsize=8, color="#333", style="italic",
    )

    # =======================================================================
    # Plot ③: Diversity vs. selection probability
    # =======================================================================
    ax3 = fig.add_subplot(gs[1, 0])

    mean_probs  = sim.boltzmann_probs_mat.mean(axis=0)
    early_probs = sim.boltzmann_probs_mat[:T//4].mean(axis=0)   # first 25% of rounds
    late_probs  = sim.boltzmann_probs_mat[3*T//4:].mean(axis=0) # last 25% of rounds
    divers      = np.array([c.D for c in clients])

    # Plot all-round mean (solid) and early/late (faded)
    for k, client in enumerate(clients):
        ax3.scatter(divers[k], mean_probs[k],
                    s=130, color=clr[k], edgecolors="white", lw=1.2, zorder=4)
        ax3.scatter(divers[k], early_probs[k],
                    s=60, color=clr[k], marker="^", alpha=0.55, zorder=3)
        ax3.scatter(divers[k], late_probs[k],
                    s=60, color=clr[k], marker="v", alpha=0.55, zorder=3)
        ax3.annotate(
            f"C{k}", (divers[k], mean_probs[k]),
            textcoords="offset points", xytext=(5, 4), fontsize=8.5,
        )

    # Linear trend line over mean probs
    if K > 2:
        z = np.polyfit(divers, mean_probs, 1)
        p_fn = np.poly1d(z)
        x_ln = np.linspace(divers.min() - 0.02, divers.max() + 0.02, 200)
        ax3.plot(x_ln, p_fn(x_ln), "k--", lw=1.4, alpha=0.45, label="Trend (all rounds)")

    ax3.axhline(1.0 / K, color="#888", ls=":", lw=1.1, alpha=0.6,
                label=f"Uniform prob = {1/K:.3f}")
    ax3.set_xlabel("Diversity D_i (Wasserstein OT score)", fontsize=11)
    ax3.set_ylabel("Mean Boltzmann selection probability", fontsize=11)
    ax3.set_title(
        "③ Diversity vs. Selection Probability\n"
        "(▲ early rounds | ● all rounds | ▼ late rounds)",
        fontsize=11, fontweight="bold",
    )
    leg_el3 = tier_patches + [
        Line2D([0],[0], color="k", ls="--", lw=1.4, alpha=0.5, label="Trend"),
        Line2D([0],[0], color="#888", ls=":", lw=1.1, label="Uniform prob"),
    ]
    ax3.legend(handles=leg_el3, fontsize=8, loc="upper left")

    # =======================================================================
    # Plot ④: N_eff exposure decay trajectory
    # =======================================================================
    ax4 = fig.add_subplot(gs[1, 1])

    for k, client in enumerate(clients):
        n_effs = sim.n_eff_matrix[:, k]
        scrs   = sim.score_matrix[:, k]

        ax4.plot(n_effs, scrs, color=clr[k], lw=1.6, alpha=0.80)

        # Start marker (round 0)
        ax4.scatter(n_effs[0],  scrs[0],  s=55,  color=clr[k], marker="o",
                    zorder=5, edgecolors="white", lw=1)
        # End marker (round T-1)
        ax4.scatter(n_effs[-1], scrs[-1], s=100, color=clr[k], marker="*",
                    zorder=6, edgecolors="white", lw=0.8)

        # Asymptote label for a representative reliable client
        if client.tier == "reliable" and k == 0:
            ax4.annotate(
                f"  R_i={client.R:.2f}\n  (asymptote)",
                (n_effs[-1], client.R),
                fontsize=7.5, color="#444", va="center",
            )

    # Theoretical attenuation curve for a "perfect client" (R=1, D=1)
    n_range  = np.linspace(0, sim.n_eff_matrix.max() + 1, 300)
    U_theory = 1.0 / (1.0 + LAMBDA_ * n_range)
    ax4.plot(
        n_range, 1.0 + GAMMA * U_theory,
        "k:", lw=1.5, alpha=0.4,
        label=r"Theoretical: $1 + \gamma/(1+\lambda N)$ (R=1, D=1)",
    )

    ax4.set_xlabel(r"$N_i^{\mathrm{eff}}(t)$ — cumulative effective interactions", fontsize=10)
    ax4.set_ylabel("score_i(t)", fontsize=11)
    ax4.set_ylim(0, 2.15)
    ax4.set_title(
        "④ N_eff Exposure Decay Effect on Score\n"
        r"(○ = round 1 | ★ = round T; score → $R_i$ as $N^{\mathrm{eff}} \to \infty$)",
        fontsize=11, fontweight="bold",
    )
    ax4.legend(handles=tier_patches + [
        Line2D([0],[0], color="k", ls=":", lw=1.5, alpha=0.5, label="Theoretical (R=1,D=1)"),
    ], fontsize=8)

    # =======================================================================
    # Plot ⑤: Staleness histogram
    # =======================================================================
    ax5 = fig.add_subplot(gs[2, 0])

    S_MAX  = sim.straggler.s_max
    all_s  = np.array(sim.all_staleness)
    finite = all_s[all_s <= S_MAX]
    failed = all_s[all_s > S_MAX]

    bins   = np.arange(-0.5, S_MAX + 2.5, 1.0)
    counts_hist, _ = np.histogram(finite, bins=bins)

    # Plot finite-staleness bars
    bar_x     = np.arange(0, S_MAX + 1)
    bar_colors= ["#1565C0", "#388E3C", "#F57F17", "#B71C1C"][:S_MAX + 1]
    bars5 = ax5.bar(
        bar_x, counts_hist[1:S_MAX+2], width=0.65,
        color=bar_colors, edgecolor="white", lw=0.9, zorder=3,
    )

    # Failed / expired bucket
    ax5.bar(
        S_MAX + 1, len(failed), width=0.65,
        color="#880E4F", edgecolor="white", lw=0.9, zorder=3,
        label=f"Failed / expired (s>{S_MAX})",
    )

    # Annotate with δ^s values
    for s_val, bar in enumerate(bars5):
        dw = DELTA ** s_val
        ax5.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"δ^{s_val}={dw:.3f}",
            ha="center", va="bottom", fontsize=8.5, color="#333",
        )
    ax5.text(
        S_MAX + 1, len(failed) + 0.5,
        "δ^∞=0.0",
        ha="center", va="bottom", fontsize=8.5, color="#880E4F",
    )

    tick_labels = [str(i) for i in range(S_MAX + 1)] + [f">{S_MAX}\n(fail)"]
    ax5.set_xticks(np.arange(S_MAX + 2))
    ax5.set_xticklabels(tick_labels, fontsize=9.5)
    ax5.set_xlabel("Staleness s_i (rounds delayed)", fontsize=11)
    ax5.set_ylabel("Count across all rounds", fontsize=11)
    ax5.set_title(
        "⑤ Staleness Distribution\n"
        f"(δ={DELTA}: δ^0=1.00, δ^1={DELTA:.2f}, "
        f"δ^2={DELTA**2:.3f}, δ^3={DELTA**3:.3f})",
        fontsize=11, fontweight="bold",
    )
    ax5.legend(fontsize=9)

    # =======================================================================
    # Plot ⑥: Boltzmann vs. greedy comparison (ablation)
    # =======================================================================
    ax6 = fig.add_subplot(gs[2, 1])

    boltz_norm  = sim.selection_counts / total_sel
    greedy_norm = sim.greedy_counts    / total_sel

    w = 0.36
    bars6b = ax6.bar(
        x_axis - w / 2, boltz_norm, width=w,
        color=clr, edgecolor="white", lw=0.9, zorder=3, alpha=0.92,
        label="Boltzmann (production)",
    )
    bars6g = ax6.bar(
        x_axis + w / 2, greedy_norm, width=w,
        color=[TIER_LIGHT[c.tier] for c in clients],
        edgecolor="#444", lw=0.9, zorder=3, alpha=0.80,
        hatch="//", label="Greedy top-K (ablation)",
    )

    ax6.axhline(uniform_bsln, color="#888", ls="--", lw=1.5,
                label=f"Uniform baseline (1/{K})")

    ax6.set_xticks(x_axis); ax6.set_xticklabels(cids, fontsize=9)
    ax6.set_xlabel("Client ID", fontsize=11)
    ax6.set_ylabel("Selection frequency", fontsize=11)
    ax6.set_title(
        "⑥ Boltzmann vs. Greedy Selection (Ablation)\n"
        "Greedy → starvation of high-straggler clients | Boltzmann → no starvation",
        fontsize=11, fontweight="bold",
    )

    # Fairness metric annotation
    H_mean    = float(np.mean(sim.selection_entropy))
    H_max     = float(np.log(K))
    H_ratio   = H_mean / H_max if H_max > 0 else 0
    n_starved = int(np.sum(sim.greedy_counts == 0))

    ax6.text(
        0.02, 0.96,
        f"Boltzmann: avg H(p)={H_mean:.3f} / {H_max:.3f} (ratio={H_ratio:.2f})\n"
        f"Greedy: {n_starved}/{K} clients starved (zero selections)",
        transform=ax6.transAxes, fontsize=8.5, va="top", color="#1A237E",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#E3F2FD", alpha=0.8, ec="#90CAF9"),
    )

    ablation_handles = [
        Line2D([0],[0], color="#1565C0", lw=8, alpha=0.92, label="Boltzmann (production)"),
        Line2D([0],[0], color="#90CAF9", lw=8, alpha=0.80,
               label="Greedy top-K (ablation)", ls="-"),
        Line2D([0],[0], color="#888", ls="--", lw=1.5, label=f"Uniform (1/{K})"),
    ]
    ax6.legend(handles=ablation_handles, fontsize=8.5, loc="upper right")

    # =======================================================================
    # Final layout and save
    # =======================================================================
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n[Plot] Saved to: {output_path}")
    plt.close(fig)


# ===========================================================================
# Metrics summary
# ===========================================================================

def print_metrics_summary(
    sim:     FedMelanomaSimulation,
    clients: List[Client],
) -> None:
    """Print a clean post-simulation metrics report."""
    K    = len(clients)
    T    = sim.n_rounds
    H    = np.array(sim.selection_entropy)
    H_max = np.log(K)

    print(f"\n{'═'*60}")
    print("  POST-SIMULATION METRICS REPORT")
    print(f"{'═'*60}")

    print(f"\n  Rounds:          {T}")
    print(f"  Clients (K):     {K}")
    print(f"  Select/round (C):{sim.n_select}")

    print(f"\n  ─── Selection ───────────────────────────────────────")
    print(f"  {'Client':<10} {'Boltz':>8} {'Greedy':>8} {'D_i':>8} "
          f"{'Tier'}")
    for c in clients:
        b_cnt = sim.selection_counts[c.client_id]
        g_cnt = sim.greedy_counts[c.client_id]
        print(
            f"  C{c.client_id:<9} {b_cnt:>8} {g_cnt:>8} "
            f"{c.D:>8.4f} {c.tier}"
        )

    print(f"\n  ─── Fairness (Shannon entropy H(p)) ─────────────────")
    print(f"  Max entropy (uniform): {H_max:.4f}")
    print(f"  Mean H(p):             {H.mean():.4f}  ({100*H.mean()/H_max:.1f}% of max)")
    print(f"  Final H(p) round T:    {H[-1]:.4f}  ({100*H[-1]/H_max:.1f}% of max)")
    print(f"  Greedy starved clients:{int(np.sum(sim.greedy_counts==0))}/{K}")

    print(f"\n  ─── Staleness ────────────────────────────────────────")
    all_s = np.array(sim.all_staleness)
    s_fin = all_s[all_s <= sim.straggler.s_max]
    s_inf = all_s[all_s > sim.straggler.s_max]
    print(f"  Total selected events: {len(all_s)}")
    print(f"  On-time   (s=0):       {int(np.sum(s_fin==0))} "
          f"({100*np.mean(s_fin==0 if len(s_fin)>0 else [0]):.1f}%)")
    print(f"  Buffered  (s=1):       {int(np.sum(s_fin==1))}")
    print(f"  Stale 2+  (s≥2):       {int(np.sum(s_fin>=2))}")
    print(f"  Failed/Expired:        {len(s_inf)}")
    if len(s_fin) > 0:
        print(f"  Mean staleness (arrived): {s_fin.mean():.4f}")

    print(f"\n  ─── Straggler window ─────────────────────────────────")
    mgr = sim.straggler
    if mgr.window_history:
        print(f"  Mean W_t:   {np.mean(mgr.window_history):.2f} s")
        print(f"  Final W_t:  {mgr.window_history[-1]:.2f} s")
    print(f"  Quorum rate:{100*np.mean(mgr.quorum_met_history):.1f}%")

    print(f"\n  ─── Final client states ──────────────────────────────")
    print(f"  {'Client':<10} {'R_i':>6} {'D_i':>6} {'N_eff':>8} {'score':>8}")
    selector_tmp = Selector(gamma=GAMMA, lambda_=LAMBDA_, tau=TAU, rho=RHO, delta=DELTA)
    for c in clients:
        s = selector_tmp.compute_score(c)
        print(f"  C{c.client_id:<9} {c.R:>6.3f} {c.D:>6.3f} {c.N_eff:>8.2f} {s:>8.4f}")

    print(f"\n{'═'*60}\n")


# ===========================================================================
# Entry point
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FedMelanoma v4 — Client Selection & Straggler Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv",     default="preprocessed_data.csv",
                   help="Path to preprocessed_data.csv")
    p.add_argument("--clients", type=int, default=N_CLIENTS_DEFAULT,
                   help="Number of federated clients K")
    p.add_argument("--select",  type=int, default=N_SELECT_DEFAULT,
                   help="Clients selected per round C")
    p.add_argument("--rounds",  type=int, default=N_ROUNDS_DEFAULT,
                   help="Number of training rounds T")
    p.add_argument("--seed",    type=int, default=RANDOM_SEED,
                   help="Random seed for reproducibility")
    p.add_argument("--output",  default="fedmelanoma_simulation.png",
                   help="Output path for the diagnostic plots PNG")
    p.add_argument("--tau_soft",type=float, default=0.5,
                   help="Softmax temperature for Non-IID partitioning")
    return p.parse_args()


def main() -> None:
    """
    End-to-end FedMelanoma v4 simulation.

    Execution order:
      1. Parse CLI arguments
      2. Load + partition preprocessed_data.csv → K Non-IID clients
      3. Compute OT diversity scores D_i
      4. Instantiate Client objects
      5. Initialise Selector + StragglerManager
      6. Run T federated rounds
      7. Generate 6 diagnostic plots → PNG
      8. Print metrics summary
    """
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    print("\n" + "═" * 68)
    print("  FedMelanoma v4 — Research Simulation")
    print(f"  Canonical hyperparameters: γ={GAMMA} λ={LAMBDA_} τ={TAU} ρ={RHO} δ={DELTA}")
    print("═" * 68)

    # ── Step 1: Non-IID data partition ──────────────────────────────────────
    global_df, partitions = load_and_partition_data(
        csv_path  = args.csv,
        n_clients = args.clients,
        tau_soft  = args.tau_soft,
        rng       = rng,
    )

    # ── Step 2: OT diversity scores ─────────────────────────────────────────
    D_scores = compute_diversity_scores(partitions, global_df)

    # ── Step 3: Client objects ───────────────────────────────────────────────
    clients = create_clients(partitions, D_scores, args.clients, rng)

    # ── Step 4: Core components ──────────────────────────────────────────────
    selector = Selector(
        gamma   = GAMMA,
        lambda_ = LAMBDA_,
        tau     = TAU,
        rho     = RHO,
        delta   = DELTA,
    )
    straggler_mgr = StragglerManager(
        delta       = DELTA,
        s_max       = 3,
        quorum_frac = 0.70,
        mad_k       = 2.5,
        w_min       = 2.0,
        rng         = rng,
    )

    print(f"\n[Config] {selector}")
    print(f"[Config] {straggler_mgr}")

    # ── Step 5: Run simulation ───────────────────────────────────────────────
    sim = FedMelanomaSimulation(
        clients       = clients,
        selector      = selector,
        straggler_mgr = straggler_mgr,
        n_select      = args.select,
        n_rounds      = args.rounds,
        rng           = rng,
    )
    sim.run()

    # ── Step 6: Diagnostic plots ─────────────────────────────────────────────
    generate_plots(sim, clients, output_path=args.output)

    # ── Step 7: Metrics summary ──────────────────────────────────────────────
    print_metrics_summary(sim, clients)


if __name__ == "__main__":
    main()
