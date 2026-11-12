"""
aggregation.py  ·  FedMelanoma v4 — Section 3.10
=================================================
Staleness-Weighted Aggregation with Diversity Bonus
----------------------------------------------------

Implements ONLY Sections 3.10.1 – 3.10.3 of the FedMelanoma v4 blueprint.

Section 3.10.1 — FedAvg baseline (reference / ablation):
    w^{t+1} = w^t + Σ_{i∈At} (n_i / n) · Δw_i
    Synchronous, staleness-blind, diversity-blind.

Section 3.10.2 — FedMelanoma aggregation weights:
    c̃_i = (n_i / n_A) · δ^{s_i} · (1 + γ · D_i)
    c_i  = c̃_i / Σ_{j∈At} c̃_j
    where n_A = Σ_{j∈At} n_j,  s_i = t − t_i^{gen}

Section 3.10.3 — Adaptive time window and straggler buffering:
    W_t = max( T̃_med + k · MAD(T_1,...,T_n),  W_min )
    Trigger:  (A) quorum  |A_t| ≥ ρ_q · C,  OR  (B) window expiry W_t
    Buffer:   stragglers → s_i=1, held for round t+1; discard after S_max rounds.

Canonical hyperparameters (extracted verbatim from selector.py / straggler.py):
    δ = 0.90  | γ = 1.0  | k = 2.5  | W_min = 2.0 s
    ρ_q = 0.70 | S_max = 3 | latency history window = 10 rounds
"""

from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ===========================================================================
# Canonical hyperparameters
# (Values taken from DELTA_DEFAULT, GAMMA, MAD_K_DEFAULT, etc. in the source.)
# ===========================================================================

# δ = 0.90  ─── staleness discount, UNIFIED across three subsystems:
#   • Aggregation weight c̃_i   (Section 3.10.2):  c̃_i ∝ δ^{s_i}
#   • Reliability update R_i   (Section 3.5):     R^{t+1} = ρ·R^t + (1-ρ)·δ^{s_i}
#   • Exposure update N_eff    (Section 3.6):     N_eff  += δ^{s_i}
# Concrete decay:  δ^0=1.000  δ^1=0.900  δ^2=0.810  δ^3=0.729  δ^∞=0.000
DELTA: float = 0.90

# γ = 1.0  ─── diversity amplification weight (same γ in selector score and c̃_i)
# Ensures consistent diversity weighting across both selection and aggregation.
GAMMA: float = 1.0

# --- Section 3.10.3: adaptive window hyperparameters ---
MAD_K: float       = 2.5   # k — MAD multiplier; W_t = T̃_med + 2.5·MAD
W_MIN: float       = 2.0   # W_min — minimum window (s); prevents collapse on fast clients
QUORUM_FRAC: float = 0.70  # ρ_q — quorum fraction (70 % of C=6 → 4 on-time clients)
S_MAX: int         = 3     # S_max — max rounds a buffered update survives before discard
WINDOW_HISTORY_LEN: int = 10  # rolling history length for T̃_med and MAD computation


# ===========================================================================
# Data structures (kept minimal — only fields required by Section 3.10)
# ===========================================================================

@dataclass
class ClientResponse:
    """
    Outcome of one client's round participation (produced by simulate_response
    in straggler.py; consumed here by manage_window and compute_aggregation_weights).

    Fields used by Section 3.10
    ---------------------------
    client_id      : Identifies the client.
    response_time  : Simulated latency (seconds); float('inf') if failed.
    is_straggler   : True when response_time > W_t.
    is_failed      : True when no gradient update was submitted.
    staleness      : s_i — rounds between gradient computation and aggregation.
                       0.0       → on-time
                       1.0       → freshly buffered straggler (assigned here)
                       2.0 / 3.0 → older buffered entry at drain time
                       float('inf') → failed (set by simulate_response)
    delta_weight   : δ^{s_i} — pre-computed staleness discount ∈ [0, 1].
    round_generated: Round t_i^{gen} at which the gradient was computed.
                     Used to recompute s_i = t_current − t_i^{gen} at drain time.
    """
    client_id:       int
    response_time:   float
    is_straggler:    bool
    is_failed:       bool
    staleness:       float
    delta_weight:    float
    round_generated: int


@dataclass
class BufferedUpdate:
    """
    A late-arriving straggler update held in the server async buffer
    (Section 3.10.3).  Eligible for inclusion in round t+1; discarded at age S_max.

    Fields
    ------
    client_id          : Identifies the client.
    response           : Original ClientResponse from simulate_response.
    buffer_entry_round : The round in which this update entered the buffer.
    """
    client_id:          int
    response:           ClientResponse
    buffer_entry_round: int

    def age(self, current_round: int) -> int:
        """Rounds spent in the buffer = current_round − buffer_entry_round."""
        return current_round - self.buffer_entry_round


# ===========================================================================
# Section 3.10.2 helpers — staleness arithmetic
# ===========================================================================

def compute_staleness(current_round: int, generation_round: int) -> int:
    """
    Staleness  s_i = t − t_i^{gen}  (Section 3.10.2).

    Measures how many rounds have elapsed between when the client computed
    its gradient (at global model state w^t) and when the server aggregates it.
    Example: gradient computed at t=5, aggregated at t=8 → s_i = 3.

    The δ^{s_i} discount in c̃_i bounds the staleness-induced convergence error
    to O(S_max²) (Section 3.12, Term 4).

    Parameters
    ----------
    current_round    : Round at which aggregation occurs.
    generation_round : Round at which this client computed its gradient.

    Returns
    -------
    int : s_i = max(0, current_round − generation_round)
    """
    return max(0, current_round - generation_round)


def apply_staleness_weight(staleness: float, delta: float = DELTA) -> float:
    """
    δ^{s_i} — unified staleness discount applied identically in three places
    (Section 3.5.4 "unified staleness semantics"):

        Aggregation (3.10.2):   c̃_i = (n_i/n_A) · δ^{s_i} · (1+γ·D_i)
        Reliability (3.5):      R^{t+1} = ρ·R^t + (1−ρ)·δ^{s_i}
        Exposure    (3.6):      N_eff  += δ^{s_i}

    WHY exponential (not linear):
      Linear decay max(0, 1−s/S_max) creates an abrupt cliff at S_max.
      Exponential δ^s is smooth, bounded, and interpretable; the same δ=0.90
      governs all three subsystems simultaneously, so a s=3 stale update is
      72.9 % useful everywhere at once — not arbitrarily in one place.

    Concrete values (δ = 0.90):
      s=0 → δ^0 = 1.000  (on-time:         full credit)
      s=1 → δ^1 = 0.900  (buffered:         10 % discount)
      s=2 → δ^2 = 0.810  (two-round delay:  19 % discount)
      s=3 → δ^3 = 0.729  (at S_max border:  27 % discount)
      s=∞ → 0.000        (failure:          zero contribution)

    Parameters
    ----------
    staleness : s_i ≥ 0.  float('inf') or negative → returns 0.0.
    delta     : Staleness discount δ (canonical: 0.90).

    Returns
    -------
    float : δ^{s_i} ∈ [0, 1].
    """
    if np.isinf(staleness) or staleness < 0:
        return 0.0          # δ^∞ = 0 — failed client contributes nothing
    return float(delta ** staleness)


# ===========================================================================
# Section 3.10.1 — FedAvg baseline  (ablation reference only)
# ===========================================================================

def fedavg_baseline_weights(
    on_time:    List[ClientResponse],
    client_map: Dict[int, object],       # client_id → Client (needs .n_samples)
) -> Dict[int, float]:
    """
    Standard FedAvg aggregation weights [McMahan et al. 2017] (Section 3.10.1):

        w^{t+1} = w^t + Σ_{i∈At} (n_i / n) · Δw_i

    Synchronous, staleness-blind, diversity-blind.
    Included purely for ablation comparison against the Section 3.10.2 formula.

    Parameters
    ----------
    on_time    : Arrived ClientResponse list for this round.
    client_map : Dict mapping client_id → Client with .n_samples attribute.

    Returns
    -------
    Dict[int, float] : {client_id: n_i/n}.  Empty dict when A_t is empty.
    """
    if not on_time:
        return {}

    # n = total samples across ALL arrived clients (no staleness, no diversity)
    n_total: float = sum(
        client_map[r.client_id].n_samples
        for r in on_time
        if r.client_id in client_map
    )
    if n_total == 0:
        return {}

    return {
        r.client_id: float(client_map[r.client_id].n_samples / n_total)
        for r in on_time
        if r.client_id in client_map
    }


# ===========================================================================
# Section 3.10.2 — FedMelanoma Aggregation Weights
# ===========================================================================

def compute_aggregation_weights(
    on_time:    List[ClientResponse],
    client_map: Dict[int, object],       # client_id → Client (needs .n_samples, .D)
    delta:      float = DELTA,
    gamma:      float = GAMMA,
) -> Dict[int, float]:
    """
    Compute normalised aggregation weights c_i for arrived client set A_t.

    Section 3.10.2 formula:

        c̃_i = (n_i / n_A) · δ^{s_i} · (1 + γ · D_i)
        c_i  = c̃_i / Σ_{j∈At} c̃_j

        where  n_A = Σ_{j∈At} n_j   (total samples of all arrived clients)

    Three-factor interpretation of c̃_i:

      (n_i / n_A)     — SIZE PROPORTIONALITY
                         FedAvg maximum-likelihood baseline. Clients with more
                         samples contribute proportionally more to the global
                         gradient update. Consistent with [McMahan et al. 2017].

      δ^{s_i}         — STALENESS DISCOUNT
                         Outdated gradients contribute less.
                         On-time (s=0):          δ^0 = 1.00  (full weight)
                         Buffered straggler (s=1): δ^1 = 0.90 (−10 %)
                         Two-round delay (s=2):   δ^2 = 0.81  (−19 %)
                         At S_max boundary (s=3): δ^3 = 0.729 (−27 %)

      (1 + γ · D_i)   — DIVERSITY AMPLIFICATION
                         Clients whose data distribution diverges from the global
                         centroid (high D_i) receive a boosted aggregation weight.
                         Same γ=1.0 as the selector score (Section 3.8.1) — this
                         is the aggregation-side complement to the selection-side
                         diversity bonus, providing CONSISTENT diversity weighting
                         across both selection and aggregation.
                         D_i=0 → multiplier=1.0 (no bonus)
                         D_i=1 → multiplier=2.0 (maximum bonus)

    WHY multiplicative (not additive):
      All three factors jointly determine contribution. A large, on-time, diverse
      client should receive maximum weight. Additive combination would allow a
      zero-staleness client to dominate even with no samples, or a tiny but
      diverse client to dominate regardless of staleness. Multiplication requires
      all conditions to be simultaneously satisfied.

    Normalisation:
      Σ_i c_i = 1 is guaranteed by dividing by the sum of c̃_i values.
      This is CRITICAL under 40–60 % straggler rates: the update magnitude is
      well-scaled regardless of how many clients arrive in A_t.

    Parameters
    ----------
    on_time    : Arrived ClientResponse list (on-time + drained buffer entries).
    client_map : Dict mapping client_id → Client with .n_samples and .D.
    delta      : Staleness discount δ (canonical: 0.90).
    gamma      : Diversity amplification γ (canonical: 1.0).

    Returns
    -------
    Dict[int, float] : {client_id: c_i}, sums to 1.  Empty dict when A_t = ∅.
    """
    if not on_time:
        return {}   # Empty A_t — no aggregation this round

    # ── n_A = total samples across ALL arrived clients ───────────────────────
    n_A: float = sum(
        client_map[r.client_id].n_samples
        for r in on_time
        if r.client_id in client_map
    )
    if n_A == 0:
        return {}   # Edge case: all arrived clients report zero samples

    # ── Compute unnormalised c̃_i = (n_i/n_A) · δ^{s_i} · (1 + γ·D_i) ──────
    c_tilde: Dict[int, float] = {}
    for resp in on_time:
        cid = resp.client_id
        if cid not in client_map:
            continue
        client = client_map[cid]

        # Factor 1: size proportionality  n_i / n_A
        size_weight: float = client.n_samples / n_A

        # Factor 2: staleness discount  δ^{s_i}
        staleness_weight: float = apply_staleness_weight(resp.staleness, delta)

        # Factor 3: diversity amplification  (1 + γ·D_i)
        diversity_bonus: float = 1.0 + gamma * client.D

        # Three-factor product — c̃_i
        c_tilde[cid] = size_weight * staleness_weight * diversity_bonus

    # ── Normalise: c_i = c̃_i / Σ_j c̃_j  →  guarantees Σ c_i = 1 ──────────
    total: float = sum(c_tilde.values())
    if total < 1e-12:
        return {}   # Numerical safety: all weights collapsed near zero

    return {cid: float(v / total) for cid, v in c_tilde.items()}


# ===========================================================================
# Section 3.10.3 — Adaptive Time Window + Straggler Buffer
# ===========================================================================

class StragglerAggregationWindow:
    """
    FedMelanoma v4 — Section 3.10.3: Adaptive Window + Straggler Buffer.

    Responsibilities
    ----------------
    manage_window()  : Main round-level entry point.
                       Classifies responses (on-time / straggler / failed),
                       updates W_t, drains buffer, checks quorum trigger.
    _update_window() : Recompute  W_t = max(T̃_med + k·MAD, W_min).
    _drain_buffer()  : Release eligible buffer entries; discard expired ones.

    Adaptive Window (Section 3.10.3)
    ---------------------------------
        W_t = max( T̃_med + k · MAD(T_1,...,T_n),  W_min )

        T̃_med : median of last 10 rounds' per-client response times.
        MAD   : median(|T_i − T̃_med|)  — 50 % breakdown-point scale estimator.
        k     = 2.5  (MAD_K, canonical).
        W_min = 2.0 s (W_MIN): floor that prevents degenerate W_t → 0.

    WHY median + MAD (not mean + std):
        Log-normal completion times have heavy right tails. One 60-second delay
        inflates σ enormously, pulling W_t upward and forcing the server to wait
        for that outlier every round. Median has a 50 % breakdown point (half the
        samples can be extreme without corrupting the estimate). MAD shares that
        property. Together they form the FL analog of the boxplot IQR fence.

    Aggregation triggers on EITHER (Section 3.10.3):
        (A) QUORUM:         |A_t| ≥ ρ_q · C  (ρ_q=0.70, C=6 → need 4 on-time)
        (B) WINDOW EXPIRY:  W_t seconds have elapsed since round start.

    Straggler buffering:
        Clients with  response_time > W_t  are BUFFERED with staleness s_i=1
        for inclusion in round t+1.  Not consumed within S_max=3 rounds → DISCARD.

    WHY buffer instead of drop:
        Telemedicine / rural nodes (high-straggler tier) are simultaneously the
        MOST frequent stragglers AND the MOST likely to hold underrepresented
        patient phenotypes (dark skin, single-contributor labels, overview imaging).
        Dropping every late update systematically excludes the most vulnerable
        populations from gradient influence — producing a biased global model.
        Buffering guarantees eventual contribution; δ^{s_i} discounting correctly
        down-weights the outdated gradients without silencing the client entirely.

    Canonical configuration (straggler.py constants):
        δ=0.90  |  S_max=3  |  ρ_q=0.70  |  k=2.5  |  W_min=2.0 s  |  history=10
    """

    def __init__(
        self,
        delta:       float = DELTA,
        s_max:       int   = S_MAX,
        quorum_frac: float = QUORUM_FRAC,
        mad_k:       float = MAD_K,
        w_min:       float = W_MIN,
        rng:         Optional[np.random.Generator] = None,
    ) -> None:
        self.delta       = delta        # δ=0.90: unified staleness discount
        self.s_max       = s_max        # S_max=3: max buffer age before discard
        self.quorum_frac = quorum_frac  # ρ_q=0.70: fraction for quorum trigger
        self.mad_k       = mad_k        # k=2.5: MAD multiplier in W_t formula
        self.w_min       = w_min        # W_min=2.0 s: floor for adaptive window
        self.rng         = rng or np.random.default_rng()

        # Rolling history of per-client response times (last 10 rounds).
        # deque with maxlen automatically evicts the oldest round once full.
        # T̃_med and MAD for W_t are computed over this history.
        self._latency_history: deque = deque(maxlen=WINDOW_HISTORY_LEN)

        # Straggler buffer: client_id → BufferedUpdate.
        # Holds late updates pending inclusion in the next eligible round.
        # Key overwrite: if a client is buffered twice before draining, only
        # the most recent entry is kept (consistent with "last write wins").
        self._buffer: Dict[int, BufferedUpdate] = {}

        # W_t initialised to W_min; grows as latency history accumulates.
        self.current_W_t: float = w_min

        # Per-round diagnostic metrics (used by simulation.py for plots)
        self.window_history:     List[float] = []   # W_t per round
        self.straggler_counts:   List[int]   = []   # new buffer entries per round
        self.on_time_counts:     List[int]   = []   # on-time + drained per round
        self.failed_counts:      List[int]   = []   # failures per round
        self.quorum_met_history: List[bool]  = []   # True when quorum triggered

    # ------------------------------------------------------------------
    # Section 3.10.3 — Adaptive Window Update
    # ------------------------------------------------------------------

    def _update_window(self) -> None:
        """
        Recompute W_t after extending the latency history with current-round
        response times:

            W_t = max( T̃_med + k · MAD(T_1,...,T_n),  W_min )
                = max( T̃_med + 2.5 · MAD,  2.0 )

        T̃_med: median of all arrival times in the rolling history buffer.
        MAD:   median(|T_i − T̃_med|)  — robust scale estimate with 50 %
               breakdown point.  Avoids σ inflation from heavy right tails.

        With k=2.5 MADs, the window accommodates ~97 % of normally-distributed
        delays. Log-normal tails are heavier, so the effective capture rate is
        lower — this is intentional: outlier clients become stragglers rather
        than stalling the entire aggregation.

        Called by manage_window() AFTER feeding the current round's times into
        _latency_history (order matters: update history first, then window).
        """
        if len(self._latency_history) < 2:
            # Fewer than 2 data points: stay at W_min to avoid degenerate MAD=0
            self.current_W_t = self.w_min
            return

        times         = np.array(self._latency_history)
        t_med: float  = float(np.median(times))                 # T̃_med
        mad:   float  = float(np.median(np.abs(times - t_med))) # MAD
        # Apply W_min floor: prevents collapse when all clients are very fast
        self.current_W_t = float(max(t_med + self.mad_k * mad, self.w_min))

    # ------------------------------------------------------------------
    # Section 3.10.3 — Main Window Manager (round-level entry point)
    # ------------------------------------------------------------------

    def manage_window(
        self,
        clients:       List,                        # selected Client objects this round
        responses:     Dict[int, ClientResponse],   # cid → ClientResponse
        current_round: int,                         # t (0-based round index)
        n_selected:    int,                         # C (for quorum computation)
    ) -> Tuple[List[ClientResponse], List[ClientResponse]]:
        """
        Full Section 3.10.3 straggler handling for one federated round.

        Algorithm (5 steps):

        Step 1 — Feed latency history
            Collect response_time for non-failed clients → _latency_history.

        Step 2 — Update adaptive window
            W_t = max(T̃_med + 2.5·MAD, 2.0).

        Step 3 — Classify responses
            FAILED    → discard; no buffer entry; δ^∞=0 handled in R_i/N_eff update.
            STRAGGLER → assign staleness s_i=1; buffer for round t+1.
            ON-TIME   → assign staleness s_i=0; include in A_t immediately.

        Step 4 — Drain buffer
            Entries from prior rounds with age ≥ 1  → recompute s_i, include in A_t.
            Entries with age ≥ S_max (=3)           → DISCARD (too stale).

        Step 5 — Quorum check
            quorum_threshold = ceil(ρ_q · C) = ceil(0.70 · 6) = 4.
            If |A_t| ≥ 4 → QUORUM trigger; else → WINDOW EXPIRY trigger.
            Either way, the function returns and aggregation proceeds.

        Parameters
        ----------
        clients       : Selected Client objects this round (for context/logging).
        responses     : cid → ClientResponse from simulate_response().
        current_round : Current round index t.
        n_selected    : C (number of clients selected this round).

        Returns
        -------
        on_time_responses   : A_t — list of ClientResponse for aggregation
                              (on-time this round + drained buffer entries).
        straggler_responses : Newly buffered ClientResponse list (logging / metrics).
        """
        # ── Step 1: Feed current-round response times into rolling history ───
        # Finite (non-inf) times only; failed clients don't contribute to W_t.
        arrived_times: List[float] = [
            r.response_time
            for r in responses.values()
            if not r.is_failed and not np.isinf(r.response_time)
        ]
        self._latency_history.extend(arrived_times)

        # ── Step 2: Recompute adaptive window W_t ───────────────────────────
        self._update_window()

        # ── Step 3: Classify each response into on-time / straggler / failed ─
        on_time:    List[ClientResponse] = []
        stragglers: List[ClientResponse] = []
        n_failed = 0

        for resp in responses.values():
            cid = resp.client_id

            # ---- FAILED: no update submitted; delta^inf = 0 ----------------
            if resp.is_failed:
                n_failed += 1
                # Selector will contract R_i via: R^{t+1} = ρ·R^t (quality=0)
                # N_eff is left unchanged (no manifold coverage from a failed update)
                print(
                    f"  [Window] C{cid}: ✗ FAILED — "
                    f"no update (δ^s=0.0, N_eff unchanged)"
                )
                continue

            # ---- STRAGGLER: arrived after W_t → buffer for round t+1 ------
            if resp.is_straggler:
                # Final staleness = 1 (buffered; will be included next round)
                # δ^1 = 0.90: 10 % discount on its eventual aggregation weight
                resp.staleness    = 1.0
                resp.delta_weight = apply_staleness_weight(resp.staleness, self.delta)

                # Store in buffer (overwrite prior entry if same client buffered twice)
                self._buffer[cid] = BufferedUpdate(
                    client_id          = cid,
                    response           = resp,
                    buffer_entry_round = current_round,
                )
                stragglers.append(resp)
                print(
                    f"  [Window] C{cid}: ⟳ LATE "
                    f"(t={resp.response_time:.2f}s > W_t={self.current_W_t:.2f}s) "
                    f"→ BUFFERED  s=1  δ^s={resp.delta_weight:.4f}"
                )

            # ---- ON-TIME: arrived within W_t → include in A_t now ----------
            else:
                # s_i = 0: full quality; δ^0 = 1.0 (no discount)
                resp.staleness    = 0.0
                resp.delta_weight = apply_staleness_weight(resp.staleness, self.delta)
                on_time.append(resp)
                print(
                    f"  [Window] C{cid}: ✓ ON-TIME "
                    f"(t={resp.response_time:.2f}s ≤ W_t={self.current_W_t:.2f}s) "
                    f"δ^s={resp.delta_weight:.4f}"
                )

        # ── Step 4: Drain buffer entries from prior rounds ───────────────────
        # Returns: drained (ready for A_t) + expired (discarded, age ≥ S_max)
        drained, expired = self._drain_buffer(current_round)
        on_time.extend(drained)   # Drained entries become part of A_t

        for exp in expired:
            print(
                f"  [Buffer] C{exp.client_id}: ✗ EXPIRED "
                f"(age={exp.age(current_round)} ≥ S_max={self.s_max}) → DISCARDED"
            )

        # ── Step 5: Quorum check — log which aggregation trigger fired ───────
        # quorum_threshold = ceil(ρ_q · C) = ceil(0.70 · 6) = 4
        quorum_threshold: int  = max(1, int(np.ceil(self.quorum_frac * n_selected)))
        quorum_met:       bool = len(on_time) >= quorum_threshold
        trigger: str = "QUORUM ✓" if quorum_met else "WINDOW EXPIRY"

        print(
            f"  [Window] W_t={self.current_W_t:.2f}s  "
            f"| |A_t|={len(on_time)}  "
            f"| new_buffered={len(stragglers)}  "
            f"| failed={n_failed}  "
            f"| quorum_threshold={quorum_threshold}  "
            f"| trigger={trigger}"
        )

        # ── Record per-round diagnostics (for simulation metrics / plots) ────
        self.window_history.append(self.current_W_t)
        self.straggler_counts.append(len(stragglers))
        self.on_time_counts.append(len(on_time))
        self.failed_counts.append(n_failed)
        self.quorum_met_history.append(quorum_met)

        return on_time, stragglers

    # ------------------------------------------------------------------
    # Section 3.10.3 — Buffer Drain Logic
    # ------------------------------------------------------------------

    def _drain_buffer(
        self,
        current_round: int,
    ) -> Tuple[List[ClientResponse], List[BufferedUpdate]]:
        """
        Process the async straggler buffer at the start of each round.

        Draining rules (Section 3.10.3):

          age ≥ S_max (=3) → DISCARD
              Update is too stale to contribute meaningfully.
              The convergence bound (Section 3.12) requires updates within
              S_max rounds; beyond that the bias term O(δ^{S_max}) is acceptable
              but the gradient itself is no longer representative.

          age ≥ 1          → DRAIN
              Recompute final staleness from generation_round (not buffer_entry_round).
              Update δ^{s_i} discount accordingly; include in A_t.

          age < 1          → KEEP
              Just entered this round (edge-case guard for concurrent scenarios).

        WHY recompute staleness at drain time (not at buffer entry):
            When buffered at round t, the update has intended staleness=1 for t+1.
            But if not consumed at t+1, its true staleness at drain time t' is
            s_i = t' − t_i^{gen}.  Recomputing ensures δ^{s_i} accurately reflects
            the gradient's age at the ACTUAL aggregation moment, not the intended one.

        Returns
        -------
        drained  : List[ClientResponse] ready for inclusion in A_t.
        expired  : List[BufferedUpdate] that were discarded (age ≥ S_max).
        """
        to_drain: List[ClientResponse]     = []
        expired:  List[BufferedUpdate]     = []
        to_keep:  Dict[int, BufferedUpdate] = {}

        for cid, buf in list(self._buffer.items()):
            age = buf.age(current_round)

            if age >= self.s_max:
                # DISCARD: too stale; bounding convergence error requires this cutoff
                expired.append(buf)

            elif age >= 1:
                # DRAIN: recompute exact staleness from original generation_round
                # s_i = current_round − t_i^{gen}  (Section 3.10.2)
                final_s: int = compute_staleness(
                    current_round, buf.response.round_generated
                )
                buf.response.staleness    = float(final_s)
                buf.response.delta_weight = apply_staleness_weight(final_s, self.delta)
                to_drain.append(buf.response)

                print(
                    f"  [Buffer] C{cid}: ↩ DRAINED "
                    f"(gen_round={buf.response.round_generated}, "
                    f"current={current_round}, s={final_s}, "
                    f"δ^s={buf.response.delta_weight:.4f})"
                )
            else:
                # KEEP: just buffered this round — not yet eligible
                to_keep[cid] = buf

        self._buffer = to_keep   # Remove drained and expired entries from buffer
        return to_drain, expired

    # ------------------------------------------------------------------
    # Utility / diagnostics
    # ------------------------------------------------------------------

    def get_buffer_state(self) -> Dict[int, Dict]:
        """Snapshot of current buffer contents (debug / logging)."""
        return {
            cid: {
                "buffer_entry_round": buf.buffer_entry_round,
                "round_generated":    buf.response.round_generated,
                "response_time":      buf.response.response_time,
            }
            for cid, buf in self._buffer.items()
        }

    def flush_buffer(self) -> None:
        """Hard-clear the buffer (e.g., between independent simulation runs)."""
        self._buffer.clear()

    def summary(self) -> Dict[str, float]:
        """Aggregate statistics over all completed rounds (for metrics report)."""
        if not self.straggler_counts:
            return {}
        return {
            "total_rounds":   len(self.straggler_counts),
            "mean_on_time":   float(np.mean(self.on_time_counts)),
            "mean_straggler": float(np.mean(self.straggler_counts)),
            "mean_failed":    float(np.mean(self.failed_counts)),
            "mean_W_t":       float(np.mean(self.window_history)),
            "quorum_rate":    float(np.mean(self.quorum_met_history)),
        }

    def __repr__(self) -> str:
        return (
            f"StragglerAggregationWindow("
            f"δ={self.delta}, S_max={self.s_max}, "
            f"ρ_q={self.quorum_frac}, k={self.mad_k}, W_min={self.w_min})"
        )