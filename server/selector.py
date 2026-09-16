"""
selector.py  ·  FedMelanoma v4 — Exposure-Aware Client Selection Engine
=======================================================================
Implements the canonical selector score (Section 3.8) and Boltzmann
probabilistic sampling (Section 3.9) exactly as specified in the v4 blueprint.

Core formula (Section 3.8.1):

    score_i(t) = R_i(t) · (1 + γ · D_i / (1 + λ · N_i_eff(t)))

WHY multiplicative gating (not additive):
  An additive score α·R_i + β·D_i allows a high-diversity, zero-reliability
  client to receive a high score. The server would repeatedly attempt to select
  a client that never responds — wasting a selection slot each round. The
  multiplicative structure makes R_i a necessary condition: if R_i ≈ 0, the
  score approaches 0 regardless of diversity. A client with valuable data that
  never responds is worthless to the aggregation pipeline.

WHY Boltzmann sampling (not greedy top-K):
  Greedy top-K has three failure modes in async FL (Section 3.9.1):
    1. Async stampede: same K clients selected every round
    2. Starvation: low-score telemedicine nodes permanently excluded
    3. Brittleness: single straggler event causes abrupt selection changes
  Boltzmann guarantees p_i > 0 for all clients, respects score ranking,
  and maximises entropy subject to expected utility (Jaynes 1957).

Canonical hyperparameters (Section 3 of FedMelanoma v4 blueprint):
  ρ = 0.85  (EMA decay for reliability signal R_i)
  δ = 0.90  (staleness discount — unified across R_i, N_eff, and aggregation)
  λ = 0.05  (exposure attenuation rate, controls U_eff half-life)
  γ = 1.0   (diversity amplification weight)
  τ = 0.20  (Boltzmann temperature, calibrated to score scale [0, 2])
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Canonical hyperparameters — Section 3.8.1 and 3.9.2
# ---------------------------------------------------------------------------
RHO_DEFAULT    = 0.85   # EMA decay for R_i (half-life ≈ 4.27 rounds)
DELTA_DEFAULT  = 0.90   # Staleness discount — same δ in R_i, N_eff, aggregation
LAMBDA_DEFAULT = 0.05   # Exposure attenuation; halves diversity bonus at N_eff=20
GAMMA_DEFAULT  = 1.0    # Diversity amplification; score ∈ [0, 2] with this value
TAU_DEFAULT    = 0.20   # Boltzmann τ; at gap=0.8 → p1/p2 = e^4 ≈ 54.6×


# ===========================================================================
# Client dataclass
# ===========================================================================

@dataclass
class Client:
    """
    Represents a single federated hospital node.

    All per-round state (R, N_eff) is mutated in-place by the Selector.
    History lists are appended each round for post-hoc analysis and plotting.

    Fields
    ------
    client_id    : Unique integer identifier (0-indexed).
    n_samples    : Local dataset size n_i. Used for FedAvg-style proportional
                   aggregation weighting (Section 3.10.2).
    R            : Reliability signal R_i ∈ [0, 1]. EMA of staleness-adjusted
                   update quality. Initialised to 1.0 ("fresh optimism") per
                   blueprint Section 3.5 — all clients start trusted.
    D            : Diversity score D_i ∈ [0, 1]. STATIC OT-derived prior
                   (Section 3.2). Computed once at client registration from
                   EMD between client's metadata distribution and global centroid.
                   Never changes during training — it is a property of the
                   hospital's patient population, not of training progress.
    N_eff        : Effective interaction count N_i_eff(t). Staleness-discounted
                   cumulative sum of manifold interactions (Section 3.6).
                   Starts at 0; grows monotonically; unbounded from above.
    latency_mu   : Mean of log-normal compute delay (log-seconds). From
                   blueprint Section 4.2.2: reliable=1.39, moderate=2.77,
                   high_straggler=4.09.
    latency_sigma: Std of log-normal compute delay. reliable=0.30, moderate=0.50,
                   high_straggler=0.80.
    straggler_prob: Nominal per-round failure probability. Actual completion
                   probability is Beta(a_i, b_i) sampled fresh each round.
    tier         : 'reliable' | 'moderate' | 'high_straggler' — maps to Beta
                   parameters in StragglerManager. Used for logging and colour
                   coding in plots.

    History tracking (appended each round for metrics):
    score_history         : score_i(t) at each round.
    selected_rounds       : rounds in which this client was selected.
    staleness_history     : s_i per round (-1.0 = failure / not selected).
    contribution_history  : aggregation weight c_i per round (0 if not in A_t).
    r_history             : R_i(t) trajectory.
    n_eff_history         : N_eff(t) trajectory.
    """
    client_id:      int
    n_samples:      int
    R:              float = 1.0      # R_i^(0) = 1.0 per blueprint
    D:              float = 0.5      # D_i: static OT diversity prior
    N_eff:          float = 0.0      # N_eff^(0) = 0 (no prior interactions)
    latency_mu:     float = 1.39     # Default: reliable tier
    latency_sigma:  float = 0.30
    straggler_prob: float = 0.05
    tier:           str   = "reliable"

    # Mutable history containers — use field() to avoid shared mutable defaults
    score_history:       List[float] = field(default_factory=list)
    selected_rounds:     List[int]   = field(default_factory=list)
    staleness_history:   List[float] = field(default_factory=list)
    contribution_history:List[float] = field(default_factory=list)
    r_history:           List[float] = field(default_factory=list)
    n_eff_history:       List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Clamp to valid ranges at construction
        self.R     = float(np.clip(self.R,     0.0, 1.0))
        self.D     = float(np.clip(self.D,     0.0, 1.0))
        self.N_eff = float(max(0.0, self.N_eff))

    def __repr__(self) -> str:
        return (
            f"Client(id={self.client_id}, tier={self.tier}, "
            f"n={self.n_samples}, R={self.R:.3f}, D={self.D:.3f}, "
            f"N_eff={self.N_eff:.2f})"
        )


# ===========================================================================
# Selector
# ===========================================================================

class Selector:
    """
    FedMelanoma v4 — Canonical Exposure-Aware Client Selector.

    Public interface
    ----------------
    compute_score(client)
        → float: canonical v4 score in [0, 2].
    boltzmann_selection(clients, n_select, temperature, rng)
        → List[Client]: probabilistically selected subset.
    greedy_selection(clients, n_select)
        → List[Client]: deterministic top-K (for ablation comparison only).
    update_reliability(client, staleness)
        → None: EMA update of R_i given round staleness.
    update_exposure(client, staleness)
        → None: increment N_eff by δ^{s_i}.
    update_client_state(client, staleness, round_idx)
        → None: joint R_i + N_eff update after a round outcome.

    Design notes
    ------------
    All state mutations go through this class — Client objects are passive data
    containers. This mirrors the server-side architecture in the blueprint where
    selector_math.py owns the update logic.
    """

    def __init__(
        self,
        gamma:   float = GAMMA_DEFAULT,
        lambda_: float = LAMBDA_DEFAULT,
        tau:     float = TAU_DEFAULT,
        rho:     float = RHO_DEFAULT,
        delta:   float = DELTA_DEFAULT,
    ) -> None:
        self.gamma   = gamma     # Diversity amplification weight γ
        self.lambda_ = lambda_   # Exposure attenuation rate λ
        self.tau     = tau       # Boltzmann temperature τ
        self.rho     = rho       # EMA decay ρ for R_i
        self.delta   = delta     # Staleness discount δ (unified across the system)

    # ------------------------------------------------------------------
    # 1. Canonical selector score  (Section 3.8)
    # ------------------------------------------------------------------

    def compute_score(self, client: Client) -> float:
        """
        Compute the v4 canonical selector score:

            score_i(t) = R_i(t) · (1 + γ · D_i / (1 + λ · N_i_eff(t)))

        Factored as:
            score_i = R_i · (1 + γ · U_eff)

        where:
            U_eff = D_i / (1 + λ·N_eff)   [Section 3.7 — marginal utility]

        Structural breakdown:
          R_i(t)           : RELIABILITY GATE. If this client never responds,
                             R_i → 0 and score → 0 regardless of diversity.
                             Reliability is a necessary condition for selection.
          γ · D_i          : RAW DIVERSITY BONUS. Higher diversity → stronger
                             initial selection preference, driving the model to
                             explore minority-phenotype manifolds early.
          1 + λ·N_eff      : EXPOSURE DENOMINATOR. Each effective interaction
                             attenuates the diversity bonus. This implements
                             diminishing marginal returns: after 20 on-time
                             interactions (N_eff=20), U_eff is halved.
          (1 + γ·U_eff)    : BOUNDED MULTIPLIER ∈ [1, 2] for γ=1, D∈[0,1].
                             Score is therefore bounded: score_i ∈ [0, 2].

        Asymptotic behaviour (Section 3.8.4):
          As N_eff → ∞: score_i → R_i·1 = R_i.
          The system transitions from diversity-weighted exploration to
          reliability-weighted exploitation automatically — no explicit
          annealing schedule needed.

        Returns
        -------
        float : score_i ∈ [0, 2].
        """
        # Marginal utility attenuation U_eff (Section 3.7)
        # Denominator 1 + λ·N_eff ≥ 1, so U_eff ∈ [0, D_i] ⊆ [0, 1]
        U_eff = client.D / (1.0 + self.lambda_ * client.N_eff)

        # Canonical multiplicative gating: reliability gates the diversity bonus
        score = client.R * (1.0 + self.gamma * U_eff)

        # Theoretical bounds: R_i ∈ [0,1], multiplier ∈ [1,2] → score ∈ [0,2]
        return float(np.clip(score, 0.0, 2.0))

    def compute_all_scores(self, clients: List[Client]) -> np.ndarray:
        """Compute scores for a list of clients, returning an ndarray."""
        return np.array([self.compute_score(c) for c in clients])

    def compute_marginal_utility(self, client: Client) -> float:
        """
        Compute U_eff = D_i / (1 + λ·N_eff) in isolation.
        Useful for logging and the N_eff decay plot.
        """
        return float(client.D / (1.0 + self.lambda_ * client.N_eff))

    # ------------------------------------------------------------------
    # 2. Boltzmann (softmax) selection  (Section 3.9)
    # ------------------------------------------------------------------

    def boltzmann_selection(
        self,
        clients:     List[Client],
        n_select:    int,
        temperature: Optional[float] = None,
        rng:         Optional[np.random.Generator] = None,
    ) -> List[Client]:
        """
        Probabilistic client selection via Boltzmann (softmax) sampling:

            p_i(t) = exp(score_i / τ) / Σ_j exp(score_j / τ)

        Selection is WITHOUT REPLACEMENT: each sampled client is removed from
        the pool before the next draw. This prevents a single client from being
        selected multiple times in one round (blueprint Section 3.9.2).

        WHY τ = 0.20 (not 1.0 or higher):
          With score ∈ [0, 2], a score gap of Δ=0.8 between the best and second
          client gives:  p1/p2 = exp(0.8/0.20) = e^4 ≈ 54.6
          This is "strong exploitation" — the server strongly prefers high-scoring
          clients but still gives every client a non-zero chance. At τ→0 this
          collapses to greedy; at τ→∞ it becomes uniform random. τ=0.20 sits
          at the appropriate exploitation level for this score scale.

        Numerical stability:
          Raw logits = score/τ. For score=2.0, τ=0.20 → logit=10 → exp=22026.
          We subtract max(logits) before exp (standard log-sum-exp trick).
          This does NOT change the resulting probabilities (translation invariance
          of softmax) but prevents float32 overflow.

        Parameters
        ----------
        clients     : Pool of all K available clients.
        n_select    : Number of clients to select (C=6 per blueprint).
        temperature : Override τ for ablation studies.
        rng         : numpy Generator for reproducibility.

        Returns
        -------
        List[Client] of length min(n_select, len(clients)).
        """
        tau = temperature if temperature is not None else self.tau
        if rng is None:
            rng = np.random.default_rng()

        n_select = min(n_select, len(clients))
        pool     = list(clients)   # Mutable shallow copy for without-replacement draws
        selected = []

        # Pre-compute all scores for debug logging (before any draws)
        all_scores = {c.client_id: self.compute_score(c) for c in pool}
        score_dbg  = "  ".join(f"C{cid}={s:.4f}" for cid, s in sorted(all_scores.items()))
        print(f"    [Selector] All scores: {score_dbg}")

        for draw_idx in range(n_select):
            if not pool:
                break

            # Recompute scores over remaining pool (N_eff unchanged within a round)
            scores = np.array([self.compute_score(c) for c in pool])

            # --- Numerically stable softmax (log-sum-exp trick) ---
            logits         = scores / tau
            logits_stable  = logits - logits.max()      # subtract max → no overflow
            exp_logits     = np.exp(logits_stable)
            probs          = exp_logits / exp_logits.sum()

            # --- Sample one index without replacement ---
            idx = int(rng.choice(len(pool), p=probs))
            selected.append(pool[idx])
            pool.pop(idx)

        return selected

    # ------------------------------------------------------------------
    # 3. Greedy top-K (ablation baseline only)  (Section 3.9.1)
    # ------------------------------------------------------------------

    def greedy_selection(
        self,
        clients:  List[Client],
        n_select: int,
    ) -> List[Client]:
        """
        Deterministic top-K selection for ablation comparison (Plot 6).

        This is what FedMelanoma v4 does NOT use in production.
        Three failure modes (blueprint Section 3.9.1):
          1. Async stampede — same K clients every round
          2. Starvation — low-score telemedicine nodes never selected
          3. Brittle — single straggler event causes abrupt set change

        Used here only to produce the comparison in the ablation plot.
        """
        scored = sorted(clients, key=lambda c: self.compute_score(c), reverse=True)
        return scored[:min(n_select, len(scored))]

    # ------------------------------------------------------------------
    # 4. Reliability update  (Section 3.5)
    # ------------------------------------------------------------------

    def update_reliability(self, client: Client, staleness: float) -> None:
        """
        Update R_i via EMA of staleness-adjusted update quality:

            R_i^{t+1} = ρ · R_i^t + (1 - ρ) · δ^{s_i}

        WHY EMA (not simple average):
          The EMA with ρ=0.85 has effective half-life = -log(2)/log(0.85) ≈ 4.27
          rounds. This is intentional: transient failures (1–2 rounds of outage)
          have a visible but recoverable effect on R_i, while persistent failures
          (5+ rounds) drive R_i below useful selection thresholds. A client that
          fails 5 times in a row goes from R=1.0 → 0.444 (see blueprint Table
          Section 3.5.3). It recovers to R≈0.75 within 5 subsequent on-time
          rounds — critical for not permanently penalising rural clinics.

        WHY δ^{s_i} as the quality signal:
          The same staleness parameter δ appears in:
            - R_i update  (this function)
            - N_eff increment  (update_exposure)
            - Aggregation weight c_i  (Section 3.10.2)
          This "unified staleness semantics" (Section 3.5.4) means a round-3
          stale update is δ^3=0.729 useful in all three subsystems simultaneously.
          Interpretive consistency is a deliberate design feature.

        Staleness conventions:
          s=0         → on-time, full quality    → δ^0 = 1.0
          s=1         → buffered straggler       → δ^1 = 0.90
          s=2         → two-round delay          → δ^2 = 0.81
          s=∞ / fail  → no update received       → quality = 0.0
                        R_i contracts: R^{t+1} = ρ·R^t (multiplicative decay)

        Parameters
        ----------
        client    : Client whose reliability to update.
        staleness : s_i for this round. Pass float('inf') for failures.
        """
        if np.isinf(staleness):
            # Complete failure: δ^∞ = 0, so R_i contracts by factor ρ
            # This preserves the distinction between total failure (R shrinks fast)
            # and partial staleness (R shrinks slowly proportional to δ^s)
            quality = 0.0
        else:
            quality = float(self.delta ** staleness)   # δ^{s_i} ∈ (0, 1]

        # EMA update: new observation receives weight (1-ρ), history weight ρ
        client.R = self.rho * client.R + (1.0 - self.rho) * quality
        client.R = float(np.clip(client.R, 0.0, 1.0))  # Bounded by construction

    # ------------------------------------------------------------------
    # 5. Exposure update  (Section 3.6)
    # ------------------------------------------------------------------

    def update_exposure(self, client: Client, staleness: float) -> None:
        """
        Increment N_i_eff by the staleness-discounted interaction credit:

            N_i_eff(t) = Σ_{τ=1}^{t} δ^{s_i^(τ)}

        This accumulates the effective number of successful manifold interactions —
        the optimizer's "coverage" of client i's gradient manifold.

        WHY staleness-discounted (not raw count):
          A stale update contributes δ^{s_i} to the global model (via c_i in
          aggregation). It should also contribute δ^{s_i} to N_eff — the exposure
          counter should reflect effective manifold coverage, not raw selection
          count. A sequence of 5 stale updates (s=2 each) contributes
          5 × 0.81 = 4.05 to N_eff, less than 5 on-time updates which give 5.0.
          This is correct: the optimizer has less coverage from stale gradients.

        WHY unbounded (no decay):
          N_eff is a cumulative sum — it never decreases. This models the fact
          that gradient manifold coverage is not "forgotten" between rounds.
          Once the model has trained on a client's distribution, that coverage
          persists in the weights. The attenuation via (1 + λ·N_eff) in U_eff
          provides the functional effect of diminishing returns.

        On failure (s=∞): δ^∞ = 0, N_eff unchanged.
        On non-selection: N_eff unchanged (client did not interact this round).

        Parameters
        ----------
        client    : Client whose exposure count to update.
        staleness : s_i for this round. Pass float('inf') for failures.
        """
        if np.isinf(staleness):
            increment = 0.0   # No manifold coverage from a failed update
        else:
            increment = float(self.delta ** staleness)   # δ^{s_i} ∈ (0, 1]

        client.N_eff += increment
        # N_eff is unbounded above — it is a monotone cumulative sum

    # ------------------------------------------------------------------
    # 6. Joint state update (R_i + N_eff + history)  (Section 4 Phase 6)
    # ------------------------------------------------------------------

    def update_client_state(
        self,
        client:      Client,
        staleness:   float,
        round_idx:   int,
        was_selected: bool = True,
    ) -> None:
        """
        Jointly update R_i and N_eff after a round outcome, and append
        to per-client history lists for analysis.

        Called at the end of each round for every client in S^t (selected set).
        Non-selected clients are NOT updated (blueprint Section 4.3.2):
          "For non-selected clients: R^{t+1} = R^t, N_eff^{t+1} = N_eff^t"

        Parameters
        ----------
        client       : Client to update.
        staleness    : s_i for this round (float('inf') = failure).
        round_idx    : Current round index (0-based, for history).
        was_selected : If False, record history only (no R/N_eff update).
        """
        if was_selected:
            self.update_reliability(client, staleness)
            self.update_exposure(client, staleness)
            client.selected_rounds.append(round_idx)

        # Record history regardless — allows plotting full trajectories
        finite_s = staleness if not np.isinf(staleness) else -1.0
        client.staleness_history.append(finite_s)
        client.r_history.append(client.R)
        client.n_eff_history.append(client.N_eff)
        client.score_history.append(self.compute_score(client))

    # ------------------------------------------------------------------
    # 7. Boltzmann probability vector (for metrics / logging)
    # ------------------------------------------------------------------

    def boltzmann_probs(
        self,
        clients:     List[Client],
        temperature: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute the full Boltzmann probability vector over all clients
        (before any without-replacement draws).

        Returns
        -------
        np.ndarray of shape (K,), sums to 1.
        """
        tau    = temperature if temperature is not None else self.tau
        scores = self.compute_all_scores(clients)
        logits = scores / tau
        logits -= logits.max()
        probs   = np.exp(logits)
        return probs / probs.sum()

    # ------------------------------------------------------------------
    # 8. Fairness / entropy metrics
    # ------------------------------------------------------------------

    def selection_entropy(self, clients: List[Client]) -> float:
        """
        Compute Shannon entropy H(p) = -Σ p_i log(p_i) of the Boltzmann
        selection distribution.

        H = log(K) → uniform selection (maximum fairness, no score information)
        H = 0      → deterministic (greedy, all mass on one client)

        Used as fairness metric in the simulation (Section 5 of blueprint).
        Higher entropy → more equitable distribution across client tiers.

        Returns
        -------
        float : H(p) ≥ 0.
        """
        probs = self.boltzmann_probs(clients)
        eps   = 1e-12   # Avoid log(0)
        return float(-np.sum(probs * np.log(probs + eps)))

    # ------------------------------------------------------------------
    # Utility: score breakdown for logging
    # ------------------------------------------------------------------

    def score_breakdown(self, client: Client) -> Dict[str, float]:
        """
        Return a dict of all intermediate quantities for one client.
        Used for detailed per-round logging.
        """
        U_eff = self.compute_marginal_utility(client)
        score = self.compute_score(client)
        return {
            "client_id":  client.client_id,
            "R_i":        round(client.R, 4),
            "D_i":        round(client.D, 4),
            "N_eff":      round(client.N_eff, 4),
            "U_eff":      round(U_eff, 4),
            "multiplier": round(1.0 + self.gamma * U_eff, 4),
            "score":      round(score, 4),
        }

    def __repr__(self) -> str:
        return (
            f"Selector(γ={self.gamma}, λ={self.lambda_}, "
            f"τ={self.tau}, ρ={self.rho}, δ={self.delta})"
        )
