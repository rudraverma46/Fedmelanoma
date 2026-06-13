"""
straggler.py  ·  FedMelanoma v4 — Straggler Detection and Management
=====================================================================
Implements the complete asynchronous straggler handling pipeline described in
blueprint Sections 3.10 and 4.2.2, including:

  • Per-client response simulation via log-normal latency + Beta completion
  • Adaptive aggregation window  W_t = max(T̃_med + k·MAD, W_min), k=2.5
  • Buffered late-update management (s_i = 1 for next round inclusion)
  • Staleness computation and δ^{s_i} discount for aggregation weights
  • Quorum-or-window-expiry aggregation trigger

Design rationale
----------------
The straggler model captures a structural correlation endemic to federated
healthcare: resource-constrained clients (telemedicine, rural clinics) are
simultaneously the MOST likely to be stragglers AND the MOST likely to carry
underrepresented patient phenotypes (dark skin, single-contributor labels,
clinical-overview imaging). A naive synchronous FL system that drops all late
updates therefore systematically excludes the most vulnerable patient populations
from gradient influence — producing models biased against them. The buffered
async design ensures these clients always eventually contribute, with staleness
discounting that correctly weights their outdated gradients.

WHY log-normal for latency (Section 4.2.2):
  Compute delays in distributed systems accumulate multiplicatively: a 2× slow
  CPU combined with a 2× congested network produces a 4× total delay. Products
  of independent positive random variables converge to log-normal by the Central
  Limit Theorem on the log scale. Log-normal correctly models the right-skewed
  distribution with heavy tails seen in real federated deployments.

WHY Beta for straggler probability (Section 4.2.2):
  The per-round completion probability of a client is not fixed — it varies
  with local workload, network quality, and power availability. Beta(a, b) is
  the natural prior for a probability value. The three tiers correspond to:
    Reliable:       Beta(19, 1)  → mean=0.95, concentrated near 1
    Moderate:       Beta(7,  3)  → mean=0.70, moderate spread
    High-Straggler: Beta(2,  3)  → mean=0.40, skewed toward failure

WHY median + MAD for W_t (not mean + std):
  Log-normal completion times have heavy right tails. A single 60-second delay
  would inflate σ dramatically, pushing W_t upward to wait for that outlier in
  every subsequent round. Median is O(1) robust to outliers. MAD (median absolute
  deviation) has a 50% breakdown point — up to half the observations can be
  extreme outliers without pulling the scale estimate. The window formula
  W_t = T̃_med + 2.5·MAD is the FL analog of the boxplot IQR fence.

Canonical configuration (Section 3.10.3 + 4.2.2):
  k = 2.5          (MAD multiplier)
  W_min = 2.0 s    (minimum window prevents degenerate collapse)
  ρ_q = 0.70       (quorum fraction — aggregate on ≥70% arrival)
  S_max = 3        (buffer discard threshold)
  δ = 0.90         (staleness discount — unified with Selector)
"""

from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Canonical configuration constants  (Section 3.10.3)
# ---------------------------------------------------------------------------
S_MAX_DEFAULT         = 3      # Max rounds a buffered update survives before discard
QUORUM_FRAC_DEFAULT   = 0.70   # Fraction of selected clients needed for quorum trigger
MAD_K_DEFAULT         = 2.5    # MAD multiplier for adaptive window
W_MIN_DEFAULT         = 2.0    # Minimum aggregation window (seconds)
DELTA_DEFAULT         = 0.90   # Staleness discount (matches selector.py)
WINDOW_HISTORY_LEN    = 10     # Rolling latency history length for W_t computation

# Tier Beta parameters (Section 4.2.2)
BETA_PARAMS: Dict[str, Tuple[float, float]] = {
    "reliable":        (19.0, 1.0),   # mean = 0.95, very concentrated
    "moderate":        ( 7.0, 3.0),   # mean = 0.70, moderate spread
    "high_straggler":  ( 2.0, 3.0),   # mean = 0.40, skewed toward failure
}


# ===========================================================================
# ClientResponse dataclass — outcome of one simulated round response
# ===========================================================================

@dataclass
class ClientResponse:
    """
    Encapsulates the outcome of a single client's round participation.

    Fields
    ------
    client_id       : Identifies the client.
    response_time   : Simulated latency draw (seconds). float('inf') if failed.
    is_straggler    : True if arrived after W_t (late but not failed).
    is_failed       : True if no update was submitted (completion check failed).
    staleness       : s_i for this response.
                        0.0       → on-time
                        1.0       → buffered straggler (assigned by manage_window)
                        2.0, 3.0  → older buffered update at drain time
                        float('inf') → failed (set by simulate_response)
    delta_weight    : δ^{s_i} ∈ [0, 1]. Staleness discount for aggregation.
    round_generated : Round index at which this client computed its gradient.
                      Used to track staleness when draining the buffer.
    """
    client_id:        int
    response_time:    float
    is_straggler:     bool
    is_failed:        bool
    staleness:        float
    delta_weight:     float
    round_generated:  int

    def __repr__(self) -> str:
        status = "FAILED" if self.is_failed else ("STRAGGLER" if self.is_straggler else "ON-TIME")
        return (
            f"ClientResponse(C{self.client_id}, {status}, "
            f"t={self.response_time:.2f}s, s={self.staleness}, "
            f"δ^s={self.delta_weight:.4f})"
        )


# ===========================================================================
# BufferedUpdate dataclass — holds a late update across round boundaries
# ===========================================================================

@dataclass
class BufferedUpdate:
    """
    A straggler update stored in the server's async buffer.

    Late updates are buffered with staleness s=1 for inclusion in round t+1.
    Updates not consumed within S_max rounds are discarded (Section 3.10.3).

    Fields
    ------
    client_id           : Identifies the client.
    response            : Original ClientResponse from simulate_response.
    buffer_entry_round  : The round in which this update was placed in buffer.
    """
    client_id:          int
    response:           ClientResponse
    buffer_entry_round: int

    def age(self, current_round: int) -> int:
        """Number of rounds this update has spent in the buffer."""
        return current_round - self.buffer_entry_round


# ===========================================================================
# StragglerManager
# ===========================================================================

class StragglerManager:
    """
    FedMelanoma v4 — Asynchronous Straggler Management System.

    Responsibilities:
      1. simulate_response  : Per-client latency + straggler event simulation
      2. compute_staleness  : s_i = current_round - generation_round
      3. apply_staleness_weight : δ^{s_i} for aggregation weight formula
      4. manage_window      : Partition arrived clients into on-time / buffered;
                              drain eligible buffer; update adaptive W_t

    Thread safety note:
      This implementation is single-threaded (designed for simulation). In the
      production Flower server (server.py), the straggler detector runs in an
      asyncio background task and uses threading.Lock on _buffer mutations.
    """

    def __init__(
        self,
        delta:        float = DELTA_DEFAULT,
        s_max:        int   = S_MAX_DEFAULT,
        quorum_frac:  float = QUORUM_FRAC_DEFAULT,
        mad_k:        float = MAD_K_DEFAULT,
        w_min:        float = W_MIN_DEFAULT,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.delta       = delta        # Staleness discount (unified with Selector)
        self.s_max       = s_max        # Max rounds before buffer discard
        self.quorum_frac = quorum_frac  # Fraction needed for quorum trigger
        self.mad_k       = mad_k        # MAD multiplier
        self.w_min       = w_min        # Minimum window (prevents W_t → 0)
        self.rng         = rng or np.random.default_rng()

        # Rolling history of individual response times (last 10 rounds)
        # Used to compute T̃_med and MAD for the adaptive window update
        self._latency_history: deque = deque(maxlen=WINDOW_HISTORY_LEN)

        # Straggler buffer: client_id → BufferedUpdate
        # Holds late updates pending inclusion in the next eligible round
        self._buffer: Dict[int, BufferedUpdate] = {}

        # Adaptive window W_t (seconds) — initialised to w_min
        # Will grow once latency history accumulates
        self.current_W_t: float = w_min

        # Per-round monitoring metrics
        self.window_history:    List[float] = []
        self.straggler_counts:  List[int]   = []
        self.on_time_counts:    List[int]   = []
        self.failed_counts:     List[int]   = []
        self.quorum_met_history:List[bool]  = []

    # ------------------------------------------------------------------
    # 1. Simulate client response  (Section 4.2.2)
    # ------------------------------------------------------------------

    def simulate_response(
        self,
        client,              # Client object from selector.py
        current_round: int,
    ) -> ClientResponse:
        """
        Simulate a single client's response for one federated round.

        Algorithm:
          1. Sample completion_prob ~ Beta(a_i, b_i) for client's tier.
             A fresh Beta draw per round models variable workload.
          2. Bernoulli(1 - completion_prob): if True → FAILURE (no update).
          3. Sample response_time ~ LogNormal(μ_i, σ_i).
          4. If response_time > W_t → STRAGGLER (late; will be buffered).
          5. Otherwise → ON-TIME.

        Staleness on failure: float('inf') → δ^∞ = 0 (zero aggregation weight)
        Staleness on straggler: set to float('inf') here; manage_window assigns
          the final value of 1.0 when buffering (Section 3.10.3).
        Staleness on on-time: 0.0 → δ^0 = 1.0 (full aggregation weight).

        Parameters
        ----------
        client        : Client with tier, latency_mu, latency_sigma attributes.
        current_round : Used to tag round_generated on the response.

        Returns
        -------
        ClientResponse with simulated outcome.
        """
        # --- Step 1: Draw completion probability from tier-specific Beta ---
        beta_a, beta_b = self._get_beta_params(client.tier)

        # Per-round resample: models variable completion probability (network jitter,
        # competing workload, power availability)
        completion_prob = float(self.rng.beta(beta_a, beta_b))

        # --- Step 2: Failure check (Bernoulli) ---
        # If random uniform > completion_prob → client fails to respond
        is_failed = bool(self.rng.random() > completion_prob)

        if is_failed:
            # δ^∞ = 0: failed client contributes nothing to model or N_eff
            return ClientResponse(
                client_id       = client.client_id,
                response_time   = float("inf"),
                is_straggler    = False,     # Failed ≠ straggler (different event)
                is_failed       = True,
                staleness       = float("inf"),
                delta_weight    = 0.0,
                round_generated = current_round,
            )

        # --- Step 3: Sample response time from log-normal distribution ---
        # LogNormal(μ, σ) where μ, σ are parameters of the underlying normal.
        # Blueprint parameters (Section 4.2.2):
        #   Reliable:       μ=1.39, σ=0.30 → median ≈ exp(1.39) ≈ 4s
        #   Moderate:       μ=2.77, σ=0.50 → median ≈ exp(2.77) ≈ 16s
        #   High-Straggler: μ=4.09, σ=0.80 → median ≈ exp(4.09) ≈ 60s
        response_time = float(
            self.rng.lognormal(mean=client.latency_mu, sigma=client.latency_sigma)
        )

        # --- Step 4: Classify as straggler if response_time > W_t ---
        # Straggler staleness is provisional (float('inf')); manage_window
        # will assign the final value of 1.0 when the update is buffered.
        is_straggler = response_time > self.current_W_t
        staleness    = 0.0 if not is_straggler else float("inf")
        delta_weight = self.apply_staleness_weight(self.delta, staleness)

        return ClientResponse(
            client_id       = client.client_id,
            response_time   = response_time,
            is_straggler    = is_straggler,
            is_failed       = False,
            staleness       = staleness,
            delta_weight    = delta_weight,
            round_generated = current_round,
        )

    # ------------------------------------------------------------------
    # 2. Compute staleness  (Section 3.5 / 3.6)
    # ------------------------------------------------------------------

    def compute_staleness(self, current_round: int, generation_round: int) -> int:
        """
        Compute the staleness s_i = current_round - generation_round.

        Staleness measures how many rounds have elapsed between when the client
        computed its gradient (at the global model state w^t) and when the server
        will aggregate it. A gradient computed at round t=5 but aggregated at
        round t=8 reflects a model state that is 3 rounds stale.

        WHY staleness matters:
          The convergence bound (Section 3.12) includes Term 4:
            O(γ²·η²·L²·E²·σ²) from the diversity overhead.
          More generally, stale gradients contribute a systematic drift term
          bounded by O(S_max²). By discounting stale contributions via δ^{s_i},
          we keep this error term bounded even when S_max=3 updates are buffered.

        Parameters
        ----------
        current_round    : The round at which aggregation occurs.
        generation_round : The round at which this client computed its gradient.

        Returns
        -------
        int : s_i = max(0, current_round - generation_round).
        """
        return max(0, current_round - generation_round)

    # ------------------------------------------------------------------
    # 3. Staleness weight  (Sections 3.5, 3.6, 3.10.2)
    # ------------------------------------------------------------------

    def apply_staleness_weight(self, delta: float, staleness: float) -> float:
        """
        Compute δ^{s_i} — the staleness discount applied uniformly across:
          • Aggregation weight:  c̃_i = (n_i/n_A) · δ^{s_i} · (1 + γ·D_i)
          • Reliability update:  R_i^{t+1} = ρ·R^t + (1-ρ)·δ^{s_i}
          • Exposure update:     N_eff(t) += δ^{s_i}

        WHY exponential decay (not linear):
          Linear decay: max(0, 1 - s/S_max) assigns weight 0 to updates at
          staleness S_max and positive weight below it. This creates an abrupt
          cliff at S_max. Exponential decay δ^s is smooth, interpretable, and
          consistent: the same δ that governs aggregation discount also governs
          R_i contraction and N_eff increment. This unified staleness semantics
          (Section 3.5.4) is a deliberate systems feature.

        Concrete values (δ=0.90):
          δ^0 = 1.000  (on-time:  full credit)
          δ^1 = 0.900  (stale 1:  buffered straggler)
          δ^2 = 0.810  (stale 2:  two-round delay)
          δ^3 = 0.729  (stale 3:  at S_max boundary)

        Parameters
        ----------
        delta     : Staleness discount factor δ (canonical: 0.90).
        staleness : s_i ≥ 0. float('inf') or negative → returns 0.0.

        Returns
        -------
        float : δ^{s_i} ∈ [0, 1].
        """
        if np.isinf(staleness) or staleness < 0:
            return 0.0
        return float(delta ** staleness)

    # ------------------------------------------------------------------
    # 4. Manage aggregation window  (Section 3.10.3)
    # ------------------------------------------------------------------

    def manage_window(
        self,
        clients:       List,                     # Selected Client objects
        responses:     Dict[int, ClientResponse], # cid → ClientResponse
        current_round: int,
        n_selected:    int,
    ) -> Tuple[List[ClientResponse], List[ClientResponse]]:
        """
        Partition client responses into on-time and straggler sets.
        Update the adaptive aggregation window W_t.
        Drain eligible buffered updates from prior rounds.
        Check quorum condition and log aggregation trigger.

        Full algorithm:
          1. Collect arrived response times → update latency history
          2. Recompute W_t = max(T̃_med + 2.5·MAD, W_min)
          3. Classify each response:
               FAILED    → discard (no buffering, no on-time)
               STRAGGLER → buffer with staleness=1, include in round t+1
               ON-TIME   → include in this round's aggregate (staleness=0)
          4. Drain buffer: if age ≥ 1, include with final staleness s
                           if age ≥ S_max, discard
          5. Check quorum: |on_time| ≥ ρ_q · n_selected?

        Aggregation triggers on (Section 3.10.3, Section 4.3.1):
          (A) Quorum: |A^t| ≥ ρ_q·C  (default: 70% of C=6 → 4 clients)
          (B) Window expiry W_t

        Parameters
        ----------
        clients       : Selected Client objects for this round.
        responses     : Dict mapping client_id → ClientResponse from simulate_response.
        current_round : Current round index (0-based).
        n_selected    : Number of clients that were selected (C = n_select).

        Returns
        -------
        on_time_responses   : ClientResponse list for aggregation this round
                              (includes on-time + drained buffer updates).
        straggler_responses : ClientResponse list of newly buffered stragglers.
        """
        # --- Step 1: Collect arrived latencies for window history ---
        arrived_times = [
            r.response_time
            for r in responses.values()
            if not r.is_failed and not np.isinf(r.response_time)
        ]
        self._latency_history.extend(arrived_times)

        # --- Step 2: Update adaptive window W_t ---
        self._update_window()

        # --- Step 3: Classify responses ---
        on_time:   List[ClientResponse] = []
        stragglers:List[ClientResponse] = []
        n_failed = 0

        for resp in responses.values():
            cid = resp.client_id

            if resp.is_failed:
                n_failed += 1
                print(
                    f"      [Straggler] C{cid}: ✗ FAILED — "
                    f"no update (δ^s=0.0, N_eff unchanged)"
                )
                continue

            if resp.is_straggler:
                # Assign final staleness = 1 (buffered for round t+1)
                resp.staleness   = 1.0
                resp.delta_weight = self.apply_staleness_weight(self.delta, 1.0)

                # Buffer the update
                buf = BufferedUpdate(
                    client_id          = cid,
                    response           = resp,
                    buffer_entry_round = current_round,
                )
                # Overwrite any prior buffered update from same client
                self._buffer[cid] = buf
                stragglers.append(resp)

                print(
                    f"      [Straggler] C{cid}: ⟳ LATE "
                    f"(t={resp.response_time:.2f}s > W_t={self.current_W_t:.2f}s) "
                    f"→ BUFFERED  s=1, δ^s={resp.delta_weight:.4f}"
                )
            else:
                # On-time: staleness = 0, full credit
                resp.staleness    = 0.0
                resp.delta_weight = self.apply_staleness_weight(self.delta, 0.0)
                on_time.append(resp)

                print(
                    f"      [Straggler] C{cid}: ✓ ON-TIME "
                    f"(t={resp.response_time:.2f}s ≤ W_t={self.current_W_t:.2f}s) "
                    f"δ^s={resp.delta_weight:.4f}"
                )

        # --- Step 4: Drain buffer (prior-round stragglers now eligible) ---
        drained, expired = self._drain_buffer(current_round)
        on_time.extend(drained)

        if expired:
            for exp_buf in expired:
                print(
                    f"      [Buffer] C{exp_buf.client_id}: ✗ EXPIRED "
                    f"(age={exp_buf.age(current_round)} ≥ S_max={self.s_max}) → DISCARDED"
                )

        # --- Step 5: Log window and quorum status ---
        quorum_threshold = max(1, int(np.ceil(self.quorum_frac * n_selected)))
        quorum_met       = len(on_time) >= quorum_threshold
        trigger          = "QUORUM ✓" if quorum_met else "WINDOW EXPIRY"

        print(
            f"      [Window] W_t={self.current_W_t:.2f}s  "
            f"| on-time={len(on_time)}  "
            f"| buffered={len(stragglers)}  "
            f"| failed={n_failed}  "
            f"| quorum_threshold={quorum_threshold}  "
            f"| trigger={trigger}"
        )

        # --- Step 6: Record monitoring metrics ---
        self.window_history.append(self.current_W_t)
        self.straggler_counts.append(len(stragglers))
        self.on_time_counts.append(len(on_time))
        self.failed_counts.append(n_failed)
        self.quorum_met_history.append(quorum_met)

        return on_time, stragglers

    # ------------------------------------------------------------------
    # Internal: adaptive window update
    # ------------------------------------------------------------------

    def _update_window(self) -> None:
        """
        Recompute W_t from rolling latency history using robust statistics.

            W_t = max(T̃_med + k · MAD(T_1,...,T_n), W_min)

        T̃_med: median of last 10 rounds' per-client response times
        MAD:   median absolute deviation = median(|T_i - T̃_med|)
        k:     2.5 (calibrated to tolerate 40–60% straggler rates)

        With k=2.5 MADs, approximately 1.25% of normally-distributed
        response times would exceed W_t (≈ 2.5σ equivalent under normality).
        Log-normal tails are heavier, so this is a conservative window.
        """
        if len(self._latency_history) < 2:
            self.current_W_t = self.w_min
            return

        times    = np.array(self._latency_history)
        t_med    = float(np.median(times))
        mad      = float(np.median(np.abs(times - t_med)))  # Median Absolute Deviation
        self.current_W_t = float(max(t_med + self.mad_k * mad, self.w_min))

    # ------------------------------------------------------------------
    # Internal: drain buffer
    # ------------------------------------------------------------------

    def _drain_buffer(
        self,
        current_round: int,
    ) -> Tuple[List[ClientResponse], List[BufferedUpdate]]:
        """
        Process the straggler buffer at the start of each round:
          - If age ≥ 1 (stored last round):  recompute staleness → include
          - If age ≥ s_max:                  discard (too stale to be useful)
          - Otherwise:                        keep waiting

        WHY staleness is recomputed at drain time (not at buffer entry):
          When a straggler is buffered at round t with entry staleness=1,
          it is intended for round t+1. But if the buffer is not consumed at
          t+1 (rare, but possible under concurrent buffering), the correct
          staleness at drain time t' is (t' - round_generated). This ensures
          the δ^{s_i} discount accurately reflects how stale the gradient is
          at the time of aggregation.

        Returns
        -------
        (drained_responses, expired_buffers)
        """
        to_drain:  List[ClientResponse] = []
        expired:   List[BufferedUpdate] = []
        to_keep:   Dict[int, BufferedUpdate] = {}

        for cid, buf in list(self._buffer.items()):
            age = buf.age(current_round)

            if age >= self.s_max:
                # Discard: update is too stale to contribute meaningfully
                expired.append(buf)

            elif age >= 1:
                # Ready to drain: compute final staleness from generation round
                final_s = self.compute_staleness(current_round, buf.response.round_generated)
                buf.response.staleness   = float(final_s)
                buf.response.delta_weight = self.apply_staleness_weight(self.delta, final_s)
                to_drain.append(buf.response)

                print(
                    f"      [Buffer] C{cid}: ↩ DRAINED "
                    f"(gen_round={buf.response.round_generated}, "
                    f"current={current_round}, s={final_s}, "
                    f"δ^s={buf.response.delta_weight:.4f})"
                )
            else:
                # Not yet eligible (just buffered this round — should not happen,
                # but guard against edge cases in concurrent scenarios)
                to_keep[cid] = buf

        self._buffer = to_keep
        return to_drain, expired

    # ------------------------------------------------------------------
    # Internal: Beta parameter lookup
    # ------------------------------------------------------------------

    def _get_beta_params(self, tier: str) -> Tuple[float, float]:
        """
        Return Beta(a, b) parameters for the given straggler tier.

        Tier → Beta mapping (Section 4.2.2):
          reliable       → Beta(19, 1):  mean = 19/20 = 0.95
          moderate       → Beta(7,  3):  mean = 7/10  = 0.70
          high_straggler → Beta(2,  3):  mean = 2/5   = 0.40

        Unknown tier defaults to Beta(10, 2) (mean≈0.83, moderately reliable).
        """
        return BETA_PARAMS.get(tier, (10.0, 2.0))

    # ------------------------------------------------------------------
    # Public utilities
    # ------------------------------------------------------------------

    def get_buffer_state(self) -> Dict[int, Dict]:
        """Return a snapshot of the current buffer for debugging."""
        return {
            cid: {
                "buffer_entry_round":  buf.buffer_entry_round,
                "round_generated":     buf.response.round_generated,
                "response_time":       buf.response.response_time,
            }
            for cid, buf in self._buffer.items()
        }

    def flush_buffer(self) -> None:
        """Hard-clear the buffer (use between simulation runs)."""
        self._buffer.clear()
        print("      [Buffer] Flushed.")

    def summary(self) -> Dict[str, float]:
        """Return aggregate statistics over all simulated rounds."""
        if not self.straggler_counts:
            return {}
        return {
            "total_rounds":         len(self.straggler_counts),
            "mean_on_time":         float(np.mean(self.on_time_counts)),
            "mean_straggler":       float(np.mean(self.straggler_counts)),
            "mean_failed":          float(np.mean(self.failed_counts)),
            "mean_W_t":             float(np.mean(self.window_history)),
            "quorum_rate":          float(np.mean(self.quorum_met_history)),
        }

    def __repr__(self) -> str:
        return (
            f"StragglerManager(δ={self.delta}, S_max={self.s_max}, "
            f"ρ_q={self.quorum_frac}, k={self.mad_k}, W_min={self.w_min})"
        )
