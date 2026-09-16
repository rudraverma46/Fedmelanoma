"""
server/strategy.py
===================
Phase 3C -- FedMelanoma Research Strategy.

Replaces Flower's default FedAvg orchestration (configure_fit's random /
fraction-based sampling, and aggregate_fit's plain size-weighted average)
with the EXISTING, UNMODIFIED FedMelanoma research algorithms:

  - Client selection    -> server/selector.py    (Selector.boltzmann_selection)
  - Straggler handling  -> server/straggler.py   (StragglerManager.simulate_response / manage_window)
  - Aggregation weights -> server/aggregation.py (compute_aggregation_weights)
  - Diversity priors    -> emd.py                (EarthMoverDistance.compute_diversity_score)

THIS FILE IS ADAPTER / ORCHESTRATION CODE ONLY.

Every mathematical formula, every hyperparameter default, and every
algorithmic decision boundary lives in the four research modules above,
imported here unmodified (none of their names are redefined, re-derived,
or approximated in this file). This file's job is exclusively to:

  1. Convert Flower's ClientProxy / Parameters / FitRes / EvaluateRes
     objects into the plain Python objects the research modules expect
     (selector.Client, straggler.ClientResponse), and convert their
     outputs back into Flower's Parameters / metrics dict types.
  2. Call the research modules' existing public functions/methods at the
     appropriate points in Flower's round lifecycle (see the callback
     mapping in each method's docstring below).
  3. Perform the ndarray arithmetic required to turn
     compute_aggregation_weights()'s c_i output into an updated
     Parameters object, implementing the blueprint's own boxed equation
     (Section 3.10.2) literally:

         w^{t+1} = w^t + Sum_{i in A_t} c_i * (w_i^{t+1} - w^t)

     This is arithmetic on already-computed weights, not a re-derivation
     of the weighting formula itself (that computation happens entirely
     inside aggregation.compute_aggregation_weights, unmodified).

No formula, constant, or decision boundary from selector.py,
aggregation.py, straggler.py, or emd.py is reproduced, altered, or
approximated anywhere in this file.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from flwr.common import (
    EvaluateIns,
    FitIns,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

# ---------------------------------------------------------------------------
# RESEARCH MODULES -- imported unmodified. This file never redefines any of
# the names below; it only calls into them.
# ---------------------------------------------------------------------------
from emd import EarthMoverDistance
from server.selector import Client as SelectorClient, Selector
from server.straggler import ClientResponse as StragglerClientResponse, StragglerManager
from server.aggregation import (
    compute_aggregation_weights,
    DELTA as AGGREGATION_DELTA,
    GAMMA as AGGREGATION_GAMMA,
)

# server/server.py's weighted_average() is pure Flower-generic metric
# bookkeeping (Phase 3B: example-count-weighted mean of a metrics dict),
# NOT a research algorithm -- reused here rather than duplicated, per
# "prefer calling existing functions over copying code".
from server.server import weighted_average

# ---------------------------------------------------------------------------
# Phase 5 ADAPTER modules -- monitoring/checkpointing/logging only. These
# READ already-computed research quantities (R_i, N_eff, D_i, c_i, W_t,
# staleness) for logging; they never compute, adjust, or duplicate any of
# that math, and D_i is never recomputed here (it stays the static prior
# set once in compute_normalized_diversity_priors() below).
# ---------------------------------------------------------------------------
from monitoring.param_hash import hash_ndarrays
from monitoring.experiment_logging import ExperimentLogger, new_run_id
from monitoring.checkpointing import (
    save_global_checkpoint,
    maybe_update_best_checkpoint,
    DEFAULT_CHECKPOINT_DIR,
)
from monitoring.straggler_logging import log_client_round_outcome, log_round_summary
from monitoring.subgroup_metrics import aggregate_client_confusion_reports, compute_fairness_gap


# ===========================================================================
# ADAPTER: static hospital registration metadata
# ===========================================================================

@dataclass
class HospitalConfig:
    """
    ADAPTER-LAYER data structure -- NOT part of any research module.

    Represents the static per-hospital metadata the server already knows
    about a client at registration time, per Blueprint Phase 7
    ("Registration protocol: Client transmits n_i, [P_phi, P_tau,
    P_lambda] ... Server computes D_i ... assigns initial R_i(0)=1.0,
    N_eff(0)=0"). This phase does not implement the live registration
    handshake itself (that network exchange is out of scope for a
    Strategy-integration phase) -- it accepts the already-known result
    of that handshake as configuration, exactly as the blueprint's own
    Phase 6 ("Selector State Maintenance") assumes D_i is already
    available before round-based training begins.
    """
    n_samples: int
    tier: str
    latency_mu: float
    latency_sigma: float
    fitzpatrick_dist: Sequence[float]
    modality_dist: Sequence[float]
    confirmation_dist: Sequence[float]


def compute_normalized_diversity_priors(
    hospital_configs: Sequence[HospitalConfig],
    global_fitzpatrick: Sequence[float],
    global_modality: Sequence[float],
    global_confirmation: Sequence[float],
    fitz_weight: float = 0.40,
    modality_weight: float = 0.35,
    confirmation_weight: float = 0.25,
) -> List[float]:
    """
    ADAPTER: calls the UNMODIFIED EarthMoverDistance.compute_diversity_score
    (emd.py) once per hospital to obtain each D_i^raw, then applies the
    blueprint's max-normalisation (Section 3.2.6):

        D_i = D_i^raw / max_j D_j^raw

    This normalisation is documented in the blueprint but is NOT
    implemented inside emd.py itself -- compute_diversity_score's return
    dict contains only "raw_diversity_score" (see emd.py). The two lines
    below are the minimum glue needed to bridge emd.py's raw output to
    the [0, 1]-bounded D_i that selector.Client.D expects (Client's own
    __post_init__ clips D to [0, 1]). No Optimal Transport / Wasserstein
    mathematics is reproduced or altered here -- every distance value
    used comes directly from emd.py's unmodified computation.
    """
    raw_scores: List[float] = []
    for hc in hospital_configs:
        result = EarthMoverDistance.compute_diversity_score(
            global_fitzpatrick=global_fitzpatrick,
            client_fitzpatrick=hc.fitzpatrick_dist,
            global_modality=global_modality,
            client_modality=hc.modality_dist,
            global_confirmation=global_confirmation,
            client_confirmation=hc.confirmation_dist,
            fitz_weight=fitz_weight,
            modality_weight=modality_weight,
            confirmation_weight=confirmation_weight,
        )
        raw_scores.append(result["raw_diversity_score"])

    max_raw = max(raw_scores) if raw_scores and max(raw_scores) > 0 else 1.0
    return [r / max_raw for r in raw_scores]


# ===========================================================================
# FedMelanomaStrategy
# ===========================================================================

class FedMelanomaStrategy(Strategy):
    """
    Flower Strategy that replaces FedAvg's default client sampling and
    parameter aggregation with the existing, unmodified FedMelanoma
    research algorithms.

    Responsibility split (see module docstring): this class contains NO
    scoring formula, NO staleness discount, NO aggregation weight
    formula, and NO Optimal Transport math of its own -- every one of
    those lives in server/selector.py, server/straggler.py,
    server/aggregation.py, and emd.py, called here unmodified.
    """

    def __init__(
        self,
        hospital_configs: Sequence[HospitalConfig],
        global_fitzpatrick: Sequence[float],
        global_modality: Sequence[float],
        global_confirmation: Sequence[float],
        n_select: Optional[int] = None,
        initial_parameters: Optional[List[np.ndarray]] = None,
        selector: Optional[Selector] = None,
        straggler_manager: Optional[StragglerManager] = None,
        local_epochs: int = 1,
        rng: Optional[np.random.Generator] = None,
        experiment_logger: Optional[ExperimentLogger] = None,
        run_id: Optional[str] = None,
        checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
        results_dir: str = "results",
        param_keys: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Phase 5 additions (all optional, all default to sane auto-created
        values so existing callers -- e.g. tests/test_phase4_runtime.py,
        which constructs FedMelanomaStrategy without any of these -- keep
        working unmodified):

        experiment_logger / run_id : where structured logs
            (logs/op|math|audit/<run_id>.jsonl) are written. If
            experiment_logger is None, one is created automatically from
            run_id (or a freshly generated run_id if that is also None).
        checkpoint_dir / results_dir : root directories for
            checkpoints/<run_id>/... and results/<run_id>/....
        param_keys : model.state_dict().keys(), in order -- lets saved
            checkpoints store a proper named state_dict instead of a bare
            ndarray list. Optional; checkpoints remain fully loadable
            without it (see monitoring/checkpointing.py).
        seed : recorded into the audit log only, for reproducibility;
            never read back by this class (the caller's own `rng` is what
            actually controls randomness, unchanged from before).
        """
        super().__init__()

        self._hospital_configs: List[HospitalConfig] = list(hospital_configs)
        self._n_select: int = n_select if n_select is not None else len(self._hospital_configs)
        self._local_epochs: int = local_epochs
        self._rng: np.random.Generator = rng or np.random.default_rng()

        self._initial_parameters_ndarrays: Optional[List[np.ndarray]] = initial_parameters

        # ---- RESEARCH MODULE CALL (unmodified): emd.py ----
        # D_i computed once, at construction time -- "D_i values are
        # computed once at client registration ... and remain fixed
        # throughout training" (Blueprint Section 3.2.6 / Phase 6).
        diversity_priors: List[float] = compute_normalized_diversity_priors(
            self._hospital_configs, global_fitzpatrick, global_modality, global_confirmation,
        )

        # ---- RESEARCH MODULE CALL (unmodified): selector.Client ----
        # One selector.Client per hospital, constructed with the
        # blueprint's own field defaults (R=1.0, N_eff=0.0).
        self._clients: Dict[int, SelectorClient] = {
            client_id: SelectorClient(
                client_id=client_id,
                n_samples=hc.n_samples,
                D=diversity_priors[client_id],
                tier=hc.tier,
                latency_mu=hc.latency_mu,
                latency_sigma=hc.latency_sigma,
            )
            for client_id, hc in enumerate(self._hospital_configs)
        }

        # ---- RESEARCH MODULE INSTANTIATION (unmodified) ----
        # Both use THEIR OWN canonical defaults -- no hyperparameter
        # (rho, delta, lambda, gamma, tau, s_max, quorum_frac, mad_k,
        # w_min) is overridden here.
        self._selector: Selector = selector or Selector()
        self._straggler_manager: StragglerManager = straggler_manager or StragglerManager(rng=self._rng)

        # ---- ADAPTER STATE ----
        # Flower ClientProxy.cid <-> selector.py integer client_id
        # mapping, built lazily the first time configure_fit sees the
        # client_manager's registered clients. Flower's cid format is
        # backend-dependent/opaque and client/client.py cannot be
        # modified to report a stable research client_id, so the mapping
        # is established here, once, from whichever cids are present.
        self._cid_to_client_id: Dict[str, int] = {}
        self._client_id_to_cid: Dict[int, str] = {}

        # Remembers which research client_ids were selected for fit()
        # this round (S^t in blueprint notation), so aggregate_fit and
        # configure_evaluate can refer back to the same set.
        self._last_round_client_ids: List[int] = []

        # w^t for the round currently in flight (the Parameters handed
        # to configure_fit), needed by aggregate_fit to compute each
        # contributing client's delta per the blueprint's boxed formula.
        self._current_global_ndarrays: Optional[List[np.ndarray]] = None

        # Each client's most recently RETURNED (not yet necessarily
        # consumed) parameters, keyed by research client_id. Populated in
        # aggregate_fit whenever a FitRes arrives; read (and cleared) when
        # that client's update is actually incorporated -- which may
        # happen in a LATER round if straggler.py buffers it first. This
        # is pure bookkeeping so a buffered/drained update can still be
        # applied once aggregation.compute_aggregation_weights includes
        # it, without re-requesting anything from the client.
        self._pending_client_ndarrays: Dict[int, List[np.ndarray]] = {}

        # ---- Phase 5: monitoring / checkpointing / broadcast-verification
        # adapter state. Nothing below computes or alters any research
        # quantity -- it only records what the research modules already
        # produced. ----
        self._run_id: str = run_id or new_run_id("fedmelanoma")
        self._logger: ExperimentLogger = experiment_logger or ExperimentLogger(self._run_id)
        self._checkpoint_dir: str = checkpoint_dir
        self._results_dir: str = results_dir
        self._param_keys: Optional[List[str]] = list(param_keys) if param_keys is not None else None
        self._best_checkpoint_state: Dict[str, float] = {"best_value": float("-inf")}

        # Round-to-round broadcast/aggregation hash chain (Requirement 2).
        self._last_broadcast_hash: Optional[str] = None
        self._last_global_output_hash: Optional[str] = None
        self._broadcast_matches_previous_round: Optional[bool] = None
        self._last_aggregated_ndarrays: Optional[List[np.ndarray]] = None
        self._last_aggregated_round: Optional[int] = None

        # R_i / N_eff snapshots taken in configure_fit (before this
        # round's update_client_state runs in aggregate_fit), purely for
        # "before" values in the straggler log -- selector.py's own state
        # is the only thing ever mutated.
        self._round_r_before: Dict[int, float] = {}
        self._round_n_eff_before: Dict[int, float] = {}
        self._round_start_times: Dict[int, float] = {}

        initial_hash = (
            hash_ndarrays(self._initial_parameters_ndarrays)
            if self._initial_parameters_ndarrays is not None
            else None
        )
        self._logger.log_run_start(
            {
                "strategy": "FedMelanomaStrategy",
                "n_hospitals": len(self._hospital_configs),
                "n_select": self._n_select,
                "local_epochs": self._local_epochs,
                "seed": seed,
                "checkpoint_dir": self._checkpoint_dir,
                "results_dir": self._results_dir,
                "initial_parameters_hash": initial_hash,
                "diversity_priors_D": {c.client_id: c.D for c in self._clients.values()},
                "selector_hyperparameters": {
                    "rho": self._selector.rho,
                    "delta": self._selector.delta,
                    "lambda_": self._selector.lambda_,
                    "gamma": self._selector.gamma,
                    "tau": self._selector.tau,
                },
                "straggler_hyperparameters": {
                    "delta": self._straggler_manager.delta,
                    "s_max": self._straggler_manager.s_max,
                    "quorum_frac": self._straggler_manager.quorum_frac,
                    "mad_k": self._straggler_manager.mad_k,
                    "w_min": self._straggler_manager.w_min,
                },
            }
        )

    # ------------------------------------------------------------------
    # ADAPTER: Flower ClientProxy.cid <-> selector.py client_id mapping
    # ------------------------------------------------------------------

    def _ensure_client_mapping(self, client_manager: ClientManager) -> None:
        if self._cid_to_client_id:
            return  # already built

        available = client_manager.all()
        cids_sorted = sorted(available.keys())
        if len(cids_sorted) != len(self._hospital_configs):
            raise ValueError(
                f"FedMelanomaStrategy was configured with "
                f"{len(self._hospital_configs)} hospital_configs but "
                f"{len(cids_sorted)} Flower clients are registered. "
                f"These counts must match 1:1 so each selector.Client maps "
                f"to exactly one Flower ClientProxy."
            )
        for client_id, cid in enumerate(cids_sorted):
            self._cid_to_client_id[cid] = client_id
            self._client_id_to_cid[client_id] = cid

    # ------------------------------------------------------------------
    # initialize_parameters -- pure Flower orchestration (no research call)
    # ------------------------------------------------------------------

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        if self._initial_parameters_ndarrays is not None:
            return ndarrays_to_parameters(self._initial_parameters_ndarrays)
        return None  # Flower falls back to asking a client for its parameters

    # ------------------------------------------------------------------
    # configure_fit -- RESEARCH MODULE CALL: selector.py client selection
    # ------------------------------------------------------------------

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Flower lifecycle callback #1 (per round): decide which clients
        train this round.

        Integration point -- server/selector.py:
            Selector.boltzmann_selection(clients, n_select, rng) is
            called unmodified over the registered hospital pool, MINUS
            any client currently sitting in the straggler buffer awaiting
            drain (see the buffered_client_ids filter below). No scoring
            or sampling logic is reimplemented here -- only the candidate
            list passed in is filtered.

            This filter exists because a client can be reselected by
            Boltzmann sampling (which has no notion of buffer state)
            before its previous round's buffered update is drained,
            producing two ClientResponse entries for the same client_id
            within a single round's on_time list -- silently corrupting
            aggregation.compute_aggregation_weights' dict-keyed
            accumulation (found and fixed during Phase 3C-2 verification;
            see the verification report for details).
        """
        self._ensure_client_mapping(client_manager)
        self._current_global_ndarrays = parameters_to_ndarrays(parameters)  # w^t for this round
        self._round_start_times[server_round] = time.time()

        # ---- Phase 5: broadcast verification (Requirement 2) ----
        # Hash what the server is about to send this round, and compare it
        # against the hash of what aggregate_fit produced LAST round (None
        # on round 1, where there is nothing to compare against yet).
        global_input_hash = hash_ndarrays(self._current_global_ndarrays)
        self._broadcast_matches_previous_round = (
            None if self._last_global_output_hash is None
            else (global_input_hash == self._last_global_output_hash)
        )
        self._last_broadcast_hash = global_input_hash
        self._logger.log_math(
            {
                "event": "round_broadcast",
                "round": server_round,
                "global_input_hash": global_input_hash,
                "previous_round_global_output_hash": self._last_global_output_hash,
                "broadcast_matches_previous_round_output": self._broadcast_matches_previous_round,
            }
        )

        # ---- ADAPTER (bugfix, see docstring below): exclude clients whose
        # previous-round update is still sitting in the straggler buffer
        # (not yet drained) from this round's selection candidate pool.
        # Uses straggler.py's own public get_buffer_state() -- no private
        # attribute access, no research-module modification.
        buffered_client_ids = set(self._straggler_manager.get_buffer_state().keys())
        candidate_clients = [
            c for c in self._clients.values() if c.client_id not in buffered_client_ids
        ]

        # ---- RESEARCH MODULE CALL (unmodified) ----
        selected_clients: List[SelectorClient] = self._selector.boltzmann_selection(
            clients=candidate_clients,
            n_select=min(self._n_select, len(candidate_clients)),
            rng=self._rng,
        )
        self._last_round_client_ids = [c.client_id for c in selected_clients]

        # ---- Phase 5: snapshot R_i/N_eff BEFORE this round mutates them
        # (in aggregate_fit's update_client_state calls), and log the
        # selector's own score breakdown for every selected client (read-
        # only calls to Selector.score_breakdown -- no new scoring math). ----
        self._round_r_before = {c.client_id: c.R for c in selected_clients}
        self._round_n_eff_before = {c.client_id: c.N_eff for c in selected_clients}
        for c in selected_clients:
            self._logger.log_math(
                {"event": "client_selected", "round": server_round, **self._selector.score_breakdown(c)}
            )
        self._logger.log_op(
            {
                "event": "round_configure_fit",
                "round": server_round,
                "n_candidates": len(candidate_clients),
                "n_excluded_buffered": len(buffered_client_ids),
                "selected_clients": list(self._last_round_client_ids),
            }
        )

        available = client_manager.all()
        config: Dict[str, Scalar] = {"local_epochs": self._local_epochs}
        fit_ins = FitIns(parameters, config)

        return [
            (available[self._client_id_to_cid[c.client_id]], fit_ins)
            for c in selected_clients
        ]

    # ------------------------------------------------------------------
    # aggregate_fit -- RESEARCH MODULE CALLS: straggler.py + aggregation.py + selector.py
    # ------------------------------------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, "FitRes"]],
        failures: List[Union[Tuple[ClientProxy, "FitRes"], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Flower lifecycle callback #2 (per round): turn returned client
        updates into a new global model.

        Integration points:
            server/straggler.py -- StragglerManager.simulate_response()
                classifies each of this round's selected clients as
                on-time / straggler / failed (Section 4.2.2's simulation
                model; Flower's synchronous simulation backend does not
                itself model network delay, so this simulated
                classification is layered on top of Flower's
                always-complete results to exercise the blueprint's
                staleness/failure semantics faithfully).
                StragglerManager.manage_window() then partitions
                on-time/straggler, updates the adaptive window W_t, and
                drains any previously-buffered stragglers -- unmodified.
            server/aggregation.py -- compute_aggregation_weights()
                computes c_i for the arrived set A_t exactly per Section
                3.10.2's boxed formula -- unmodified.
            server/selector.py -- Selector.update_client_state() updates
                R_i / N_eff for every client in S^t (this round's
                selected set) -- unmodified.
        """
        if not results or self._current_global_ndarrays is None:
            self._logger.log_op(
                {"event": "round_fit_complete", "round": server_round, "status": "no_flower_results"}
            )
            return None, {}

        old_ndarrays = self._current_global_ndarrays  # w^t

        # ---- ADAPTER: map Flower results back to research client_ids,
        # and cache each client's freshly returned parameters. ----
        result_by_client_id: Dict[int, "FitRes"] = {}
        for proxy, fit_res in results:
            client_id = self._cid_to_client_id[proxy.cid]
            result_by_client_id[client_id] = fit_res
            self._pending_client_ndarrays[client_id] = parameters_to_ndarrays(fit_res.parameters)

            # ---- Phase 5: broadcast verification, client side (Requirement 2) ----
            # Compare the hash the CLIENT itself computed on receipt (echoed
            # back in fit_res.metrics by client/client.py) against what the
            # server believes it broadcast this round -- proves the client
            # actually trained on the exact broadcast parameters, rather than
            # trusting the server's own copy of what it meant to send.
            client_received_hash = fit_res.metrics.get("client_received_param_hash")
            client_returned_hash = fit_res.metrics.get("client_returned_param_hash") or hash_ndarrays(
                self._pending_client_ndarrays[client_id]
            )
            self._logger.log_math(
                {
                    "event": "client_fit_returned",
                    "round": server_round,
                    "client_id": client_id,
                    "global_input_hash": self._last_broadcast_hash,
                    "client_received_param_hash": client_received_hash,
                    "client_received_matches_broadcast": (
                        client_received_hash == self._last_broadcast_hash
                        if client_received_hash is not None
                        else None
                    ),
                    "client_returned_param_hash": client_returned_hash,
                    "num_examples": fit_res.num_examples,
                }
            )

        # ---- RESEARCH MODULE CALL (unmodified): straggler.py simulation ----
        responses: Dict[int, StragglerClientResponse] = {}
        for client_id in self._last_round_client_ids:
            if client_id not in result_by_client_id:
                continue  # a genuine Flower-level failure, not simulated
            client = self._clients[client_id]
            responses[client_id] = self._straggler_manager.simulate_response(
                client=client, current_round=server_round,
            )

        on_time, new_stragglers = self._straggler_manager.manage_window(
            clients=[self._clients[cid] for cid in responses.keys()],
            responses=responses,
            current_round=server_round,
            n_selected=len(self._last_round_client_ids),
        )

        # ---- RESEARCH MODULE CALL (unmodified): aggregation.py ----
        # `on_time` entries are straggler.ClientResponse instances,
        # accepted directly by compute_aggregation_weights via the
        # (.client_id, .staleness) duck-typed interface that
        # aggregation.py's own ClientResponse docstring documents
        # ("produced by simulate_response in straggler.py; consumed here
        # by ... compute_aggregation_weights").
        weights: Dict[int, float] = compute_aggregation_weights(
            on_time=on_time,
            client_map=self._clients,
            delta=AGGREGATION_DELTA,
            gamma=AGGREGATION_GAMMA,
        )

        # ---- ADAPTER ARITHMETIC: blueprint Section 3.10.2 boxed formula ----
        #     w^{t+1} = w^t + Sum_{i in A_t} c_i * (w_i^{t+1} - w^t)
        # This uses the c_i values computed above by aggregation.py
        # verbatim; it does not compute or alter any weight.
        aggregated_ndarrays: List[np.ndarray] = [arr.copy() for arr in old_ndarrays]
        applied_client_ids: List[int] = []
        if weights:
            weighted_delta_sum = [np.zeros_like(arr) for arr in old_ndarrays]
            for client_id, c_i in weights.items():
                w_i = self._pending_client_ndarrays.get(client_id)
                if w_i is None:
                    # A client appeared in A_t (e.g. via buffer drain) but
                    # this adapter has no cached parameters for it -- can
                    # only occur if it never actually returned a FitRes,
                    # which manage_window's own bookkeeping should not
                    # produce. Skip defensively rather than crash a round.
                    continue
                for idx in range(len(weighted_delta_sum)):
                    delta = w_i[idx] - old_ndarrays[idx]
                    weighted_delta_sum[idx] = weighted_delta_sum[idx] + c_i * delta
                applied_client_ids.append(client_id)
                self._pending_client_ndarrays.pop(client_id, None)  # consumed

            aggregated_ndarrays = [
                old_ndarrays[idx] + weighted_delta_sum[idx] for idx in range(len(old_ndarrays))
            ]

        # ---- RESEARCH MODULE CALL (unmodified): selector.py state update ----
        # Every client in S^t (this round's selected set) gets its R_i /
        # N_eff updated per its OWN outcome this round -- on-time (s=0),
        # newly-buffered straggler (s=1), or failed (s=inf). Clients
        # merely DRAINED from a prior round's buffer are NOT updated
        # again here (their update_client_state call already happened in
        # the round they were originally buffered) -- consistent with
        # Blueprint Phase 3 Section 3.2: "For each selected client i in
        # S^t: R_i update, N_eff update"; drained entries are not part of
        # THIS round's S^t.
        for client_id in self._last_round_client_ids:
            client = self._clients[client_id]
            if client_id in responses:
                staleness = responses[client_id].staleness
            else:
                staleness = float("inf")  # genuine Flower-level failure
            self._selector.update_client_state(
                client=client, staleness=staleness, round_idx=server_round, was_selected=True,
            )

        metrics = weighted_average(
            [(fit_res.num_examples, fit_res.metrics) for fit_res in result_by_client_id.values()]
        )
        metrics["fedmelanoma_clients_aggregated"] = len(applied_client_ids)
        metrics["fedmelanoma_new_stragglers_buffered"] = len(new_stragglers)

        # ====================================================================
        # Phase 5: hash-chain closure, checkpointing, and structured logging.
        # Everything below is READ-ONLY with respect to the research state
        # mutated above (R_i/N_eff already updated; c_i/on_time/W_t/D_i
        # already computed) -- it only records those values and persists
        # the aggregated parameters. No formula here.
        # ====================================================================
        global_output_hash = hash_ndarrays(aggregated_ndarrays)
        self._last_global_output_hash = global_output_hash
        self._last_aggregated_ndarrays = aggregated_ndarrays
        self._last_aggregated_round = server_round
        metrics["global_input_hash"] = self._last_broadcast_hash
        metrics["global_output_hash"] = global_output_hash
        metrics["broadcast_matches_previous_round_output"] = self._broadcast_matches_previous_round

        # ---- Requirement 10 / TEST 9: explicit duplicate-contribution check.
        # Structurally prevented by configure_fit's buffered-client exclusion
        # (see its docstring) -- verified again here rather than assumed. ----
        on_time_ids_list = [r.client_id for r in on_time]
        if len(on_time_ids_list) != len(set(on_time_ids_list)):
            self._logger.log_audit(
                {
                    "event": "DUPLICATE_CLIENT_CONTRIBUTION_DETECTED",
                    "round": server_round,
                    "on_time_client_ids": on_time_ids_list,
                }
            )

        # ---- Checkpoint: saved immediately after successful aggregation
        # (Requirement 1). Enriched with validation metrics later, in
        # aggregate_evaluate, for this SAME round (see _last_aggregated_round
        # guard there) -- never a client-local model, always this round's
        # actual aggregated global parameters. ----
        checkpoint_path = save_global_checkpoint(
            round_num=server_round,
            ndarrays=aggregated_ndarrays,
            run_id=self._run_id,
            metrics={"fit_train_loss": metrics.get("train_loss")},
            param_keys=self._param_keys,
            checkpoint_dir=self._checkpoint_dir,
        )
        self._logger.log_audit(
            {
                "event": "checkpoint_saved",
                "round": server_round,
                "path": checkpoint_path,
                "param_hash": global_output_hash,
                "n_parameters": int(sum(a.size for a in aggregated_ndarrays)),
                "parameter_shapes": [list(a.shape) for a in aggregated_ndarrays],
            }
        )

        # ---- Round-level math/op logs ----
        quorum_threshold = max(1, math.ceil(self._straggler_manager.quorum_frac * len(self._last_round_client_ids)))
        drained_ids = [cid for cid in on_time_ids_list if cid not in responses]
        self._logger.log_math(
            {
                "event": "round_aggregate",
                "round": server_round,
                "window_w_t": self._straggler_manager.current_W_t,
                "n_selected": len(self._last_round_client_ids),
                "n_on_time": len(on_time),
                "n_straggler": len(new_stragglers),
                "n_failed": len([r for r in responses.values() if r.is_failed]),
                "n_aggregated": len(applied_client_ids),
                "quorum_threshold": quorum_threshold,
                "aggregation_c_i": weights,
                "fit_train_loss": metrics.get("train_loss"),
                "global_input_hash": self._last_broadcast_hash,
                "global_output_hash": global_output_hash,
            }
        )
        self._logger.log_op(
            {
                "event": "round_fit_complete",
                "round": server_round,
                "duration_seconds": time.time() - self._round_start_times.get(server_round, time.time()),
                "selected_clients": list(self._last_round_client_ids),
                "participating_clients": list(result_by_client_id.keys()),
                "completed_clients": applied_client_ids,
                "failed_clients": [cid for cid, r in responses.items() if r.is_failed],
                "stragglers": [r.client_id for r in new_stragglers],
                "num_examples_total": int(sum(fr.num_examples for fr in result_by_client_id.values())),
                "status": "aggregated" if applied_client_ids else "no_contributions_this_round",
            }
        )

        # ---- Dedicated per-client straggler log (Requirement/Issue 5) ----
        for client_id in self._last_round_client_ids:
            client = self._clients[client_id]
            log_client_round_outcome(
                logger=self._logger,
                round_num=server_round,
                client=client,
                response=responses.get(client_id),
                r_before=self._round_r_before.get(client_id, client.R),
                n_eff_before=self._round_n_eff_before.get(client_id, client.N_eff),
                aggregation_weight=weights.get(client_id),
                contributed_to_aggregation=client_id in applied_client_ids,
                current_window_w_t=self._straggler_manager.current_W_t,
            )
        log_round_summary(
            logger=self._logger,
            round_num=server_round,
            w_t=self._straggler_manager.current_W_t,
            selected_client_ids=list(self._last_round_client_ids),
            on_time_client_ids=on_time_ids_list,
            straggler_client_ids=[r.client_id for r in new_stragglers],
            failed_client_ids=[cid for cid, r in responses.items() if r.is_failed],
            buffered_client_ids=list(self._straggler_manager.get_buffer_state().keys()),
            drained_client_ids=drained_ids,
            expired_client_ids=None,  # not exposed by StragglerManager.manage_window()'s return value
            n_aggregated=len(applied_client_ids),
            quorum_threshold=quorum_threshold,
        )

        return ndarrays_to_parameters(aggregated_ndarrays), metrics

    # ------------------------------------------------------------------
    # configure_evaluate -- pure Flower orchestration (reuses this
    # round's selected set; no distinct evaluation-selection algorithm
    # is specified anywhere in the blueprint)
    # ------------------------------------------------------------------

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        available = client_manager.all()
        config: Dict[str, Scalar] = {}
        evaluate_ins = EvaluateIns(parameters, config)
        return [
            (available[self._client_id_to_cid[cid]], evaluate_ins)
            for cid in self._last_round_client_ids
            if cid in self._client_id_to_cid
        ]

    # ------------------------------------------------------------------
    # aggregate_evaluate -- pure Flower orchestration (generic weighted
    # average of returned metrics; not a research algorithm)
    # ------------------------------------------------------------------

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, "EvaluateRes"]],
        failures: List[Union[Tuple[ClientProxy, "EvaluateRes"], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            self._logger.log_op(
                {"event": "round_evaluate_complete", "round": server_round, "status": "no_results"}
            )
            return None, {}

        total_examples = sum(evaluate_res.num_examples for _, evaluate_res in results)
        loss = (
            sum(evaluate_res.num_examples * evaluate_res.loss for _, evaluate_res in results)
            / total_examples
            if total_examples > 0
            else None
        )
        metrics = weighted_average(
            [(evaluate_res.num_examples, evaluate_res.metrics) for _, evaluate_res in results]
        )

        # ====================================================================
        # Phase 5: federated GLOBAL subgroup snapshot (Requirement 4/5).
        #
        # Each client already computed its OWN local per-category confusion
        # counts (client/client.py's evaluate(), via
        # monitoring.subgroup_metrics.local_all_axes_confusion_counts) and
        # returned only those small integer counts -- never raw predictions
        # or metadata. This is the server summing those counts across
        # clients and computing rates ONCE from the sums (mathematically
        # exact, unlike averaging each client's own rate -- see
        # monitoring/subgroup_metrics.py's module docstring). This is the
        # LOCAL-client-metrics -> GLOBAL-federated-metrics boundary;
        # evaluate.py's centralized subgroup tables (Requirement "if a
        # global centralized evaluation dataset is used, document that")
        # are a SEPARATE, server-held-test-set computation, not this one.
        # ====================================================================
        client_reports = []
        for _, evaluate_res in results:
            raw = evaluate_res.metrics.get("subgroup_confusion_counts_json")
            if raw:
                try:
                    client_reports.append(json.loads(raw))
                except (TypeError, ValueError):
                    pass

        fairness_info: Optional[Dict[str, object]] = None
        if client_reports:
            subgroup_dfs = aggregate_client_confusion_reports(client_reports)
            fairness_info = compute_fairness_gap(subgroup_dfs["skin_type"], metric_col="sensitivity")
            for axis, df in subgroup_dfs.items():
                path = os.path.join(self._results_dir, self._run_id, "subgroup", f"round_{server_round:04d}_{axis}.csv")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                df.to_csv(path, index=False)

        self._logger.log_math(
            {
                "event": "round_evaluate",
                "round": server_round,
                "eval_accuracy": metrics.get("accuracy"),
                "eval_roc_auc": metrics.get("roc_auc"),
                "eval_pr_auc": metrics.get("pr_auc"),
                "eval_recall": metrics.get("recall"),
                "eval_specificity": metrics.get("specificity"),
                "eval_f1": metrics.get("f1"),
                "eval_balanced_accuracy": metrics.get("balanced_accuracy"),
                "eval_loss": loss,
                "fitzpatrick_fairness_gap": fairness_info["gap"] if fairness_info else None,
                "fairness_best_group": fairness_info["best_group"] if fairness_info else None,
                "fairness_worst_group": fairness_info["worst_group"] if fairness_info else None,
                "fairness_excluded_groups": fairness_info["excluded_groups"] if fairness_info else None,
            }
        )
        self._logger.log_op(
            {
                "event": "round_evaluate_complete",
                "round": server_round,
                "participating_clients": [
                    self._cid_to_client_id.get(proxy.cid) for proxy, _ in results
                ],
                "num_examples_total": int(total_examples),
                "status": "ok",
            }
        )

        # ---- Enrich this round's already-saved checkpoint with validation
        # metrics, and update best_global_model.pth (ROC-AUC criterion,
        # matching train.py's own convention -- Requirement 1). ----
        if self._last_aggregated_ndarrays is not None and server_round == self._last_aggregated_round:
            save_global_checkpoint(
                round_num=server_round,
                ndarrays=self._last_aggregated_ndarrays,
                run_id=self._run_id,
                metrics=metrics,
                param_keys=self._param_keys,
                checkpoint_dir=self._checkpoint_dir,
            )
            became_best = maybe_update_best_checkpoint(
                round_num=server_round,
                ndarrays=self._last_aggregated_ndarrays,
                run_id=self._run_id,
                metrics=metrics,
                best_state=self._best_checkpoint_state,
                param_keys=self._param_keys,
                checkpoint_dir=self._checkpoint_dir,
            )
            self._logger.log_audit(
                {
                    "event": "checkpoint_enriched",
                    "round": server_round,
                    "eval_metrics_attached": True,
                    "became_best_checkpoint": became_best,
                    "roc_auc": metrics.get("roc_auc"),
                }
            )

        return loss, metrics

    # ------------------------------------------------------------------
    # evaluate -- centralized evaluation is not used (no dataset is held
    # by this Strategy; all evaluation is client-side per Blueprint's
    # federated design)
    # ------------------------------------------------------------------

    def evaluate(
        self, server_round: int, parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return None
