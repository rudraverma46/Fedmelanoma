# resnet50_fedlesscan_novelty_env: FedLesScan strategy (server/fedlesscan.py, a port of the official code) driven by the
# seeded Beta/Bernoulli/LogNormal straggler environment (server/environment.py). Each round FedLesScan selects 6 of 10
# clients; all selected clients train, then the environment decides on time / late (> 30 s) / failed. Late updates arrive in
# a later round and are aggregated with the official staleness rule (accepted when t - t_k <= 2, score (t_k+1)/(t+1)).
# No novelty-method component (diversity prior, reliability score, Boltzmann selector, adaptive window, buffer, novelty
# aggregation weights) is constructed, imported, called or logged.
"""
server/strategy.py -- Flower Strategy for FedLesScan (adapter / orchestration only).

All selection, bookkeeping and aggregation rules live in server/fedlesscan.py; the straggler environment in
server/environment.py. This file converts Flower objects, calls those two modules and writes the logs.
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
from server.environment import TraceEnvironment
from server.fedlesscan import OFFICIAL_COMMIT, ClientResult as FedLesScanResult, FedLesScan
import server.fedlesscan as fedlesscan_constants

# server/server.py's weighted_average() is pure Flower-generic metric
# bookkeeping (Phase 3B: example-count-weighted mean of a metrics dict),
# NOT a research algorithm -- reused here rather than duplicated, per
# "prefer calling existing functions over copying code".
from server.server import weighted_average

# ---------------------------------------------------------------------------
# Phase 5 ADAPTER modules -- monitoring/checkpointing/logging only. These
# record FedLesScan quantities (client histories, scores, effective weights) for logging; they compute nothing.
# ---------------------------------------------------------------------------
from monitoring.param_hash import hash_ndarrays
from monitoring.experiment_logging import ExperimentLogger, new_run_id
from monitoring.checkpointing import (
    save_global_checkpoint,
    maybe_update_best_checkpoint,
    DEFAULT_CHECKPOINT_DIR,
)
from monitoring.subgroup_metrics import aggregate_client_confusion_reports, compute_fairness_gap


# ===========================================================================
# ADAPTER: static hospital registration metadata
# ===========================================================================

@dataclass
class HospitalConfig:
    """What the registry knows about a hospital: sample count, straggler tier and its latency parameters."""
    n_samples: int
    tier: str
    latency_mu: float
    latency_sigma: float


@dataclass
class ClientInfo:
    client_id: int
    n_samples: int
    tier: str


# ===========================================================================
# FedMelanomaStrategy
# ===========================================================================

class FedMelanomaStrategy(Strategy):
    """
    Flower Strategy for FedLesScan. Selection, client histories and staleness-aware aggregation are in
    server/fedlesscan.py; the straggler environment (failure and latency draws) in server/environment.py.
    This class only converts Flower objects, calls them and writes the logs.
    """

    def __init__(
        self,
        hospital_configs: Sequence[HospitalConfig],
        n_select: Optional[int] = None,
        initial_parameters: Optional[List[np.ndarray]] = None,
        local_epochs: int = 1,
        rng: Optional[np.random.Generator] = None,
        experiment_logger: Optional[ExperimentLogger] = None,
        run_id: Optional[str] = None,
        checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
        results_dir: str = "results",
        param_keys: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
        max_rounds: int = 50,
        round_deadline_seconds: float = 30.0,
        allowed_stragglers: int = 6,
        staleness_tolerance: int = 2,
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

        self._clients: Dict[int, ClientInfo] = {
            client_id: ClientInfo(client_id=client_id, n_samples=hc.n_samples, tier=hc.tier)
            for client_id, hc in enumerate(self._hospital_configs)
        }

        # ---- FedLesScan (server/fedlesscan.py): the only federated algorithm this run uses ----
        self._fedlesscan: FedLesScan = FedLesScan(
            client_ids=sorted(self._clients),
            n_clients_in_round=self._n_select,
            max_rounds=max_rounds,
            seed=42 if seed is None else int(seed),
            allowed_stragglers=allowed_stragglers,
            tolerance=staleness_tolerance,
        )
        # novelty_env: seeded per-client Beta/Bernoulli/LogNormal trace (seed 42, stream 7919) and a fixed deadline.
        self._round_deadline: float = float(round_deadline_seconds)
        self._env = TraceEnvironment(
            42 if seed is None else int(seed),
            [{"tier": hc.tier, "latency_mu": hc.latency_mu, "latency_sigma": hc.latency_sigma} for hc in self._hospital_configs],
        )
        self._round_plan: Dict[int, dict] = {}   # per-round outcome record (read by tools/dry_run_schedule.py)
        self._last_round_failed_ids: List[int] = []
        self._last_round_selected_ids: List[int] = []
        self._last_round_selection_info: dict = {}
        self._last_round_eval_drawn: List[int] = []

        # ---- ADAPTER STATE ----
        # Flower ClientProxy.cid <-> integer client_id
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
        self._last_round_dropped_ids: List[int] = []

        # w^t for the round currently in flight (the Parameters handed
        # to configure_fit), needed by aggregate_fit to compute each
        # contributing client's delta per the blueprint's boxed formula.
        self._current_global_ndarrays: Optional[List[np.ndarray]] = None

        # This round's returned parameters, keyed by research client_id.
        # Populated in aggregate_fit when each FitRes arrives and emptied
        # at the end of the same aggregate_fit: nothing survives a round,
        # so a late update can never reach a later round's aggregate.
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
                "fedlesscan": {
                    "official_commit": OFFICIAL_COMMIT,
                    "clients_per_round": self._n_select,
                    "max_rounds": max_rounds,
                    "round_deadline_seconds": self._round_deadline,
                    "environment": "novelty tiers; failure after training; deadline on simulated latency",
                    "renormalised_aggregation": False,
                    "allowed_stragglers": allowed_stragglers,
                    "staleness_tolerance": staleness_tolerance,
                    "training_time_ema_alpha": fedlesscan_constants.TRAINING_TIME_EMA_ALPHA,
                    "missed_round_ema_alpha": fedlesscan_constants.MISSED_ROUND_EMA_ALPHA,
                    "missed_round_penalty_scale": fedlesscan_constants.MISSED_ROUND_PENALTY_SCALE,
                    "dbscan_min_samples": fedlesscan_constants.DBSCAN_MIN_SAMPLES,
                    "eps_grid": [fedlesscan_constants.EPS_START, fedlesscan_constants.EPS_STOP, fedlesscan_constants.EPS_STEP],
                },
            }
        )

    # ------------------------------------------------------------------
    # ADAPTER: Flower ClientProxy.cid <-> client_id mapping
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
                f"These counts must match 1:1 so each client maps "
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
    # configure_fit -- FedLesScan client selection
    # ------------------------------------------------------------------

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Flower lifecycle callback #1 (per round): FedLesScan.select() (rookies first, then DBSCAN clusters of known
        clients ordered fastest first, then clients in cooldown) chooses the clients that train this round.
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

        # ---- FedLesScan selection (Intelligent_selection.py db_fit), 0-based round ----
        # Every selected client trains (train first, then the environment decides its outcome).
        client_ids = sorted(self._clients)
        selected_ids, info = self._fedlesscan.select(server_round - 1)
        self._last_round_selected_ids = list(selected_ids)
        self._last_round_failed_ids = []          # novelty_env: nobody fails before training
        self._last_round_dropped_ids = []
        self._last_round_client_ids = list(selected_ids)
        self._last_round_selection_info = info
        selected_clients: List[ClientInfo] = [self._clients[c] for c in self._last_round_client_ids]
        self._logger.log_math({"event": "fedlesscan_selection", "round": server_round, **info,
                               "selected": list(selected_ids), "failed_at_invocation": list(self._last_round_failed_ids)})
        self._logger.log_op(
            {
                "event": "round_configure_fit",
                "round": server_round,
                "clients_addressed": list(selected_ids),
                "dropped_before_training": list(self._last_round_failed_ids),
                "selected_clients": list(self._last_round_client_ids),
                "not_selected": [c for c in client_ids if c not in selected_ids],
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
    # aggregate_fit -- FedLesScan bookkeeping and staleness-aware aggregation
    # ------------------------------------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, "FitRes"]],
        failures: List[Union[Tuple[ClientProxy, "FitRes"], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Flower lifecycle callback #2 (per round): every selected client has trained. The environment (server/environment.py)
        draws its outcome: failed (discarded), on time (latency <= deadline) or late. Late updates arrive in a later round
        (FedLesScan.arrivals). FedLesScan then updates histories and aggregates the stored results with its official rule
        (accepted when t - t_k <= tolerance, score (t_k + 1)/(t + 1), sample counts, no renormalisation).
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

        # ---- FedLesScan bookkeeping and staleness-aware aggregation ----
        # Client side (official client.py:213-227): each finished client stores its result and its
        # training time in minutes. Controller side (serverless_strategy.py:252-293): successes reset
        # their cooldown; selected clients that did not finish get the round as missed and a cooldown.
        round0 = server_round - 1
        deadline = self._round_deadline
        on_time_ids: List[int] = []
        late_ids: List[int] = []
        failed_ids: List[int] = []
        job_log: Dict[int, dict] = {}
        # Every selected client trained (train first, then the failure draw, as in the novelty run).
        for cid in self._last_round_selected_ids:
            k = self._fedlesscan.next_job_index(cid)
            is_failed, latency = self._env.draw(cid, k)
            if cid not in result_by_client_id:
                failed_ids.append(cid)
                job_log[cid] = {"job": k, "outcome": "missing_flower_result"}
                continue
            if is_failed:
                failed_ids.append(cid)
                job_log[cid] = {"job": k, "outcome": "failed_after_training"}
                continue
            fit_res = result_by_client_id[cid]
            card = int(fit_res.num_examples)
            if latency <= deadline:
                on_time_ids.append(cid)
                job_log[cid] = {"job": k, "outcome": "on_time", "latency_s": latency}
                self._fedlesscan.store_result(FedLesScanResult(
                    client_id=cid, round_id=round0, cardinality=card, parameters=self._pending_client_ndarrays[cid]))
                self._fedlesscan.record_client_success(cid, round0, latency / 60.0, card)
            else:
                lag = int(math.ceil(latency / deadline)) - 1
                late_ids.append(cid)
                job_log[cid] = {"job": k, "outcome": "late", "latency_s": latency, "arrives_round0": round0 + lag}
                self._fedlesscan.launch_late(cid, round0, round0 + lag, card, self._pending_client_ndarrays[cid], latency / 60.0)
        # Late updates whose simulated finish falls in this round: stored with their own round_id.
        arrived = self._fedlesscan.arrivals(round0)
        for job in arrived:
            self._fedlesscan.store_result(FedLesScanResult(
                client_id=job["client_id"], round_id=job["round_id"], cardinality=job["cardinality"], parameters=job["parameters"]))
            self._fedlesscan.record_client_success(job["client_id"], job["round_id"], job["minutes"], job["cardinality"])
        succeeded = list(on_time_ids)
        invocation = self._fedlesscan.finish_round(round0, self._last_round_selected_ids, succeeded)
        aggregated_ndarrays, aggregation_record = self._fedlesscan.aggregate(round0)
        if aggregated_ndarrays is None:
            aggregated_ndarrays = [a.copy() for a in old_ndarrays]
            self._logger.log_op({"event": "round_no_contributions_model_kept", "round": server_round})
        self._logger.log_math({"event": "fedlesscan_jobs", "round": server_round, "jobs": job_log,
                               "on_time": on_time_ids, "late": late_ids, "failed": failed_ids,
                               "arrived_late": [{"client_id": j["client_id"], "round_id0": j["round_id"]} for j in arrived],
                               "in_flight_after": [{"client_id": j["client_id"], "arrival_round0": j["arrival_round"]}
                                                   for j in self._fedlesscan.in_flight]})

        self._round_plan[server_round] = {
            "selected": list(self._last_round_selected_ids), "on_time": list(on_time_ids), "late": list(late_ids),
            "failed": list(failed_ids), "arrived_late": [[j["client_id"], j["round_id"] + 1] for j in arrived],
            "aggregated": [[int(r["client_id"]), int(r["round_id"]) + 1] for r in aggregation_record],
            "job_log": {str(c): v for c, v in job_log.items()},
        }

        weights: Dict[int, float] = {}
        for r in aggregation_record:
            weights[int(r["client_id"])] = weights.get(int(r["client_id"]), 0.0) + float(r["effective_weight"])
        applied_client_ids: List[int] = [int(r["client_id"]) for r in aggregation_record]
        self._logger.log_math(
            {
                "event": "fedlesscan_aggregation",
                "round": server_round,
                "round0": round0,
                "tolerance": self._fedlesscan.tolerance,
                "candidates": aggregation_record,
                "sum_effective_weights": float(sum(weights.values())),
                "all_scores_equal_one": all(r["score"] == 1.0 for r in aggregation_record),
                "invocation": invocation,
                "histories_after": {
                    cid: {"training_times_min": h.training_times, "missed_rounds0": h.missed_rounds,
                          "client_backoff": h.client_backoff}
                    for cid, h in self._fedlesscan.histories.items()
                },
            }
        )
        self._pending_client_ndarrays.clear()

        metrics = weighted_average(
            [(fit_res.num_examples, fit_res.metrics) for fit_res in result_by_client_id.values()]
        )
        metrics["fedmelanoma_clients_aggregated"] = len(applied_client_ids)
        metrics["fedmelanoma_new_stragglers_buffered"] = len(late_ids)

        # ====================================================================
        # Phase 5: hash-chain closure, checkpointing, and structured logging.
        # Everything below only records the values computed above and persists the aggregated parameters.
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
        on_time_ids_list = list(on_time_ids)
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
        self._logger.log_math(
            {
                "event": "round_aggregate",
                "round": server_round,
                "round_deadline_s": float(deadline),  # fixed deadline on the simulated latency
                "n_selected": len(self._last_round_selected_ids),
                "selected": list(self._last_round_selected_ids),
                "n_on_time": len(on_time_ids),
                "n_straggler": len(late_ids),
                "n_failed": len(failed_ids),
                "n_aggregated": len(applied_client_ids),
                "effective_weights": weights,
                "dropped_before_training": [],
                "discarded_after_training": list(failed_ids),
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
                "failed_clients": list(failed_ids),
                "stragglers": list(late_ids),
                "num_examples_total": int(sum(fr.num_examples for fr in result_by_client_id.values())),
                "status": "aggregated" if applied_client_ids else "no_contributions_this_round",
            }
        )

        # ---- Per-client and per-round environment log (logs/op/stragglers/<run>.jsonl) ----
        for client_id in sorted(self._clients):
            job = job_log.get(client_id)
            outcome = None if job is None else {"on_time": "on_time", "late": "late"}.get(job["outcome"], "failed")
            history = self._fedlesscan.histories[client_id]
            self._logger.log_straggler({
                "event": "client_round_outcome", "round": server_round, "client_id": client_id,
                "tier": self._clients[client_id].tier, "n_samples": self._clients[client_id].n_samples,
                "selected": job is not None, "response_classification": outcome if outcome else "not_selected",
                "latency_seconds": None if job is None else job.get("latency_s"),
                "round_deadline_s": float(deadline),
                "training_times_min": list(history.training_times), "missed_rounds0": list(history.missed_rounds),
                "client_backoff": history.client_backoff,
                "contributed_to_aggregation": client_id in applied_client_ids,
                "effective_weight": weights.get(client_id),
            })
        self._logger.log_straggler({
            "event": "round_summary", "round": server_round, "round_deadline_s": float(deadline),
            "selected_set": list(self._last_round_selected_ids), "on_time_set": list(on_time_ids),
            "straggler_set": list(late_ids), "failed_set": list(failed_ids),
            "arrived_late_set": [j["client_id"] for j in arrived], "in_flight": len(self._fedlesscan.in_flight),
            "n_selected": len(self._last_round_selected_ids), "n_on_time": len(on_time_ids), "n_straggler": len(late_ids),
            "n_failed": len(failed_ids), "n_aggregated": len(applied_client_ids),
        })

        return ndarrays_to_parameters(aggregated_ndarrays), metrics

    # ------------------------------------------------------------------
    # configure_evaluate -- pure Flower orchestration (reuses this
    # round's selected set; no distinct evaluation-selection algorithm
    # is specified anywhere in the blueprint)
    # ------------------------------------------------------------------

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        # novelty_env: as in the novelty run, the global model is evaluated on the local test sets of the clients
        # selected (and trained) this round.
        self._ensure_client_mapping(client_manager)
        available = client_manager.all()
        evaluated = list(self._last_round_selected_ids)
        self._last_round_eval_drawn = evaluated
        config: Dict[str, Scalar] = {}
        evaluate_ins = EvaluateIns(parameters, config)
        return [
            (available[self._client_id_to_cid[cid]], evaluate_ins)
            for cid in evaluated
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
