"""
monitoring/straggler_logging.py
=================================
Dedicated straggler/reliability logging: one JSONL record per selected
client per round (logs/op/stragglers/<run_id>.jsonl via
ExperimentLogger.log_straggler), plus one round-level summary record per
round (also in the same file, event="round_summary").

This module only READS state already computed by the untouched research
modules (selector.Client's R/D/N_eff, straggler.ClientResponse,
aggregation.compute_aggregation_weights' c_i, StragglerManager's W_t) --
it does not compute, adjust, or duplicate any of that math itself.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from monitoring.experiment_logging import ExperimentLogger


def log_client_round_outcome(
    logger: ExperimentLogger,
    round_num: int,
    client,  # server.selector.Client
    response,  # server.straggler.ClientResponse, or None if never selected
    r_before: float,
    n_eff_before: float,
    aggregation_weight: Optional[float],
    contributed_to_aggregation: bool,
    current_window_w_t: float,
) -> None:
    """
    One record per selected client per round -- enough to reconstruct
    exactly what happened to every selected client in every round
    (Requirement: "logging must allow us to reconstruct what happened to
    every selected client in every round").
    """
    if response is None:
        classification = "not_selected"
        latency = None
        staleness = None
        is_failed = None
        is_straggler = None
    else:
        is_failed = bool(response.is_failed)
        is_straggler = bool(response.is_straggler)
        classification = "failed" if is_failed else ("straggler" if is_straggler else "on_time")
        latency = None if math.isinf(response.response_time) else response.response_time
        staleness = None if math.isinf(response.staleness) else response.staleness

    logger.log_straggler(
        {
            "event": "client_round_outcome",
            "round": round_num,
            "client_id": client.client_id,
            "tier": client.tier,
            "n_samples": client.n_samples,
            "selected": response is not None,
            "response_classification": classification,
            "latency_seconds": latency,
            "latency_threshold_w_t": current_window_w_t,
            "staleness": staleness,
            "reliability_R_before": r_before,
            "reliability_R_after": client.R,
            "n_eff_before": n_eff_before,
            "n_eff_after": client.N_eff,
            "diversity_D_static": client.D,
            "contributed_to_aggregation": contributed_to_aggregation,
            "aggregation_weight_c_i": aggregation_weight,
        }
    )


def log_round_summary(
    logger: ExperimentLogger,
    round_num: int,
    w_t: float,
    selected_client_ids: List[int],
    on_time_client_ids: List[int],
    straggler_client_ids: List[int],
    failed_client_ids: List[int],
    buffered_client_ids: List[int],
    drained_client_ids: List[int],
    expired_client_ids: List[int],
    n_aggregated: int,
    quorum_threshold: Optional[int] = None,
) -> None:
    """Round-level straggler/window summary, one record per round."""
    logger.log_straggler(
        {
            "event": "round_summary",
            "round": round_num,
            "adaptive_window_w_t": w_t,
            "quorum_threshold": quorum_threshold,
            "selected_set": selected_client_ids,
            "on_time_set": on_time_client_ids,
            "straggler_set": straggler_client_ids,
            "failed_set": failed_client_ids,
            "buffered_set": buffered_client_ids,
            "drained_set": drained_client_ids,
            "expired_set": expired_client_ids,
            "n_selected": len(selected_client_ids),
            "n_on_time": len(on_time_client_ids),
            "n_straggler": len(straggler_client_ids),
            "n_failed": len(failed_client_ids),
            "n_aggregated": n_aggregated,
        }
    )
