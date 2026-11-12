"""
aggregation.py  ·  Standard FedAvg aggregation weights
======================================================

The Novelty reference's server/aggregation.py with everything except its Section 3.10.1 FedAvg
baseline removed. fedavg_baseline_weights below is that function, unchanged.

Removed from this experiment: compute_aggregation_weights (the diversity and staleness weights
c_i), compute_staleness, apply_staleness_weight, StragglerAggregationWindow, DELTA, GAMMA and the
other constants of that formula.

    c_i = n_i / sum_j n_j        over the clients aggregated this round
    w^{t+1} = w^t + sum_i c_i (w_i - w^t)

Because the c_i sum to 1, the second line equals the textbook form sum_i c_i w_i.
"""

from __future__ import annotations

from typing import Dict, List

from server.straggler import ClientResponse


def fedavg_baseline_weights(
    on_time:    List[ClientResponse],
    client_map: Dict[int, object],       # client_id → Client (needs .n_samples)
) -> Dict[int, float]:
    """
    Standard FedAvg aggregation weights [McMahan et al. 2017] (Section 3.10.1):

        w^{t+1} = w^t + Σ_{i∈At} (n_i / n) · Δw_i

    Synchronous, staleness-blind, diversity-blind.

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
