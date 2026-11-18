"""Example-count-weighted average of client metrics, copied verbatim from
resnet50_fedavg_textbook/server/server.py (weighted_average). The same function averages the per-round
evaluation metrics of the textbook FedAvg run, so the curves are computed the same way.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from flwr.common import Metrics


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate client-reported metrics via a simple example-count-weighted
    average -- this is standard, generic FedAvg metric bookkeeping (as
    shown in Flower's own quickstart examples), not the FedMelanoma
    aggregation formula in server/aggregation.py.

    Each `metrics` entry is (num_examples, metrics_dict) as returned by
    FedMelanomaClient.fit() / .evaluate() (Phase 3A). Keys not present on
    a given client are treated as 0.0 for that client's contribution.

    Phase 5 fix: FedMelanomaClient now also returns a few NON-numeric
    diagnostic fields (parameter hashes, a JSON-encoded subgroup-metrics
    blob -- see client/client.py) alongside the numeric ones. A key is
    only included in the weighted average if every client that reported
    it reported a real number; non-numeric keys are skipped here rather
    than crashing (the original implementation unconditionally did
    `num_examples * value`, which raises TypeError on a str value once
    divided by total_examples). Strategy code that needs those non-
    numeric fields (server/strategy.py) reads them directly from each
    client's raw FitRes/EvaluateRes.metrics instead of through this
    average.
    """
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    keys = set()
    for _, client_metrics in metrics:
        keys.update(client_metrics.keys())

    aggregated: Dict[str, float] = {}
    for key in keys:
        values = [client_metrics.get(key) for _, client_metrics in metrics]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values if v is not None):
            continue  # non-numeric (e.g. a parameter hash or JSON blob) -- not averageable
        aggregated[key] = (
            sum(num_examples * client_metrics.get(key, 0.0) for num_examples, client_metrics in metrics)
            / total_examples
        )
    return aggregated
