"""Ablation variants, applied to a FedMelanomaStrategy after server_fn builds it.

No reference file is edited. Each variant either subclasses a research class and overrides one
method, or swaps the aggregation-weight function that server/strategy.py looks up at call time.

H1 (common random numbers): the simulated environment of client k in round t is drawn from
default_rng([seed, t, k]) instead of the strategy's shared generator. Every arm then meets the same
failures and latencies for the same (round, client), and selection no longer shifts later draws.
The draws themselves still go through the unmodified StragglerManager.simulate_response.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

import server.strategy as strategy_module
from server.aggregation import compute_aggregation_weights, fedavg_baseline_weights
from server.selector import Client, Selector
from server.straggler import StragglerManager


class UniformSelector(Selector):
    """E01: n clients uniformly at random without replacement. R_i / N_eff are still updated."""

    def boltzmann_selection(self, clients: List[Client], n_select: int, temperature: Optional[float] = None,
                            rng: Optional[np.random.Generator] = None) -> List[Client]:
        rng = rng or np.random.default_rng()
        picks = rng.choice(len(clients), size=min(n_select, len(clients)), replace=False)
        return [clients[int(i)] for i in picks]


class CommonRandomEnvironment(StragglerManager):
    """H1: per-(seed, round, client) generator for simulate_response; everything else unchanged."""

    def __init__(self, env_seed: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.env_seed = int(env_seed)

    def simulate_response(self, client, current_round: int):
        shared = self.rng
        self.rng = np.random.default_rng([self.env_seed, int(current_round), int(client.client_id)])
        try:
            return super().simulate_response(client, current_round)
        finally:
            self.rng = shared


class DiscardLateUpdates:
    """E05 mixin: late updates are dropped instead of buffered. R_i / N_eff still see s = 1."""

    def manage_window(self, clients, responses, current_round, n_selected):
        on_time, late = super().manage_window(clients, responses, current_round, n_selected)
        for response in late:
            self._buffer.pop(response.client_id, None)
        return on_time, late


def _aggregation_function(mode: str):
    if mode == "fedmelanoma":
        return compute_aggregation_weights
    if mode == "no_diversity":   # E03: c_i with gamma = 0, size and staleness factors unchanged
        return lambda on_time, client_map, delta, gamma: compute_aggregation_weights(
            on_time, client_map, delta=delta, gamma=0.0)
    if mode == "fedavg":         # E04: n_i / n over the arrived set
        return lambda on_time, client_map, delta, gamma: fedavg_baseline_weights(on_time, client_map)
    raise ValueError(f"unknown aggregation mode {mode!r}")


def _selector(mode: str) -> Selector:
    if mode == "boltzmann":
        return Selector()
    if mode == "uniform":
        return UniformSelector()
    if mode == "no_diversity":   # E02: score = R_i
        return Selector(gamma=0.0)
    raise ValueError(f"unknown selection mode {mode!r}")


def _straggler_manager(rng: np.random.Generator, seed: int, buffer: bool, crn: bool) -> StragglerManager:
    base = CommonRandomEnvironment if crn else StragglerManager
    cls = base if buffer else type(f"DiscardLate{base.__name__}", (DiscardLateUpdates, base), {})
    return cls(env_seed=seed, rng=rng) if crn else cls(rng=rng)


def apply_variant(strategy, variant: Dict[str, Any], harness: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Install the variant on a freshly built strategy (before round 1 or before a resume restore)."""
    strategy._selector = _selector(variant["selection"])
    strategy._straggler_manager = _straggler_manager(strategy._rng, seed, bool(variant["straggler_buffer"]),
                                                     bool(harness.get("crn_environment")))
    strategy_module.compute_aggregation_weights = _aggregation_function(variant["aggregation"])
    record = {
        "event": "ablation_variant", "variant": variant, "harness": harness,
        "selector_class": type(strategy._selector).__name__, "selector_gamma": strategy._selector.gamma,
        "straggler_manager_class": type(strategy._straggler_manager).__name__,
        "aggregation_mode": variant["aggregation"],
    }
    strategy._logger.log_audit(record)
    return record
