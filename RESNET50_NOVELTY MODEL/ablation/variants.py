"""Ablation variants, applied to a FedMelanomaStrategy after server_fn builds it.

No reference file is edited. Each variant either subclasses a research class and overrides one
method, or swaps the aggregation-weight function that server/strategy.py looks up at call time.

H1 (common random numbers): the simulated environment of client k in round t is drawn from
default_rng([seed, t, k]) instead of the strategy's shared generator. Every arm then meets the same
failures and latencies for the same (round, client), and selection no longer shifts later draws.
The draws themselves still go through the unmodified StragglerManager.simulate_response.

Blueprint v4 Section 7 arms (A1-A8) use the optional keys in OPTIONAL_VARIANT_DEFAULTS. They are left out of
a run's variant unless set, so adding them changed no existing run id.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

import server.strategy as strategy_module
from ablation.common import OPTIONAL_VARIANT_DEFAULTS
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


class GreedySelector(Selector):
    """A3: deterministic top-K by score (blueprint tau -> 0), via the selector's own greedy_selection."""

    def boltzmann_selection(self, clients: List[Client], n_select: int, temperature: Optional[float] = None,
                            rng: Optional[np.random.Generator] = None) -> List[Client]:
        return self.greedy_selection(clients, n_select)


class BinaryReliability:
    """A4 mixin: R_i <- 1[on-time] (s = 0), replacing the staleness-adjusted EMA. N_eff is unchanged."""

    def update_reliability(self, client: Client, staleness: float) -> None:
        client.R = 1.0 if staleness == 0 else 0.0


class StaticReliability:
    """A5 mixin: R_i = 1 for every client and round. N_eff is unchanged."""

    def update_reliability(self, client: Client, staleness: float) -> None:
        client.R = 1.0


def total_variation_diversity_priors(hospital_configs, global_fitzpatrick, global_modality, global_confirmation,
                                     fitz_weight: float = 0.40, modality_weight: float = 0.35,
                                     confirmation_weight: float = 0.25) -> List[float]:
    """A6: D_i from total variation 0.5 * sum|p - q| per axis instead of OT, same axis weights and max-normalisation."""
    def tv(p, q) -> float:
        p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
        return 0.5 * float(np.abs(p / p.sum() - q / q.sum()).sum())

    raw = [fitz_weight * tv(global_fitzpatrick, hc.fitzpatrick_dist)
           + modality_weight * tv(global_modality, hc.modality_dist)
           + confirmation_weight * tv(global_confirmation, hc.confirmation_dist) for hc in hospital_configs]
    top = max(raw) if raw and max(raw) > 0 else 1.0
    return [r / top for r in raw]


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


class WaitForAll:
    """A8 mixin (synchronous FL): the server waits for every selected client that completes, so late arrivals are
    aggregated this round with s = 0 and nothing is buffered. Failed clients still send nothing."""

    def manage_window(self, clients, responses, current_round, n_selected):
        for response in responses.values():
            if not response.is_failed:
                response.is_straggler = False
        return super().manage_window(clients, responses, current_round, n_selected)


def _aggregation_function(mode: str):
    if mode == "fedmelanoma":
        return compute_aggregation_weights
    if mode == "no_diversity":   # E03: c_i with gamma = 0, size and staleness factors unchanged
        return lambda on_time, client_map, delta, gamma: compute_aggregation_weights(
            on_time, client_map, delta=delta, gamma=0.0)
    if mode == "fedavg":         # E04: n_i / n over the arrived set
        return lambda on_time, client_map, delta, gamma: fedavg_baseline_weights(on_time, client_map)
    raise ValueError(f"unknown aggregation mode {mode!r}")


def _selector(mode: str, attenuation: bool = True, reliability: str = "ema") -> Selector:
    bases = {"boltzmann": Selector, "uniform": UniformSelector, "no_diversity": Selector, "greedy": GreedySelector}
    mixins = {"ema": (), "binary": (BinaryReliability,), "static": (StaticReliability,)}
    if mode not in bases:
        raise ValueError(f"unknown selection mode {mode!r}")
    if reliability not in mixins:
        raise ValueError(f"unknown reliability mode {reliability!r}")
    base = bases[mode]
    cls = base if not mixins[reliability] else type(
        f"{reliability.capitalize()}Reliability{base.__name__}", (*mixins[reliability], base), {})
    kwargs = {}
    if mode == "no_diversity":   # E02 / A1: score = R_i
        kwargs["gamma"] = 0.0
    if not attenuation:          # A2: lambda = 0, no N_eff decay of the diversity bonus
        kwargs["lambda_"] = 0.0
    return cls(**kwargs)


def _straggler_manager(rng: np.random.Generator, seed: int, buffer: bool, crn: bool,
                       synchronous: bool = False) -> StragglerManager:
    base = CommonRandomEnvironment if crn else StragglerManager
    if synchronous:
        cls = type(f"WaitForAll{base.__name__}", (WaitForAll, base), {})
    else:
        cls = base if buffer else type(f"DiscardLate{base.__name__}", (DiscardLateUpdates, base), {})
    return cls(env_seed=seed, rng=rng) if crn else cls(rng=rng)


def option(variant: Dict[str, Any], key: str) -> Any:
    return variant.get(key, OPTIONAL_VARIANT_DEFAULTS[key])


def prepare_variant(variant: Dict[str, Any]) -> None:
    """Changes that must be in place before server_fn builds the strategy (D_i is computed in its constructor)."""
    metric = option(variant, "diversity_metric")
    if metric == "tv":
        strategy_module.compute_normalized_diversity_priors = total_variation_diversity_priors
    elif metric != "ot":
        raise ValueError(f"unknown diversity metric {metric!r}")


def apply_variant(strategy, variant: Dict[str, Any], harness: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Install the variant on a freshly built strategy (before round 1 or before a resume restore)."""
    synchronous = bool(option(variant, "synchronous"))
    if synchronous and not variant["straggler_buffer"]:
        raise ValueError("synchronous and straggler_buffer: false are different arms; set only one")
    strategy._selector = _selector(variant["selection"], bool(option(variant, "attenuation")),
                                   option(variant, "reliability"))
    strategy._straggler_manager = _straggler_manager(strategy._rng, seed, bool(variant["straggler_buffer"]),
                                                     bool(harness.get("crn_environment")), synchronous)
    strategy_module.compute_aggregation_weights = _aggregation_function(variant["aggregation"])
    record = {
        "event": "ablation_variant", "variant": variant, "harness": harness,
        "selector_class": type(strategy._selector).__name__, "selector_gamma": strategy._selector.gamma,
        "selector_lambda": strategy._selector.lambda_, "diversity_metric": option(variant, "diversity_metric"),
        "diversity_priors_D": {c.client_id: c.D for c in strategy._clients.values()},
        "straggler_manager_class": type(strategy._straggler_manager).__name__,
        "aggregation_mode": variant["aggregation"],
    }
    strategy._logger.log_audit(record)
    return record
