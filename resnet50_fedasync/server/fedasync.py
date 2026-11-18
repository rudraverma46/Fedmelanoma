"""FedAsync: asynchronous federated optimization (Xie, Koyejo, Gupta, arXiv 1903.03934, Algorithm 1).

Server (updater thread), for every client update that arrives, in arrival order:

    alpha_t = alpha * s(t - tau)                                   (staleness-adaptive mixing weight)
    x_t     = (1 - alpha_t) * x_{t-1} + alpha_t * x_new            (mixing update)

    x_new : the model the client pushed, trained from the global model x_tau it was sent
    tau   : the global version (timestamp) the client started from
    t     : the global version the update produces; t - tau is the staleness

Staleness functions (paper, Section 5 / experiments), s(0) = 1 and non-increasing:

    constant    s(t - tau) = 1
    polynomial  s_a(t - tau) = (t - tau + 1)^(-a)
    hinge       s_{a,b}(t - tau) = 1                              if t - tau <= b
                                 = 1 / (a * (t - tau - b) + 1)      otherwise

Worker: minimises g_{x_tau}(x; z) = f(x; z) + (rho / 2) * ||x - x_tau||^2 for H local steps, starting from
x_tau. That proximal term is implemented by client/proximal.py (copied from resnet50_fedprox_textbook, where
mu plays the role of rho); this module never touches the client.

Staleness convention used here (documented deviation-free choice of indexing). The server keeps an integer
version v, the number of updates applied so far. A client dispatched when the version is v_d carries
tau = v_d. When its update is applied at server version v (before applying), staleness = v - tau, so an
update applied with no other update in between has staleness 0 and s = 1, as the paper requires
("s(t - tau) should be 1 when t = tau"). The new version is v + 1.

Settings chosen by the user (2026-11-13): alpha = 0.6, hinge staleness with a = 10, b = 4, rho = 0.005.
The paper accepts every update; no update is rejected here either (a rejection rule would be an addition
to the method). Every update is still recorded with accepted = True so the count of rejections is
explicit (zero).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import numpy as np

PAPER = "Xie, Koyejo, Gupta. Asynchronous Federated Optimization. arXiv:1903.03934 (v5), Algorithm 1"


def staleness_constant(staleness: int) -> float:
    return 1.0


def staleness_polynomial(staleness: int, a: float = 0.5) -> float:
    return float((staleness + 1) ** (-a))


def staleness_hinge(staleness: int, a: float = 10.0, b: float = 4.0) -> float:
    if staleness <= b:
        return 1.0
    return float(1.0 / (a * (staleness - b) + 1.0))


def make_staleness_function(name: str, a: float = 0.0, b: float = 0.0) -> Callable[[int], float]:
    if name == "constant":
        return staleness_constant
    if name == "polynomial":
        return lambda s: staleness_polynomial(s, a)
    if name == "hinge":
        return lambda s: staleness_hinge(s, a, b)
    raise ValueError(f"unknown staleness function {name!r}")


def mix(global_arrays: Sequence[np.ndarray], client_arrays: Sequence[np.ndarray], alpha_t: float) -> List[np.ndarray]:
    """x_t = (1 - alpha_t) x_{t-1} + alpha_t x_new, for every tensor of the model state.

    Floating tensors (weights, biases, BatchNorm running statistics) are mixed in float64 and stored back
    in their own dtype. Integer tensors (BatchNorm num_batches_tracked, a counter that the model never
    uses because every BatchNorm has a momentum) are mixed the same way and rounded to the nearest integer.
    """
    if len(global_arrays) != len(client_arrays):
        raise ValueError("global and client models have different numbers of tensors")
    out = []
    for g, c in zip(global_arrays, client_arrays):
        if g.shape != c.shape:
            raise ValueError(f"shape mismatch {g.shape} vs {c.shape}")
        mixed = (1.0 - alpha_t) * g.astype(np.float64) + alpha_t * c.astype(np.float64)
        if g.dtype.kind in "iu":
            mixed = np.rint(mixed)
        out.append(mixed.astype(g.dtype))
    return out


@dataclass
class UpdateRecord:
    client_id: int
    tau: int
    version_before: int
    version_after: int
    staleness: int
    s_value: float
    alpha_t: float
    accepted: bool
    arrival_time: float
    dispatch_time: float


class FedAsyncServer:
    """The updater: holds x and its version, applies each arriving update immediately."""

    def __init__(self, initial: Sequence[np.ndarray], alpha: float, staleness_fn: Callable[[int], float]) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.x: List[np.ndarray] = [a.copy() for a in initial]
        self.version: int = 0
        self.alpha = float(alpha)
        self.staleness_fn = staleness_fn

    def snapshot(self) -> tuple:
        """What the scheduler sends a triggered worker: (x_tau, tau)."""
        return [a.copy() for a in self.x], self.version

    def apply(self, client_id: int, client_arrays: Sequence[np.ndarray], tau: int,
              arrival_time: float = 0.0, dispatch_time: float = 0.0) -> UpdateRecord:
        if tau > self.version:
            raise ValueError("an update cannot start from a future version")
        staleness = self.version - tau
        s_value = float(self.staleness_fn(staleness))
        alpha_t = self.alpha * s_value
        self.x = mix(self.x, client_arrays, alpha_t)
        record = UpdateRecord(client_id=int(client_id), tau=int(tau), version_before=self.version,
                              version_after=self.version + 1, staleness=int(staleness), s_value=s_value,
                              alpha_t=float(alpha_t), accepted=True, arrival_time=float(arrival_time),
                              dispatch_time=float(dispatch_time))
        self.version += 1
        return record

    def state(self) -> Dict:
        return {"x": self.x, "version": self.version, "alpha": self.alpha}
