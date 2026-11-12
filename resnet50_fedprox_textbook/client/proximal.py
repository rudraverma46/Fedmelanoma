"""The FedProx proximal term [Li et al., MLSys 2020]. The only new mathematics in this experiment.

Client k minimises, instead of its task loss L_k(w):

    F_k(w) = L_k(w) + (mu / 2) * ||w - w_t||_2^2

w   : the client's trainable parameters as they stand at the current step.
w_t : the global trainable parameters the client received at the start of this round.

snapshot_global_reference() copies w_t once per round, right after the received parameters are
loaded and before the first optimizer step. The copies are detached, so no gradient reaches them,
and nothing writes to them, so the reference stays fixed across every batch and every local epoch.

The sum runs over model.parameters() with requires_grad: convolution and linear weights and
biases, and BatchNorm scale and shift. BatchNorm running statistics are buffers, not trainable
parameters, so they are not part of w. The norm is squared, as in the paper.
"""
from __future__ import annotations

from typing import List

import torch


def trainable_parameters(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def snapshot_global_reference(model: torch.nn.Module) -> List[torch.Tensor]:
    """Immutable copy of w_t. Call once per round, after the global parameters are loaded."""
    return [p.detach().clone() for p in trainable_parameters(model)]


def proximal_penalty(model: torch.nn.Module, reference: List[torch.Tensor], mu: float) -> torch.Tensor:
    """(mu / 2) * sum_j ||w_j - w_t,j||^2, differentiable with respect to the live parameters."""
    params = trainable_parameters(model)
    if len(params) != len(reference):
        raise ValueError(f"reference holds {len(reference)} tensors, model has {len(params)}")
    squared_distance = params[0].new_zeros(())
    for param, frozen in zip(params, reference):
        squared_distance = squared_distance + (param - frozen).pow(2).sum()
    return 0.5 * mu * squared_distance


@torch.no_grad()
def parameter_distance(model: torch.nn.Module, reference: List[torch.Tensor]) -> float:
    """||w - w_t||_2, for logging only."""
    total = sum(float((p - r).pow(2).sum()) for p, r in zip(trainable_parameters(model), reference))
    return total ** 0.5
