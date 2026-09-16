"""
fedprox_fednova.py — Hybrid FedProx (client) + FedNova (server) federated
optimization, as used by the top-performing published result found for
federated melanoma classification (Springer ICACECS 2025, "Federated
Learning for Skin Cancer Detection with Optimized Client Selection and
Hybrid Aggregation Strategies", ~93.56% accuracy with an EfficientNet-B0
backbone).

Why these two algorithms compose cleanly
-----------------------------------------
FedProx and FedNova modify two DIFFERENT stages of the federated pipeline,
so combining them is not a conflict, it's literally what "hybrid" means
here:

  FedProx  (client-side, Li et al., MLSys 2020)
      Adds a proximal regularization term to each client's LOCAL loss:

          L_i(w) = L_task(w) + (mu / 2) * || w - w_global ||^2

      w_global is a frozen snapshot of the global model at the start of the
      round. This penalizes local weights that drift too far from the
      global model during local training, which directly targets the
      "client drift" problem under Non-IID data (this dataset's per-hospital
      malignancy rates span ~0-35%, a textbook Non-IID setting — see
      ../../docs/FedMelanoma_Blueprint_v4.md Section 1.3 / 2.3).

  FedNova (server-side, Wang et al., NeurIPS 2020)
      Corrects "objective inconsistency": under FedAvg, clients that take
      more local SGD steps (tau_i) end up contributing disproportionately
      to the direction of the global update, biasing convergence toward
      whichever clients trained longest. FedNova normalizes each client's
      cumulative update by its own step count before combining:

          d_i          = (w_i_final - w_global) / tau_i      # per-step direction
          tau_eff      = sum_i p_i * tau_i                    # effective global step
          w_new        = w_global + tau_eff * sum_i p_i * d_i
          p_i          = n_i / sum_j n_j                      # data-size weight

      With uniform local_epochs and batch_size across clients (this repo's
      default), tau_i differs only through per-hospital dataset size
      (n_i -> different number of mini-batches per local epoch), so FedNova
      still meaningfully re-weights compared to plain FedAvg.

Buffer handling (BatchNorm running stats)
------------------------------------------
FedNova's step-normalized update rule is only meaningful for gradient-
descended trainable PARAMETERS. BatchNorm running_mean / running_var /
num_batches_tracked are not optimized by SGD, they're running statistics,
so normalizing them by tau_i has no theoretical grounding. This module
therefore splits every state_dict into:
    - trainable parameters -> FedNova-normalized aggregation
    - buffers              -> simple n_i-weighted average (FedAvg-style)
This split is made explicit via `split_state_keys()` below.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple
import time

import torch
import torch.nn as nn


# ===========================================================================
# Key splitting: trainable parameters vs. buffers
# ===========================================================================

def split_state_keys(model: nn.Module) -> Tuple[List[str], List[str]]:
    """
    Partition a model's state_dict keys into (parameter_keys, buffer_keys).

    Parameters  : returned by model.named_parameters() — updated by the
                  optimizer via gradients (backbone weights, classifier,
                  metadata MLP weights, BatchNorm affine gamma/beta).
    Buffers     : returned by model.named_buffers() — BatchNorm
                  running_mean / running_var / num_batches_tracked. Not
                  gradient-descended; FedNova normalization does not apply.
    """
    param_keys = [name for name, _ in model.named_parameters()]
    buffer_keys = [name for name, _ in model.named_buffers()]
    return param_keys, buffer_keys


# ===========================================================================
# Client-side: FedProx local training
# ===========================================================================

@dataclass
class LocalTrainResult:
    """Outcome of one client's local FedProx training for one round."""
    param_delta: Dict[str, torch.Tensor]   # w_i_final - w_global, params only
    final_buffers: Dict[str, torch.Tensor] # final BatchNorm buffer values
    tau: int                               # number of local SGD steps taken
    n_samples: int                         # local training set size (n_i)
    avg_loss: float                        # mean task loss (diagnostics only;
                                            # excludes the proximal term so
                                            # it's comparable across clients)


def local_train_fedprox(
    model: nn.Module,
    global_state_dict: Dict[str, torch.Tensor],
    dataloader,
    criterion,
    class_weights: torch.Tensor,
    device: torch.device,
    mu: float,
    local_epochs: int,
    lr: float,
    weight_decay: float,
    param_keys: List[str],
    buffer_keys: List[str],
    progress_label: str = "",
    progress_every: int = 5,
) -> LocalTrainResult:
    """
    Run FedProx local training for one client for one federated round.

    Parameters
    ----------
    model             : Freshly constructed model instance (weights will be
                        overwritten with global_state_dict before training).
    global_state_dict : The global model's state_dict at the start of this
                        round — the proximal anchor point w_global.
    dataloader        : This client's local training DataLoader
                        (image, metadata, label) batches, e.g. from
                        client/dataset.py's MelanomaDataset.
    criterion         : Loss function taking (logits, labels,
                        class_weights=...) — e.g. focal_loss.py's
                        ClassWeightedFocalLoss, reused unchanged from the
                        rest of this repo.
    class_weights     : This client's locally computed inverse-frequency
                        class weights (never transmitted raw counts,
                        consistent with docs/FedMelanoma_Blueprint_v4.md
                        Section 3.4.3).
    device            : torch.device for training.
    mu                : FedProx proximal term coefficient. mu=0 recovers
                        plain local SGD (equivalent to FedAvg's client
                        step). Typical effective range: 0.001-1.0; this
                        experiment defaults to 0.01 (see README.md).
    local_epochs      : Number of local passes over this client's data (E).
    lr, weight_decay  : AdamW hyperparameters for the local optimizer.
    param_keys        : Keys in state_dict() that are trainable parameters
                        (from split_state_keys) — FedNova will normalize
                        the delta of exactly these keys.
    buffer_keys       : Keys that are BatchNorm running statistics —
                        passed through as final values, not deltas.

    Returns
    -------
    LocalTrainResult with the parameter delta, final buffer values, the
    number of local SGD steps taken (tau_i, for FedNova normalization), the
    local dataset size (n_i, for size-proportional weighting), and the
    mean task loss for logging.
    """
    model.load_state_dict(global_state_dict)
    model.to(device)
    model.train()

    # Snapshot global TRAINABLE parameters only (buffers excluded — the
    # proximal term is only defined over gradient-descended weights).
    global_params_snapshot = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
    }

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    tau = 0
    running_loss = 0.0
    n_batches = 0
    total_batches_per_epoch = len(dataloader)
    start_time = time.time()

    for epoch_idx in range(local_epochs):
        for batch_idx, (images, meta, labels) in enumerate(dataloader):
            images = images.to(device)
            meta = meta.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(images, meta)
            task_loss = criterion(logits, labels, class_weights=class_weights)

            # --- FedProx proximal term: (mu/2) * ||w - w_global||^2 -------
            if mu > 0.0:
                prox_term = torch.zeros((), device=device)
                for name, p in model.named_parameters():
                    g = global_params_snapshot[name]
                    prox_term = prox_term + (p - g).pow(2).sum()
                loss = task_loss + (mu / 2.0) * prox_term
            else:
                loss = task_loss

            loss.backward()
            optimizer.step()

            tau += 1
            running_loss += float(task_loss.detach().item())
            n_batches += 1

            if progress_label and (batch_idx % progress_every == 0):
                elapsed = time.time() - start_time
                print(
                    f"      [{progress_label}] epoch {epoch_idx + 1}/{local_epochs}  "
                    f"batch {batch_idx + 1}/{total_batches_per_epoch}  "
                    f"avg_loss={running_loss / max(n_batches, 1):.4f}  "
                    f"elapsed={elapsed:.0f}s"
                )

    final_state = model.state_dict()

    param_delta = {
        key: (final_state[key] - global_state_dict[key]).detach().cpu()
        for key in param_keys
    }
    final_buffers = {
        key: final_state[key].detach().cpu().clone()
        for key in buffer_keys
    }

    return LocalTrainResult(
        param_delta=param_delta,
        final_buffers=final_buffers,
        tau=max(tau, 1),
        n_samples=len(dataloader.dataset),
        avg_loss=running_loss / max(n_batches, 1),
    )


# ===========================================================================
# Server-side: FedNova aggregation
# ===========================================================================

def fednova_aggregate(
    global_state_dict: Dict[str, torch.Tensor],
    client_results: List[LocalTrainResult],
    param_keys: List[str],
    buffer_keys: List[str],
) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    Combine per-client local updates into a new global state_dict via
    FedNova's step-normalized aggregation rule (Wang et al., NeurIPS 2020).

        p_i     = n_i / sum_j n_j
        d_i     = param_delta_i / tau_i
        tau_eff = sum_i p_i * tau_i
        w_new   = w_global + tau_eff * sum_i p_i * d_i

    Buffers (BatchNorm running stats) are combined separately via a plain
    n_i-weighted average — see the module docstring for why FedNova's
    per-step normalization does not apply to them.

    Parameters
    ----------
    global_state_dict : The global model's state_dict at the start of the
                        round (w_global for this round).
    client_results     : One LocalTrainResult per participating client.
    param_keys          : Trainable-parameter keys (FedNova-normalized).
    buffer_keys         : BatchNorm buffer keys (plain weighted average).

    Returns
    -------
    (new_global_state_dict, tau_eff) — the updated global weights and the
    effective step count for this round (useful for logging / diagnosing
    how heterogeneous client compute was this round).
    """
    if not client_results:
        raise ValueError("fednova_aggregate: client_results is empty — no updates to aggregate.")

    n_total = sum(r.n_samples for r in client_results)
    if n_total == 0:
        raise ValueError("fednova_aggregate: all clients reported zero samples.")

    weights = [r.n_samples / n_total for r in client_results]
    tau_eff = sum(w * r.tau for w, r in zip(weights, client_results))

    new_state: Dict[str, torch.Tensor] = {}

    # --- Trainable parameters: FedNova step-normalized aggregation --------
    for key in param_keys:
        d_bar = torch.zeros_like(global_state_dict[key], dtype=torch.float32)
        for w, r in zip(weights, client_results):
            d_bar += (w / r.tau) * r.param_delta[key].to(torch.float32)
        new_state[key] = global_state_dict[key] + tau_eff * d_bar

    # --- Buffers: plain n_i-weighted average (FedAvg-style) ----------------
    for key in buffer_keys:
        ref = global_state_dict[key]
        if ref.dtype.is_floating_point:
            acc = torch.zeros_like(ref, dtype=torch.float32)
            for w, r in zip(weights, client_results):
                acc += w * r.final_buffers[key].to(torch.float32)
            new_state[key] = acc.to(ref.dtype)
        else:
            # e.g. num_batches_tracked (int64) — weighted-round to nearest int
            acc = sum(w * float(r.final_buffers[key]) for w, r in zip(weights, client_results))
            new_state[key] = torch.tensor(round(acc), dtype=ref.dtype)

    return new_state, tau_eff


def clone_state_dict_cpu(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Deep-copy a state_dict onto CPU (used for checkpointing the global model)."""
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}
