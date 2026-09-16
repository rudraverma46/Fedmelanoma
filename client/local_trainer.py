"""
client/local_trainer.py
========================
Reusable local-training components shared between the centralized scripts
(`train.py`, `evaluate.py`) and the future Flower ClientApp.

This module does NOT introduce any new behavior or research logic. Every
function below is either:

  (a) moved verbatim from `train.py` / `evaluate.py` (only relocated), or
  (b) the same loop body that was duplicated across both files, unified
      into a single implementation with a `verbose` flag so each caller's
      original console output is preserved exactly, or
  (c) a trivial extraction of an inline snippet (optimizer construction,
      the OpenCV/PyTorch multiprocessing fix) into a named function.

No model architecture, dataset processing, loss function, metric
definition, or research algorithm (selector / aggregation / straggler /
focal loss / EMD) is touched by this module.

Phase 3 (Flower ClientApp) will import from here instead of duplicating
this logic again — e.g. the "AdamW momentum/variance buffers reset at the
start of each round's local training" requirement in Blueprint §4.3.3
maps directly onto calling `build_optimizer()` fresh each round.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from client.focal_loss import ClassWeightedFocalLoss
from models.model import FedMelanomaModel


# ===========================================================================
# Environment setup
# ===========================================================================

def configure_multiprocessing_env() -> None:
    """
    Fix OpenCV + PyTorch multiprocessing conflict on Linux.

    Previously duplicated verbatim in train.py's main() and evaluate.py's
    build_test_loader(); centralized here so the two copies cannot drift.
    """
    import cv2
    cv2.setNumThreads(0)
    os.environ["OMP_NUM_THREADS"] = "1"


# ===========================================================================
# Optimizer setup
# ===========================================================================

def build_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """
    Construct the AdamW optimizer used for local training.

    Extracted from the inline `torch.optim.AdamW(...)` call in train.py's
    main(). Identical parameters, identical resulting optimizer state.
    """
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )


# ===========================================================================
# Training loop (moved verbatim from train.py, with `device` as an
# explicit parameter instead of a module-level global, so it works
# outside train.py's module scope)
# ===========================================================================

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: ClassWeightedFocalLoss,
    optimizer: torch.optim.Optimizer,
    class_weights: torch.Tensor,
    epoch_start: float,
    device: torch.device,
) -> float:
    """Run one full training epoch and return average batch loss."""
    model.train()
    running_loss: float = 0.0
    num_batches: int = 0

    for batch_idx, (images, meta, labels) in enumerate(loader):

        images = images.to(device)
        meta = meta.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images, meta)

        loss = criterion(
            logits,
            labels,
            class_weights=class_weights,
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

        if batch_idx % 100 == 0:

            progress = (batch_idx + 1) / len(loader)

            elapsed = time.time() - epoch_start

            eta_epoch = elapsed / progress - elapsed

            print(
                f"Epoch Progress: "
                f"{100*progress:.1f}% | "
                f"ETA Epoch: {eta_epoch/60:.2f} min"
            )

    avg_loss: float = running_loss / max(num_batches, 1)
    return avg_loss


# ===========================================================================
# Shared inference core
#
# This loop was duplicated between train.py's `evaluate()` (no progress
# printing) and evaluate.py's `run_inference()` (progress printing with
# ETA every 50 batches). `verbose` reproduces each caller's original
# console output exactly.
# ===========================================================================

def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    verbose: bool = False,
    positive_class_idx: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference over *loader* without gradient computation.

    Returns
    -------
    y_true : (N,) int array of ground-truth labels
    y_pred : (N,) int array of argmax predictions (threshold = 0.50)
    y_prob : (N,) float array of softmax probability for the positive class
    """
    all_labels: list = []
    all_preds:  list = []
    all_probs:  list = []

    n_batches = len(loader)
    t0 = time.time()

    with torch.no_grad():
        for batch_idx, (images, meta, labels) in enumerate(loader):
            images = images.to(device)
            meta   = meta.to(device)
            labels = labels.to(device)

            logits: torch.Tensor = model(images, meta)

            probs = F.softmax(logits, dim=1)[:, positive_class_idx]
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

            if verbose and ((batch_idx + 1) % 50 == 0 or (batch_idx + 1) == n_batches):
                elapsed = time.time() - t0
                pct = (batch_idx + 1) / n_batches
                eta = elapsed / pct - elapsed
                print(
                    f"  Inference: {batch_idx+1}/{n_batches} batches "
                    f"({100*pct:.1f}%)  |  ETA: {eta:.1f}s"
                )

    return (
        np.array(all_labels, dtype=np.int64),
        np.array(all_preds,  dtype=np.int64),
        np.array(all_probs,  dtype=np.float64),
    )


# ===========================================================================
# Validation (moved from train.py's `evaluate()`; the inference loop is
# now delegated to run_inference() instead of being duplicated)
# ===========================================================================

def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    positive_class_idx: int = 1,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on *loader* and compute per-epoch validation metrics.

    Numerically identical to the original train.py `evaluate()`. Returns
    metrics : dict with keys accuracy, precision, recall, f1, roc_auc
    all_labels / all_preds / all_probs as in run_inference().
    """
    model.eval()

    y_true, y_pred, y_prob = run_inference(
        model, loader, device, verbose=False, positive_class_idx=positive_class_idx
    )

    # Core scalar metrics (macro average for multi-class safety; binary here)
    accuracy:  float = accuracy_score(y_true, y_pred)
    precision: float = precision_score(
        y_true, y_pred, pos_label=positive_class_idx, zero_division=0
    )
    recall: float = recall_score(
        y_true, y_pred, pos_label=positive_class_idx, zero_division=0
    )
    f1: float = f1_score(
        y_true, y_pred, pos_label=positive_class_idx, zero_division=0
    )

    # ROC-AUC requires probability scores
    try:
        roc_auc: float = roc_auc_score(y_true, y_prob)
    except ValueError:
        # Edge case: only one class present in labels
        roc_auc = 0.0

    metrics: Dict[str, float] = {
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "roc_auc":   roc_auc,
    }

    return metrics, y_true, y_pred, y_prob


# ===========================================================================
# Checkpoint utilities
# ===========================================================================

def save_checkpoint(
    path: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: Dict[str, float],
) -> None:
    """
    Persist model + optimiser state alongside epoch metrics.

    Moved verbatim from train.py's `save_checkpoint()`.
    """
    checkpoint = {
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "accuracy":             metrics["accuracy"],
        "precision":            metrics["precision"],
        "recall":               metrics["recall"],
        "f1":                   metrics["f1"],
        "roc_auc":              metrics["roc_auc"],
    }
    torch.save(checkpoint, path)
    print(f"  [✓] Checkpoint saved → {path}  (ROC-AUC: {metrics['roc_auc']:.4f})")


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """
    Instantiate FedMelanomaModel, load saved weights, set to eval mode.

    Moved verbatim from evaluate.py's `load_model()`. The checkpoint
    stores both model and optimiser state; only the model state dict is
    needed here.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run train_melanoma.py first to generate the checkpoint."
        )

    model = FedMelanomaModel().to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    saved_epoch   = checkpoint.get("epoch", "unknown")
    saved_roc_auc = checkpoint.get("roc_auc", float("nan"))
    print(f"[✓] Loaded checkpoint: {checkpoint_path}")
    print(f"    Saved at epoch {saved_epoch}  |  "
          f"Validation ROC-AUC at save time: {saved_roc_auc:.4f}")
    return model
