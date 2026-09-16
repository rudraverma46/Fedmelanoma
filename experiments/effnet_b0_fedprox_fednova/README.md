# EfficientNet-B0 + Hybrid FedProx/FedNova

This experiment reproduces the highest-accuracy published approach found
during a literature review of federated learning for melanoma / skin-cancer
classification:

> **Federated Learning for Skin Cancer Detection with Optimized Client
> Selection and Hybrid Aggregation Strategies**
> Springer, Proceedings of ICACECS 2025 — reports ~93.56% validation
> accuracy on ISIC-derived data using an **EfficientNet-B0** backbone with a
> **hybrid FedProx + FedNova** aggregation strategy and a class-aware client
> selection policy.

This repo's canonical v4 architecture (`docs/FedMelanoma_Blueprint_v4.md`)
uses a different, more elaborate scheme: EfficientNet-**B3**, Boltzmann
exposure-aware client scheduling, and staleness-weighted aggregation
(`server/selector.py`, `server/aggregation.py`). This experiment is a
**deliberately isolated alternative** for comparison — it does not modify
or replace any canonical v4 files.

## What's actually implemented

| Component | This experiment | Canonical v4 blueprint |
|---|---|---|
| Backbone | EfficientNet-B0 (1280-d features, ~4.0M params) | EfficientNet-B3 (1536-d features, ~12M params) |
| Client-side objective | FedProx: task loss + `(mu/2)·‖w - w_global‖²` | Class-Weighted Focal Loss only |
| Server-side aggregation | FedNova: per-step-normalized weighted combination | Staleness + diversity weighted (Section 3.10) |
| Client selection | Full participation (all 10 hospitals every round) | Boltzmann exposure-aware sampling (Section 3.8-3.9) |
| Training loop | **Real**: real images, real gradients, real rounds | Simulation only (`server/simulation/`) models client selection/straggler timing, not real training |

**Why full participation instead of the paper's client selection:** the
source paper describes a "class-aware, class-diverse" selection policy but
does not publish its exact scoring formula, so it can't be replicated
precisely. FedProx and FedNova themselves *are* fully and precisely
specified in their original papers, so those are implemented exactly. The
natural next step to get closer to the paper would be swapping in this
repo's own fully-specified Boltzmann selector (`server/selector.py`) as the
selection policy for this training loop — that integration is not done here.

## Files

- `model.py` — `FedMelanomaModelB0`: same image+metadata fusion architecture
  used elsewhere in this repo, with the backbone swapped to EfficientNet-B0.
- `fedprox_fednova.py` — the algorithm core: `local_train_fedprox()` (client)
  and `fednova_aggregate()` (server), with the exact formulas and design
  rationale in the module docstring.
- `train_fedprox_fednova.py` — the full training loop: loads the existing
  10-hospital `hospital_partitions/` layout (reusing `client/dataset.py` and
  `client/focal_loss.py` unchanged), runs T federated rounds, evaluates on
  the pooled global test set each round, and checkpoints the best ROC-AUC.

## Running it

Requires the `hospital_partitions/` directory (10 hospital subfolders, each
with `metadata_train.csv`, `metadata_test.csv`, `images/Train/`,
`images/Test/`) — produced by `datadistribution/` from the raw ISIC images
and `preprocessed_data.csv`. This directory is **not** checked into the
repo (it's several GB of patient-derived images).

```bash
# Fast correctness smoke test (~1 minute, tiny data subset, not meaningful accuracy)
python train_fedprox_fednova.py \
    --data-root /path/to/hospital_partitions \
    --rounds 2 --local-epochs 1 --limit-per-hospital 12

# Real training run
python train_fedprox_fednova.py \
    --data-root /path/to/hospital_partitions \
    --rounds 10 --local-epochs 2 --mu 0.01 --batch-size 16
```

Key hyperparameters (all overridable via CLI flags):

| Flag | Default | Notes |
|---|---|---|
| `--mu` | 0.01 | FedProx proximal coefficient. `0.0` recovers plain FedAvg-style local SGD. |
| `--local-epochs` | 2 | E — local passes per client per round. |
| `--rounds` | 10 | T — federated rounds. |
| `--batch-size` | 16 | Reduced from the centralized script's 20 for CPU feasibility. |
| `--lr` / `--weight-decay` | 1e-4 / 1e-4 | Matches `without_federated/client/train.py`'s AdamW settings. |

## Verified

A 2-round, 1-local-epoch, 12-samples-per-hospital smoke test was run
end-to-end against the real `hospital_partitions/` data on this machine
(CPU-only) to confirm: data loads correctly, FedProx local training runs
without shape/dtype errors, FedNova aggregation combines all 10 clients'
updates correctly (including proper separation of trainable parameters
from BatchNorm running-statistics buffers), evaluation computes valid
metrics, and checkpointing works. Per-hospital training loss decreased
round-over-round for most hospitals, confirming real learning signal.
This tiny run is **not** representative of final accuracy — it exists only
to prove the pipeline is correct before a full-scale run.

## Honest compute note

This repo has no GPU available in the environment this was built in. A
full-scale run (all ~9,740 samples, `--rounds 10 --local-epochs 2`) is
estimated at several hours on CPU alone (EfficientNet-B0 forward+backward
at 224×224, ~7,800 total training images per full pass across all 10
hospitals). GPU hardware would bring this down to a small fraction of that
time. Reaching the paper's exact ~93.56% figure would additionally require
matching their client selection policy and possibly their exact
hyperparameter sweep, neither of which is fully published.
