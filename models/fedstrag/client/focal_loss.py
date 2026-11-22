"""
fedmelanoma/client/focal_loss.py
================================================================================
Section 3.4 — Class-Weighted Focal Loss
FedMelanoma v4 Blueprint, Canonical Implementation
================================================================================

MATHEMATICAL SPECIFICATION (verbatim from blueprint Section 3.4)
─────────────────────────────────────────────────────────────────

§3.4.2  Focal modulation:
    p_t^(j)  =  p̂_{y_j}^(j)                     [true-class probability]
    L_focal  =  −(1 − p_t)^{γ_f} · log(p_t)      [γ_f = 2 canonical]

§3.4.3a  Inverse-frequency weight (computed LOCALLY — never transmitted):
    α_c^(i)  =  n_total^(i) / (2 · n_c^(i))

§3.4.3b  Normalisation (so α̂_0 + α̂_1 = 1):
    α̂_c^(i)  =  α_c^(i) / (α_0^(i) + α_1^(i))

§3.4.3c  Class-Weighted Focal Loss (BOXED formula):

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │   L_CW-Focal  =  −(1/N) Σ_{j=1}^{N}  α̂_{y_j}^(i)                    │
    │                                        · (1 − p_t^(j))^{γ_f}          │
    │                                        · log(p_t^(j))                  │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

NUMERICAL STABILITY STRATEGY
─────────────────────────────
1. log(p_t) is extracted from F.log_softmax, NOT from log(softmax).
   log_softmax uses the log-sum-exp trick: log(softmax(z))_c = z_c − log(Σ exp(z_j))
   This avoids catastrophic cancellation when logits are large-magnitude.

2. p_t = exp(log_pt) is computed AFTER clamping log_pt to [log(eps), 0],
   bounding p_t ∈ [eps, 1]. This prevents log(0) = −∞ and exp(−∞) = 0
   producing 0·(−∞) = NaN in the focal weight computation.

3. (1 − p_t)^{γ_f} is safe since p_t ∈ [eps, 1] implies (1 − p_t) ∈ [0, 1−eps].
   No further clamping needed for the focal weight itself.

PRIVACY CONSTRAINT (Section 3.4.3)
────────────────────────────────────
"Client i computes inverse-frequency weights locally —
 raw class counts are never transmitted."

Implementation: compute_class_weights() is a pure local function. Its outputs
(α̂_c values) are NEVER added to the model's state_dict, NEVER serialised via
pickle, and NEVER included in the Δw_i gradient update transmitted to the server.
The FedMelanomaClient must ensure this by calling compute_class_weights() inside
its local_train() scope only.

BLUEPRINT ALIGNMENT CHECKLIST
──────────────────────────────
✓  p_t = softmax(logits)[y_j]                     — §3.4.2
✓  focal term = (1 − p_t)^{γ_f} · log(p_t)       — §3.4.2
✓  γ_f = 2.0 default                              — §3.4.2, §1.5 diagram
✓  α_c = n_total / (2 · n_c)                      — §3.4.3a
✓  α̂_c = α_c / (α_0 + α_1)                        — §3.4.3b
✓  L = −(1/N) Σ α̂_{y_j} · focal                  — §3.4.3c (boxed)
✓  class counts never transmitted                  — §3.4.3 privacy constraint
✓  num_classes = 2  (benign=0, malignant=1)        — §2.1 schema

NOT IMPLEMENTED HERE (out of scope per task):
✗  Client selection, reliability scores            — §3.5–3.9
✗  Aggregation weights c_i                        — §3.10
✗  Model architecture (EfficientNet-B3)            — §3.3
✗  Temperature calibration                        — §3.11

Dependencies: torch >= 2.0, Python >= 3.11
"""

from __future__ import annotations

import math
from typing import Final, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Module-level constants — directly from blueprint
# ─────────────────────────────────────────────────────────────────────────────

#: Canonical focal parameter γ_f from §3.4.2 and architecture diagram §1.5
GAMMA_F_DEFAULT: Final[float] = 2.0

#: Binary classification per blueprint §2.1 schema (benign=0, malignant=1)
NUM_CLASSES_DEFAULT: Final[int] = 2

#: Numerical floor — guards log(0) and division by zero in weight computation
EPS_DEFAULT: Final[float] = 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# compute_class_weights
# Implements §3.4.3a (α_c) and §3.4.3b (α̂_c)
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(
    targets: Tensor,
    num_classes: int = NUM_CLASSES_DEFAULT,
    eps: float = EPS_DEFAULT,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Compute normalised inverse-frequency class weights α̂_c^(i) on-device.

    This function implements equations §3.4.3a and §3.4.3b of the blueprint.
    It runs entirely on the client device. The resulting weights are NEVER
    serialised or transmitted to the server (privacy constraint §3.4.3).

    Blueprint equations
    ───────────────────
    §3.4.3a — Inverse-frequency weight:

        α_c^(i)  =  n_total^(i) / ( 2 · n_c^(i) )

        Where:
          n_total^(i)  = total samples in client i's local batch/partition
          n_c^(i)      = samples from class c in client i's local data
          2            = num_classes for binary (benign=0, malignant=1)

        Derivation: Inverse frequency. The factor 2 (= num_classes) normalises
        so that the balanced case (n_0 = n_1 = n_total/2) gives:
            α_0 = n_total / (2 · n_total/2) = 1.0
            α_1 = n_total / (2 · n_total/2) = 1.0
        i.e., equal raw weights before normalisation.

        In the heavily imbalanced case from §3.4.3 (n_0=980, n_1=20):
            α_0 = 1000 / (2 · 980) ≈ 0.510
            α_1 = 1000 / (2 · 20)  = 25.0
        The malignant class receives ~49× the raw weight of benign.

    §3.4.3b — Normalisation (sum-to-one):

        α̂_c^(i)  =  α_c^(i) / ( α_0^(i) + α_1^(i) )

        For the imbalanced example:
            α̂_0 = 0.510 / (0.510 + 25.0) ≈ 0.020
            α̂_1 = 25.0  / (0.510 + 25.0) ≈ 0.980

        Sum: α̂_0 + α̂_1 = 1.0 (guaranteed by construction).

        Blueprint confirmation: "amplifying malignant gradient by ≈ 49×"
        → α̂_1 / α̂_0 = 0.980 / 0.020 = 49.0 ✓

    Missing-class handling
    ──────────────────────
    If class c is absent (n_c = 0):
      - α_c would be n_total / 0 = ∞ — undefined and numerically explosive.
      - Correct behaviour: set α_c = 0.0 for absent classes.
      - Justification: no samples of class c exist in this batch, so the
        weight for class c is never indexed by any sample's loss term.
        A zero weight is mathematically harmless and prevents NaN propagation.
      - NOTE: This differs from adding eps to n_c (which would give a very
        large but finite weight to missing classes). Setting to 0 is the
        correct choice because it preserves sum-to-one property for
        present classes while avoiding phantom gradient inflation.

    Degenerate case (all classes missing / empty batch):
      Falls back to uniform weights 1/num_classes. This should never occur
      in normal training but is included for robustness in edge case testing.

    Args:
        targets     : Integer class labels, shape (N,), dtype long.
                      Values must be in {0, 1, ..., num_classes-1}.
        num_classes : C. Blueprint specifies 2 (§2.1). Must be ≥ 2.
        eps         : Numerical floor for the zero-sum guard (not for α_c
                      itself — we explicitly zero missing classes, not clamp).
        device      : Output tensor device. Defaults to targets.device.

    Returns:
        alpha_hat : shape (num_classes,), dtype float32.
                    α̂_c^(i) ∈ [0, 1] for all c.
                    Σ_c α̂_c = 1.0 (when ≥ 1 class is present).

    Raises:
        ValueError : targets not 1-D, or values outside [0, num_classes-1].

    Example (blueprint §3.4.3 verification):
        >>> targets = torch.tensor([0]*980 + [1]*20)   # 98% benign, 2% malignant
        >>> w = compute_class_weights(targets)
        >>> print(w)                                   # [0.0204, 0.9796] ≈ [0.02, 0.98]
        >>> print(w[1] / w[0])                         # ≈ 48.02 ≈ 49×
    """
    if targets.dim() != 1:
        raise ValueError(
            f"targets must be 1-D (N,), got shape {tuple(targets.shape)}"
        )

    device = device if device is not None else targets.device
    N: int = int(targets.shape[0])

    # ── Edge case: empty batch ───────────────────────────────────────────────
    # No samples → no gradient information → uniform weights
    if N == 0:
        return torch.full(
            (num_classes,),
            fill_value=1.0 / num_classes,
            dtype=torch.float32,
            device=device,
        )

    # ── Value-range check ────────────────────────────────────────────────────
    t_min, t_max = int(targets.min().item()), int(targets.max().item())
    if t_min < 0 or t_max >= num_classes:
        raise ValueError(
            f"targets values must be in [0, {num_classes - 1}]; "
            f"got min={t_min}, max={t_max}"
        )

    # ── Step 1: count samples per class ─────────────────────────────────────
    # n_c shape: (num_classes,), dtype float32
    # bincount counts occurrences of each value in 0..targets.max()
    # minlength=num_classes ensures we always get exactly C counts,
    # even if the highest class index doesn't appear in this batch.
    n_c: Tensor = (
        targets.bincount(minlength=num_classes)
        .float()
        .to(device)
    )
    # n_total = N = Σ n_c (scalar float, not a tensor, for clarity)
    n_total: float = float(N)

    # ── Step 2: Equation §3.4.3a — inverse-frequency weights ─────────────────
    # α_c = n_total / (2 · n_c)
    #
    # We need n_c > 0 for division. Strategy:
    #   1. Build a boolean mask of present classes.
    #   2. Replace 0-counts with 1.0 temporarily (division placeholder).
    #   3. Compute α_c for all classes.
    #   4. Zero out α_c for absent classes (overwrite placeholder result).
    present_mask: Tensor = n_c > 0                       # shape (num_classes,)

    n_c_safe: Tensor = n_c.clone()
    n_c_safe[~present_mask] = 1.0                        # avoids ÷0; overwritten below

    # §3.4.3a: α_c = n_total / (2 · n_c)
    # The literal "2" in the blueprint is num_classes for binary classification.
    # We use float(num_classes) to support future multi-class extensions
    # while remaining exactly equal to "2" for the binary case the blueprint specifies.
    alpha: Tensor = n_total / (float(num_classes) * n_c_safe)  # shape (C,)

    # Zero out absent classes (see docstring on missing-class handling)
    alpha[~present_mask] = 0.0

    # ── Step 3: Equation §3.4.3b — normalisation ─────────────────────────────
    # α̂_c = α_c / (α_0 + α_1) = α_c / Σ_c α_c
    # The blueprint writes (α_0 + α_1) for the binary case; the general form
    # is Σ_c α_c, which reduces to exactly (α_0 + α_1) when num_classes=2.
    alpha_sum: Tensor = alpha.sum()

    if alpha_sum.item() < eps:
        # Degenerate: no present classes (shouldn't happen if N>0 but guard anyway)
        alpha_hat: Tensor = torch.full(
            (num_classes,),
            fill_value=1.0 / num_classes,
            dtype=torch.float32,
            device=device,
        )
    else:
        # §3.4.3b: α̂_c = α_c / Σ_c α_c
        alpha_hat = alpha / alpha_sum                    # shape (C,), sums to 1.0

    return alpha_hat


# ─────────────────────────────────────────────────────────────────────────────
# ClassWeightedFocalLoss
# Implements §3.4.2 + §3.4.3c (boxed equation)
# ─────────────────────────────────────────────────────────────────────────────

class ClassWeightedFocalLoss(nn.Module):
    """
    Class-Weighted Focal Loss — FedMelanoma v4, Section 3.4.

    Implements the exact boxed formula from §3.4.3c:

        L_CW-Focal  =  −(1/N) · Σ_{j=1}^{N}
                            α̂_{y_j}^(i)              [§3.4.3b — local class weight]
                          · (1 − p_t^(j))^{γ_f}      [§3.4.2  — focal modulation]
                          · log(p_t^(j))              [§3.4.2  — cross-entropy core]

    Three multiplicative factors per sample j
    ──────────────────────────────────────────
    1. α̂_{y_j}^(i) ∈ [0, 1]
       Class weight for sample j's true class. Computed via §3.4.3a–b.
       Rare class (malignant) receives ~49× more weight than common class (benign)
       under the 17:1 imbalance in this dataset.
       → Corrects FAILURE MODE 1: majority-class domination.

    2. (1 − p_t^(j))^{γ_f}  ∈ [0, 1]
       Focal modulation factor. Inversely proportional to model confidence.
       γ_f=2: p_t=0.9 → 0.01 (100× downweight); p_t=0.1 → 0.81 (near-full weight).
       → Corrects FAILURE MODE 2: easy-example saturation.

    3. log(p_t^(j))  ∈ (−∞, 0]
       Cross-entropy core. The standard maximum-likelihood loss term.
       Negative sign makes the overall loss non-negative.

    Numerical implementation strategy
    ──────────────────────────────────
    Naive:    p_t = softmax(logits)[y_j]; loss = −(1−p_t)^γ · log(p_t)
    Problem:  softmax can overflow for large logits; log(0) = −∞; 0·(−∞) = NaN

    Stable:   log_pt = log_softmax(logits)[y_j]   ← log-sum-exp trick
              pt = exp(clamp(log_pt, min=log(eps)))
              focal_weight = (1 − pt)^γ
              loss = −focal_weight · log_pt         ← no log() call here

    The key insight: we extract log(p_t) from log_softmax (one numerically
    stable operation) and use it directly. We compute p_t = exp(log_pt) only
    for the focal weight — and clamp log_pt before exp() to keep p_t ∈ [eps, 1].

    Shape contract
    ──────────────
    Input  logits  : (N, C) — raw unnormalised scores. Do NOT pre-apply softmax.
    Input  targets : (N,)   — integer labels in {0, ..., C−1}. dtype=torch.long.
    Input  weights : (C,)   — optional pre-computed α̂_c. If None, auto-computed.
    Output          : ()    scalar if reduction ∈ {'mean', 'sum'}
                      (N,)  per-sample losses if reduction == 'none'

    Args:
        gamma_f      : Focal exponent γ_f ≥ 0. Blueprint canonical: 2.0 (§3.4.2).
                       γ_f=0 → pure class-weighted cross-entropy (no focal term).
                       γ_f=1 → linear down-weighting.
                       γ_f=2 → quadratic (blueprint default, Lin et al. 2017 [6]).
                       γ_f=5 → very aggressive focus on hardest examples.
        num_classes  : C. Blueprint specifies 2 (benign=0, malignant=1, §2.1).
        reduction    : How to aggregate per-sample losses over the batch:
                       'mean' → (1/N)Σ l_j        [blueprint default, §3.4.3c]
                       'sum'  → Σ l_j              [gradient accumulation]
                       'none' → raw (N,) tensor    [per-sample diagnostics]
        eps          : Numerical floor. log_pt clamped at log(eps). Default 1e-8.
        auto_weight  : If True (default), compute α̂_c from batch targets when
                       class_weights is not passed to forward(). Recommended usage:
                       compute weights once per epoch from the full partition
                       (more stable than per-batch computation) and pass explicitly.

    Example:
        >>> loss_fn = ClassWeightedFocalLoss(gamma_f=2.0, reduction='mean')
        >>> logits  = torch.randn(32, 2)
        >>> targets = torch.randint(0, 2, (32,))
        >>> loss    = loss_fn(logits, targets)   # scalar tensor
        >>> loss.backward()                      # gradient flows
    """

    def __init__(
        self,
        gamma_f:     float = GAMMA_F_DEFAULT,
        num_classes: int   = NUM_CLASSES_DEFAULT,
        reduction:   Literal["mean", "sum", "none"] = "mean",
        eps:         float = EPS_DEFAULT,
        auto_weight: bool  = True,
    ) -> None:
        super().__init__()

        # ── Parameter validation ─────────────────────────────────────────────
        if gamma_f < 0.0:
            raise ValueError(f"gamma_f must be ≥ 0.0, got {gamma_f!r}")
        if num_classes < 2:
            raise ValueError(f"num_classes must be ≥ 2, got {num_classes!r}")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(
                f"reduction must be 'mean', 'sum', or 'none'; got {reduction!r}"
            )
        if not (0.0 < eps < 1.0):
            raise ValueError(f"eps must be in (0, 1), got {eps!r}")

        # Store as plain Python scalars — no nn.Parameter; these are not learned.
        # They are hyperparameters fixed by the blueprint (γ_f = 2, §3.4.2).
        self.gamma_f:     float = float(gamma_f)
        self.num_classes: int   = int(num_classes)
        self.reduction:   str   = reduction
        self.eps:         float = float(eps)
        self.auto_weight: bool  = bool(auto_weight)

        # Precompute log(eps) once — used as the clamping floor for log_pt.
        # Storing avoids repeated math.log() calls in the hot forward path.
        self._log_eps: float = math.log(self.eps)   # e.g. log(1e-8) ≈ −18.42

    # ──────────────────────────────────────────────────────────────────────────
    # forward — implements §3.4.3c boxed equation
    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        logits:        Tensor,
        targets:       Tensor,
        class_weights: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute the Class-Weighted Focal Loss for one mini-batch.

        Maps directly to the boxed equation in §3.4.3c. Each numbered comment
        below corresponds to one multiplicative factor in the formula.

        Args:
            logits        : Raw unnormalised scores from the fusion head (§3.3.3).
                            Shape (N, C), dtype float32 or float16.
                            MUST be pre-softmax logits — NOT probabilities.
            targets       : Ground-truth class indices, shape (N,), dtype torch.long.
                            Values in {0, 1, ..., C−1}.
                            0 = benign, 1 = malignant (blueprint §2.1 schema).
            class_weights : Pre-computed α̂_c^(i) from compute_class_weights().
                            Shape (C,), dtype float32.
                            Recommended: compute once per epoch from the full local
                            partition, not per mini-batch, for stable weighting.
                            If None and auto_weight=True: computed from this batch.
                            If None and auto_weight=False: raises RuntimeError.

        Returns:
            If reduction == 'mean' or 'sum':
                Scalar tensor (shape ()). Supports .backward().
            If reduction == 'none':
                Per-sample loss tensor, shape (N,). Supports .backward().

        Raises:
            ValueError    : Shape or value range violation.
            RuntimeError  : class_weights is None with auto_weight=False.
        """
        # ── Validate shapes and value ranges ────────────────────────────────
        self._validate_inputs(logits, targets, class_weights)

        N: int = int(logits.shape[0])

        # ── Edge case: empty batch ──────────────────────────────────────────
        # Return a zero scalar (connected to graph so .backward() is a no-op,
        # not an error). Shape depends on reduction mode.
        if N == 0:
            if self.reduction == "none":
                return logits.new_zeros(0)
            # sum(empty) * 0 preserves device/dtype and creates a grad_fn node
            return logits.sum() * 0.0

        # ── Resolve class weights  α̂_c^(i)  [§3.4.3b] ──────────────────────
        # Privacy invariant: compute_class_weights() runs fully on-device.
        # Its outputs are local tensors only — never serialised or transmitted.
        if class_weights is None:
            if not self.auto_weight:
                raise RuntimeError(
                    "class_weights is None but auto_weight=False. "
                    "Either pass class_weights=compute_class_weights(targets) "
                    "or set auto_weight=True."
                )
            # Auto-compute from current batch targets.
            # For more stable weights: pre-compute once per epoch from the
            # full partition and pass explicitly to each forward() call.
            weights: Tensor = compute_class_weights(
                targets,
                num_classes=self.num_classes,
                eps=self.eps,
                device=logits.device,
            )
        else:
            # Cast to float32 on the correct device.
            # The user may pass float64 or half-precision weights.
            weights = class_weights.to(dtype=torch.float32, device=logits.device)

        # ── STEP 1: log-softmax over logits  →  log(p_c) for all classes ──
        #
        # log_p[j, c]  =  log( exp(logits[j,c]) / Σ_k exp(logits[j,k]) )
        #              =  logits[j,c] − log( Σ_k exp(logits[j,k]) )
        #
        # Implemented via the log-sum-exp trick for numerical stability:
        #   log_softmax(z)_c = z_c − max(z) − log( Σ_k exp(z_k − max(z)) )
        # This prevents overflow (logits shifted to ≤0 before exp).
        #
        # We cast logits to float32 for the loss computation even if the model
        # runs in mixed precision — loss should always be full precision.
        #
        # Blueprint §3.3.3 connection: "softmax transformation yields ... p̂_c"
        # We use log_softmax here; the exp is deferred to Step 3.
        log_p: Tensor = F.log_softmax(logits.float(), dim=-1)   # (N, C)

        # ── STEP 2: gather log(p_t^(j)) — true-class log-probability ────────
        #
        # For each sample j:  log_pt[j] = log_p[j, targets[j]]
        #
        # Blueprint §3.4.2: "p_t = p̂_{y_j}" — probability for the true class y_j.
        # We select the corresponding log-probability for each sample.
        #
        # targets.unsqueeze(1) : (N,) → (N, 1)  — required index shape for gather
        # .gather(1, ...)       : select along class dimension (dim=1)
        # .squeeze(1)           : (N, 1) → (N,)  — back to per-sample scalar
        log_pt: Tensor = log_p.gather(
            dim=1,
            index=targets.unsqueeze(1),     # (N, 1) integer indices
        ).squeeze(1)                         # (N,)  — log(p_t^(j)) per sample

        # Numerical clamp: ensure log_pt ∈ [log(eps), 0]
        # Upper bound 0: log_softmax ≤ 0 always, clamp enforces this.
        # Lower bound log(eps): prevents exp(log_pt) → 0 causing (1-0)^γ·(−∞)=NaN.
        log_pt = log_pt.clamp(min=self._log_eps, max=0.0)   # (N,)

        # ── STEP 3: true-class probability p_t^(j)  ─────────────────────────
        #
        # p_t[j]  =  exp( log_pt[j] )  ∈  [eps, 1.0]
        #
        # Blueprint §3.4.2: "Let p_t = p̂_{y_j}"
        # We compute p_t ONLY for the focal weight (1−p_t)^γ, not for log(p_t)
        # (which we already have as log_pt). This avoids a redundant log() call.
        pt: Tensor = log_pt.exp()                            # (N,), ∈ [eps, 1.0]

        # ── STEP 4: focal modulation factor (1 − p_t^(j))^{γ_f}  ───────────
        #
        # focal_weight[j]  =  (1 − p_t[j])^{γ_f}  ∈  [0, (1−eps)^γ_f]
        #
        # Blueprint §3.4.2 numerical verification:
        #   Easy example:  p_t=0.90 → (0.10)^2 = 0.0100  (100× downweight)
        #   Medium:        p_t=0.50 → (0.50)^2 = 0.2500
        #   Hard example:  p_t=0.10 → (0.90)^2 = 0.8100  (near-full weight)
        #
        # Special case γ_f=0: (1−p_t)^0 = 1.0 → pure class-weighted CE (no focal).
        # torch.Tensor.pow() handles γ_f=0 correctly: x^0 = 1 for x ≥ 0.
        focal_weight: Tensor = (1.0 - pt).pow(self.gamma_f)  # (N,)

        # ── STEP 5: per-sample class weight  α̂_{y_j}^(i)  ──────────────────
        #
        # per_sample_alpha[j]  =  weights[ targets[j] ]  =  α̂_{y_j}^(i)
        #
        # Blueprint §3.4.3b: "α̂_{y_j}^(i)" indexes the normalised weight
        # tensor by each sample's true class label.
        # weights shape: (C,); targets shape: (N,) → output shape: (N,)
        per_sample_alpha: Tensor = weights[targets]           # (N,)

        # ── STEP 6: per-sample loss l_j  ─────────────────────────────────────
        #
        # l_j  =  − α̂_{y_j}^(i)  ·  (1 − p_t^(j))^{γ_f}  ·  log(p_t^(j))
        #
        # This is the un-reduced form of §3.4.3c (before the (1/N)Σ).
        # All three tensors are (N,), pointwise multiplication.
        # The negative sign makes l_j ≥ 0 since log_pt ≤ 0.
        loss_per_sample: Tensor = -(
            per_sample_alpha     # α̂_{y_j}   ∈ [0, 1]
            * focal_weight       # (1−p_t)^γ  ∈ [0, 1]
            * log_pt             # log(p_t)   ∈ [log(eps), 0]
        )                                                      # (N,), ≥ 0

        # ── STEP 7: reduction  →  scalar loss (§3.4.3c: "−1/N · Σ") ────────
        #
        # 'mean': (1/N)Σ l_j  — implements the "1/N" in the boxed formula.
        # 'sum':  Σ l_j        — for gradient accumulation across micro-batches.
        # 'none': l_j          — per-sample losses for diagnostics or custom agg.
        if self.reduction == "mean":
            return loss_per_sample.mean()    # scalar: (1/N)Σ l_j
        elif self.reduction == "sum":
            return loss_per_sample.sum()     # scalar: Σ l_j
        else:                                # 'none'
            return loss_per_sample           # (N,)

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _validate_inputs(
        self,
        logits:        Tensor,
        targets:       Tensor,
        class_weights: Optional[Tensor],
    ) -> None:
        """
        Validate tensor shapes, dtypes, and value ranges.

        Called at the top of forward() before any computation.
        Raises ValueError on any violation with a descriptive message.
        """
        # logits: must be exactly 2-D with shape (N, num_classes)
        if logits.dim() != 2:
            raise ValueError(
                f"logits must be 2-D (N, C), got shape {tuple(logits.shape)}"
            )
        N, C = logits.shape[0], logits.shape[1]
        if C != self.num_classes:
            raise ValueError(
                f"logits.shape[1] = {C} must equal num_classes = {self.num_classes}. "
                f"Got logits of shape {tuple(logits.shape)}."
            )

        # targets: must be 1-D with length N and long dtype
        if targets.dim() != 1:
            raise ValueError(
                f"targets must be 1-D (N,), got shape {tuple(targets.shape)}"
            )
        if targets.shape[0] != N:
            raise ValueError(
                f"Batch size mismatch: logits has N={N}, targets has N={targets.shape[0]}."
            )
        if N > 0:
            t_min = int(targets.min().item())
            t_max = int(targets.max().item())
            if t_min < 0 or t_max >= self.num_classes:
                raise ValueError(
                    f"targets values must be in [0, {self.num_classes - 1}]; "
                    f"got range [{t_min}, {t_max}]. "
                    f"Blueprint §2.1: 0=benign, 1=malignant."
                )

        # class_weights: must be 1-D (C,) if provided
        if class_weights is not None:
            if class_weights.dim() != 1:
                raise ValueError(
                    f"class_weights must be 1-D (C,), "
                    f"got shape {tuple(class_weights.shape)}"
                )
            if class_weights.shape[0] != self.num_classes:
                raise ValueError(
                    f"class_weights.shape[0] = {class_weights.shape[0]} "
                    f"must equal num_classes = {self.num_classes}."
                )
            if torch.any(class_weights < 0).item():
                raise ValueError(
                    "class_weights must be non-negative; got at least one negative value."
                )

    def extra_repr(self) -> str:
        """
        Displayed by print(model) / repr(loss_fn).
        Shows all hyperparameters and their blueprint source.
        """
        return (
            f"gamma_f={self.gamma_f} [§3.4.2], "
            f"num_classes={self.num_classes}, "
            f"reduction='{self.reduction}', "
            f"eps={self.eps}, "
            f"auto_weight={self.auto_weight}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone verification — blueprint §3.4.3 example check
# Run: python focal_loss.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("  FedMelanoma v4 — Section 3.4 Class-Weighted Focal Loss")
    print("  Blueprint example verification")
    print("=" * 70)

    torch.manual_seed(0)

    # ── Blueprint §3.4.3 numerical example ───────────────────────────────────
    # "At 2% local malignancy (n_0=980, n_1=20): α̂_1 ≈ 0.98, amplifying
    #  malignant gradient by ≈ 49×."
    print("\n[1] Blueprint §3.4.3 weight verification (n_0=980, n_1=20)")
    targets_demo = torch.tensor([0] * 980 + [1] * 20)
    w = compute_class_weights(targets_demo)
    print(f"    α̂_0 (benign)    = {w[0].item():.6f}")
    print(f"    α̂_1 (malignant) = {w[1].item():.6f}")
    print(f"    α̂_0 + α̂_1       = {w.sum().item():.6f}  [must be 1.0]")
    print(f"    α̂_1 / α̂_0       = {(w[1]/w[0]).item():.2f}×  [blueprint: ≈49×]")

    expected_ratio = (1000 / (2 * 20)) / (1000 / (2 * 980))
    raw_sum        = (1000 / (2 * 20)) + (1000 / (2 * 980))
    expected_w1    = (1000 / (2 * 20)) / raw_sum
    print(f"    Expected α̂_1   = {expected_w1:.6f}  [manual calculation]")
    assert abs(w[1].item() - expected_w1) < 1e-5, "Weight mismatch!"
    print("    ✓ Weight values match blueprint §3.4.3 exactly.")

    # ── Focal modulation verification §3.4.2 ─────────────────────────────────
    print("\n[2] Focal modulation (1−p_t)^γ_f at γ_f=2 [§3.4.2]")
    for p_t, label in [(0.9, "easy"), (0.5, "medium"), (0.1, "hard")]:
        fm = (1 - p_t) ** 2
        print(f"    p_t={p_t:.1f} ({label:6s}): (1-p_t)^2 = {fm:.4f}"
              + (" [100× downweight — §3.4.2]" if p_t == 0.9 else ""))

    # ── Forward pass with canonical parameters ────────────────────────────────
    print("\n[3] Forward pass — ClassWeightedFocalLoss(gamma_f=2, reduction='mean')")
    loss_fn = ClassWeightedFocalLoss(gamma_f=2.0, reduction="mean")
    print(f"    {loss_fn}")

    N_batch = 64
    logits  = torch.randn(N_batch, 2, requires_grad=True)
    targets = torch.randint(0, 2, (N_batch,))
    loss    = loss_fn(logits, targets)
    print(f"    Input:  logits {tuple(logits.shape)}, targets {tuple(targets.shape)}")
    print(f"    Output: loss = {loss.item():.6f}  (scalar)")
    print(f"    Loss ≥ 0: {loss.item() >= 0}")

    loss.backward()
    print(f"    Backward: OK (grad_fn={loss.grad_fn.__class__.__name__})")

    # ── γ_f=0 equivalence with class-weighted cross-entropy ──────────────────
    print("\n[4] γ_f=0 equivalence: should match class-weighted cross-entropy")
    torch.manual_seed(42)
    logits2  = torch.randn(32, 2)
    targets2 = torch.randint(0, 2, (32,))
    weights2 = compute_class_weights(targets2)

    loss_focal0 = ClassWeightedFocalLoss(gamma_f=0.0, reduction="mean")(
        logits2, targets2, class_weights=weights2
    )
    # Manual class-weighted CE: −Σ α̂_{y_j} log p_{y_j} / N
    log_p2  = F.log_softmax(logits2, dim=-1)
    log_pt2 = log_p2.gather(1, targets2.unsqueeze(1)).squeeze(1)
    manual_ce = -(weights2[targets2] * log_pt2).mean()

    print(f"    Focal(γ=0)  loss = {loss_focal0.item():.8f}")
    print(f"    Manual CWCE loss = {manual_ce.item():.8f}")
    print(f"    Difference       = {abs(loss_focal0.item() - manual_ce.item()):.2e}")
    assert abs(loss_focal0.item() - manual_ce.item()) < 1e-5, "γ_f=0 equivalence failed!"
    print("    ✓ γ_f=0 matches class-weighted cross-entropy exactly.")

    print("\n  All blueprint verification checks passed. ✓")
    sys.exit(0)
