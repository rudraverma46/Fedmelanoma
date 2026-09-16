# FedMelanoma: Canonical Architecture Blueprint
## Version 4.0 — Final Authoritative Specification

**A Privacy-Preserving Federated Learning System for Melanoma Detection  
with Exposure-Aware Utility Attenuation and Boltzmann Client Scheduling**

> *"The measure of a learning system's integrity is not its performance on the average patient — it is its performance on the patient most easily forgotten."*

---

## Abstract

FedMelanoma is a federated learning system for melanoma detection that jointly addresses multi-axis Non-IID data heterogeneity and asynchronous straggler dynamics without centralizing patient data. This v4 blueprint establishes the canonical mathematical architecture, replacing all prior additive selection heuristics with a theoretically grounded, multiplicatively structured selector. The central contribution is a **exposure-aware marginal utility framework** that models diminishing optimization returns from repeated client interactions via a staleness-discounted effective interaction count $N_i^{\text{eff}}(t)$. Client selection uses **Boltzmann (softmax) probabilistic sampling** over a composite score that multiplicatively gates reliability against bounded diversity utility, ensuring exploration of minority-phenotype clients without starvation. All diversity priors are computed via cost-matrix Optimal Transport over three metadata axes derived directly from the preprocessed dataset: Fitzpatrick skin type (1D Wasserstein), imaging modality (LP-based OT with semantic cost matrix), and diagnosis confirmation type (LP-based OT with epistemic confidence cost matrix). Post-hoc temperature scaling provides calibrated clinical confidence estimates. The blueprint is fully aligned with `preprocessed_data.csv` (9,740 records, 5.55% malignancy prevalence).

**Canonical hyperparameters:** $\rho = 0.85$, $\delta = 0.90$, $\lambda = 0.05$, $\gamma = 1.0$, $\tau = 0.20$.

---

## Table of Contents

1. [Product Requirements & System Overview](#1-prd)
2. [Dataset Specification & Statistical Grounding](#2-dataset)
3. [Mathematical Foundations](#3-math)
   - 3.1 Non-IID Partitioning: Softmax-Temperature Assignment
   - 3.2 Diversity Prior: Cost-Matrix Optimal Transport
   - 3.3 Model Architecture: EfficientNet-B3 + Metadata MLP
   - 3.4 Class-Weighted Focal Loss
   - 3.5 Reliability Signal $R_i^{(t)}$
   - 3.6 Effective Interaction Count $N_i^{\text{eff}}(t)$
   - 3.7 Marginal Utility Attenuation $U_i^{\text{eff}}(t)$
   - 3.8 Canonical Selector Score
   - 3.9 Boltzmann Client Scheduling
   - 3.10 Staleness-Weighted Aggregation with Diversity Bonus
   - 3.11 Logit Calibration & Clinical Reliability
   - 3.12 Theoretical Convergence Analysis
4. [End-to-End Implementation Strategy](#4-implementation)
5. [Evaluation & Validation Framework](#5-eval)
6. [Comparison with Existing FL Methods](#6-comparison)
7. [Ablation Studies](#7-ablation)
8. [Limitations](#8-limitations)
9. [Future Work](#9-future)
10. [System Configuration Schema](#10-config)
11. [Repository Architecture](#11-repo)
12. [References](#12-refs)

---

## 1. Product Requirements & System Overview

### 1.1 Mission Statement

FedMelanoma is a federated learning system for melanoma classification that operates under two simultaneous adversarial conditions endemic to real-world healthcare AI deployments:

**(A) Multi-axis Non-IID heterogeneity.** Hospital nodes differ systematically in imaging modality, patient skin phenotype, and diagnostic confirmation methodology. These differences are not random noise — they are structurally correlated with facility type, geography, and resource availability, creating precisely the data distributions that standard FL algorithms fail to represent.

**(B) Asynchronous straggler dynamics.** Resource-constrained clients — rural clinics, telemedicine nodes, primary care facilities — are both the most likely to submit late or failed updates and the most likely to carry underrepresented patient phenotypes. A system that ignores this structural correlation actively produces globally biased models under the guise of principled averaging.

The canonical contribution of v4 is a client scheduling framework that is **not an additive heuristic**. It is a multiplicatively structured, theoretically grounded utility estimator that correctly models why repeated interaction with the same client eventually contributes less to global optimization progress, why minority-distribution clients should be selected more aggressively early in training and less aggressively once their gradient manifold is well-covered, and why stochastic selection via Boltzmann sampling is strictly superior to greedy top-$K$ for heterogeneous asynchronous FL.

### 1.2 Regulatory & Privacy Requirements

| Requirement | Constraint | Mechanism |
|---|---|---|
| HIPAA § 164.514 | No raw PHI leaves client premises | Transmit weight deltas $\Delta\mathbf{w}_i$ only |
| GDPR Article 25 | Privacy-by-design, data minimisation | gRPC/mTLS + per-client X.509 certificates |
| Clinical auditability | Model versioning, participation records | Write-once SHA-256 audit log per round |
| Equity (SDG 10) | No skin-type performance blind spots | OT-diversity-gated Boltzmann scheduling |

### 1.3 Three Non-IID Axes

Let $\mathcal{H} = \{h_1, \ldots, h_K\}$ be $K = 10$ hospital clients. Each holds $\mathcal{D}_i = \{(\mathbf{x}_j, y_j, \mathbf{m}_j)\}_{j=1}^{n_i}$. The metadata vector is exactly:

$$\mathbf{m}_i = [\varphi_i,\ \tau_i^{\text{mod}},\ \lambda_i]^T$$

where $\varphi_i$ = Fitzpatrick skin type, $\tau_i^{\text{mod}}$ = imaging modality, $\lambda_i$ = confirmation methodology. **Anatomic site is not present in this dataset and is permanently removed from all formulations.**

**Feature skew** $P(\mathbf{x})$: Dermoscopy (84.45%) vs. clinical close-up (14.37%) vs. clinical overview (1.18%) produce fundamentally distinct pixel statistics, augmentation effects, and pre-trained feature alignments.

**Label quantity skew** $P(y)$: Global malignancy prevalence is 5.55%. Under Non-IID partitioning, per-client rates range from ~0–2% (telemedicine, single-contributor confirmation) to ~30–35% (tertiary oncology, histopathology-confirmed). This 15–17$\times$ spread induces severe gradient disagreement.

**Concept shift** $P(y | \mathbf{x})$: Histopathology-confirmed samples carry 37.38% malignancy; single-contributor assessment carries 0.00%. The same dermoscopic image pattern can receive opposite labels at two institutions, producing irreconcilable label semantic inconsistency that no loss function can resolve without multi-source cross-entropy reweighting.

### 1.4 Functional Requirements

| ID | Specification |
|---|---|
| FR-1 | Characterise and mitigate all three Non-IID axes simultaneously |
| FR-2 | Tolerate 40–60% per-round straggler rates without blocking the training loop |
| FR-3 | Select clients via Boltzmann sampling over the canonical exposure-aware utility score |
| FR-4 | All clients share parameter-compatible architecture: EfficientNet-B3 + Metadata MLP (13-d input) |
| FR-5 | gRPC/mTLS communication with zstd compression |
| FR-6 | Post-hoc temperature calibration at every evaluation checkpoint |
| FR-7 | Grad-CAM clinical inference interface |

### 1.5 Architecture Flow

```
preprocessed_data.csv  (9,740 records)
  isic_id | label | skin_type | image_type | confirm_type
                │
                ▼
    Non-IID Partitioner
    P(k | m_i) = softmax(W·f(m_i) / τ_soft),   W ∈ ℝ^{K×3},  τ_soft = 0.5
                │
    ┌───────────┴──────────────────┐
    K=10 heterogeneous partitions
                │
    ▼ (parallel, isolated)
    ┌──────────────────────────────────────────┐
    │  Per-Client Local Training               │
    │  Image: resize 224×224 → augment         │
    │  Metadata: one-hot(6+3+4=13 dims)        │
    │  EfficientNet-B3 → 1536-d                │
    │  MetadataMLP(13→64→32)                   │
    │  Fusion → 1568-d → Dropout → logit       │
    │  Class-Weighted Focal Loss (γ_f=2)       │
    │  E=5 local epochs, AdamW                 │
    └──────────────────────────────────────────┘
                │
    Δwᵢ = wᵢ^{t+1} − w^t  (zstd compressed, mTLS)
                │
    ┌──────────────────────────────────────────┐
    │  AGGREGATION SERVER                      │
    │                                          │
    │  Selection: Boltzmann sampling over      │
    │    score_i = R_i·(1 + γ·D_i/(1+λN_i))  │
    │                                          │
    │  Straggler: W_t = T̃_med + k·MAD         │
    │                                          │
    │  Aggregation:                            │
    │    c_i = (n_i/n_A)·δ^{s_i}·(1+γD_i)/Z  │
    │    w^{t+1} = w^t + Σ c_i·Δwᵢ            │
    │                                          │
    │  Calibration: T* via NLL minimization    │
    └──────────────────────────────────────────┘
                │
    Global Model (checkpointed) + T*
                │
    Flask / Grad-CAM Clinical Interface
```

---

## 2. Dataset Specification & Statistical Grounding

### 2.1 Schema

The preprocessed dataset contains **9,740 samples** with **no missing values** across five columns:

| Column | Type | Values |
|---|---|---|
| `isic_id` | string | ISIC unique identifier |
| `label` | int {0,1} | 0 = Benign, 1 = Malignant |
| `skin_type` | string | I, II, III, IV, V, VI |
| `image_type` | string | dermoscopic; clinical: close-up; clinical: overview |
| `confirm_type` | string | histopathology; serial imaging showing no change; single image expert consensus; single contributor clinical assessment |

### 2.2 Label Distribution

$$n = 9{,}740, \quad n_{\text{malignant}} = 541 \ (5.55\%), \quad n_{\text{benign}} = 9{,}199 \ (94.45\%), \quad \text{ratio} \approx 17:1$$

The 17:1 imbalance makes accuracy a degenerate metric and motivates the class-weighted focal loss.

### 2.3 Feature Distributions

**Fitzpatrick Skin Type:**

| Type | $n$ | Global $P$ | Malignancy Rate | Notes |
|---|---|---|---|---|
| I | 2,599 | 26.68% | 2.85% | Most common, near-global mean |
| II | 3,549 | 36.44% | 11.44% | Dominant class |
| III | 1,182 | 12.14% | 4.57% | |
| IV | 844 | 8.67% | 0.83% | |
| V | 815 | 8.37% | **0.00%** | Critical dataset artifact |
| VI | 751 | 7.71% | **0.00%** | Critical dataset artifact |

The 0.00% malignancy for Types V–VI reflects underrepresentation of dark-skin melanoma in the ISIC archive, not a biological truth. This makes high-$D_i$ clients carrying these phenotypes the most important to preserve in the global model.

**Imaging Modality:**

| Modality | $n$ | Global $P$ | Malignancy Rate |
|---|---|---|---|
| Dermoscopic | 8,225 | 84.45% | 5.08% |
| Clinical: close-up | 1,400 | 14.37% | 4.29% |
| Clinical: overview | 115 | 1.18% | **54.78%** |

Clinical overview carries a 54.78% malignancy rate — 9.9$\times$ the global mean. Clients holding this modality carry extreme predictive value.

**Confirmation Type:**

| Confirmation | $n$ | Global $P$ | Malignancy Rate |
|---|---|---|---|
| Single contributor clinical assessment | 4,879 | 50.09% | 0.00% |
| Single image expert consensus | 1,752 | 17.99% | 0.29% |
| Serial imaging showing no change | 1,675 | 17.20% | 0.00% |
| Histopathology | 1,434 | 14.72% | **37.38%** |

### 2.4 Metadata Feature Vector

The three-axis metadata vector is:

$$\mathbf{m}_i = [\varphi_i,\ \tau_i^{\text{mod}},\ \lambda_i]^T$$

One-hot encoded to a **13-dimensional** input for the Metadata MLP:

$$f(\mathbf{m}_i) \in \mathbb{R}^{13} = \underbrace{\mathbb{R}^6}_{\text{Fitzpatrick}} \oplus \underbrace{\mathbb{R}^3}_{\text{modality}} \oplus \underbrace{\mathbb{R}^4}_{\text{confirmation}}$$

---

## 3. Mathematical Foundations

### 3.1 Non-IID Partitioning: Softmax-Temperature Assignment

#### 3.1.1 Facility Profile Matrix

To simulate $K = 10$ clinically realistic, heterogeneous distributions from the preprocessed dataset, define the **facility profile matrix** $\mathbf{W} \in \mathbb{R}^{K \times 3}$, where row $\mathbf{w}_k = [w_k^\varphi,\ w_k^\tau,\ w_k^\lambda]$ encodes hospital $k$'s affinity across the three metadata axes.

#### 3.1.2 Metadata Affinity Encoding

The affinity encoding function $f: \mathcal{M} \to \mathbb{R}^3$ maps raw metadata to a scalar affinity vector:

$$f(\mathbf{m}_i) = [f_\varphi(\varphi_i),\ f_\tau(\tau_i),\ f_\lambda(\lambda_i)]^T$$

$$f_\varphi(\varphi_i) = \begin{cases} 0.10 & \text{Type I} \\ 0.20 & \text{Type II} \\ 0.40 & \text{Type III} \\ 0.60 & \text{Type IV} \\ 0.80 & \text{Type V} \\ 1.00 & \text{Type VI} \end{cases} \qquad f_\tau(\tau_i) = \begin{cases} 0.95 & \text{dermoscopic} \\ 0.60 & \text{clinical: close-up} \\ 0.30 & \text{clinical: overview} \end{cases} \qquad f_\lambda(\lambda_i) = \begin{cases} 1.00 & \text{histopathology} \\ 0.90 & \text{serial imaging} \\ 0.85 & \text{expert consensus} \\ 0.75 & \text{single contributor} \end{cases}$$

#### 3.1.3 Softmax-Temperature Assignment

$$\boxed{P(k \mid \mathbf{m}_i) = \frac{\exp\!\left(\mathbf{w}_k \cdot f(\mathbf{m}_i) \;/\; \tau_{\text{soft}}\right)}{\displaystyle\sum_{j=1}^{K} \exp\!\left(\mathbf{w}_j \cdot f(\mathbf{m}_i) \;/\; \tau_{\text{soft}}\right)}}$$

At $\tau_{\text{soft}} = 0.5$: strong Non-IID skew. Tertiary oncology nodes receive histopathology-confirmed, dermoscopic, fair-skin samples with high probability; telemedicine nodes receive single-contributor, clinical-overview, dark-skin samples. This produces the 15–17$\times$ malignancy rate spread across clients that the system is designed to handle.

---

### 3.2 Diversity Prior: Cost-Matrix Optimal Transport

#### 3.2.1 Why Optimal Transport Over KL Divergence

KL divergence $D_{\text{KL}}(P \| Q)$ is undefined when $Q(c) = 0$ at categories where $P(c) > 0$ — a condition that regularly occurs when comparing hospital distributions over rare metadata categories (e.g., a client with zero clinical overview samples). Total variation distance is defined but assigns unit cost to every category swap, ignoring semantic structure: it treats the shift from dermoscopic to clinical close-up as equal in cost to the shift from dermoscopic to clinical overview, which is empirically and clinically incorrect.

**Earth Mover's Distance** (Wasserstein-1 / Optimal Transport) [11, 12] respects the geometry of the category space. It asks: what is the minimum work — measured as (mass moved) $\times$ (semantic distance traveled) — to transform distribution $P$ into distribution $Q$? This makes it the metrically correct measure for comparing hospital metadata distributions.

#### 3.2.2 General Optimal Transport Formulation

Let $P = (p_1,\ldots,p_n)$ and $Q = (q_1,\ldots,q_m)$ be discrete probability distributions. A **transport plan** $\boldsymbol{\pi} \in \mathbb{R}_{\geq 0}^{n \times m}$ must satisfy the marginal constraints:

$$\sum_{j=1}^m \pi_{ij} = p_i \quad \forall i, \qquad \sum_{i=1}^n \pi_{ij} = q_j \quad \forall j$$

The feasible set is $\Pi(P, Q) = \{\boldsymbol{\pi} \geq 0 : \boldsymbol{\pi}\mathbf{1} = P,\ \boldsymbol{\pi}^T\mathbf{1} = Q\}$. Given a cost matrix $\mathbf{C} \in \mathbb{R}_{\geq 0}^{n \times m}$ where $C_{ij}$ is the semantic cost of moving unit mass from category $i$ to category $j$:

$$\boxed{\text{OT}(P, Q; \mathbf{C}) = \min_{\boldsymbol{\pi} \in \Pi(P,Q)} \langle \boldsymbol{\pi}, \mathbf{C} \rangle_F = \min_{\boldsymbol{\pi} \in \Pi(P,Q)} \sum_{i,j} \pi_{ij} C_{ij}}$$

This is a linear program solved in $\mathcal{O}(n^3 \log n)$ time. For our small category counts ($n, m \leq 6$), exact LP computation via HiGHS is computationally negligible. For scale, the **Sinkhorn algorithm** [13] provides an entropy-regularized approximation: $\text{OT}_\varepsilon(P, Q; \mathbf{C}) = \min_{\pi} \langle \pi, \mathbf{C} \rangle_F + \varepsilon H(\pi)$, converging in $\mathcal{O}(n^2/\varepsilon^2)$ matrix-vector products.

#### 3.2.3 1D Wasserstein Distance for Fitzpatrick (Ordinal)

For Fitzpatrick skin type, the ordinal structure ($\text{I} < \text{II} < \cdots < \text{VI}$) with unit spacing supports the **1D Wasserstein-1** distance computed via the CDF area formula:

$$W_1(P_\varphi, Q_\varphi) = \sum_{k=1}^{5} |\text{CDF}_{P_\varphi}(k) - \text{CDF}_{Q_\varphi}(k)|$$

This is the area between the two CDFs — equivalently, the Optimal Transport distance on $\mathbb{R}$ with the Euclidean ground metric. It correctly penalizes mass moved across many ordinal steps (e.g., Type I $\to$ Type VI costs 5 units) more than adjacent moves (Type I $\to$ Type II costs 1 unit). This property is essential: a telemedicine node concentrating on Types V–VI is distributionally far from the global mean (Type I–II dominant), and this distance should be large.

#### 3.2.4 Semantic Cost Matrix for Imaging Modality

$$\mathbf{C}_{\text{mod}} = \begin{pmatrix} 0.00 & 0.30 & 0.70 \\ 0.30 & 0.00 & 0.45 \\ 0.70 & 0.45 & 0.00 \end{pmatrix} \quad \begin{array}{l} \text{rows/cols: (dermoscopic,} \\ \quad \text{clinical:close-up,} \\ \quad \text{clinical:overview)} \end{array}$$

**Semantic justification:**

- $C_{12} = C_{21} = 0.30$ (dermoscopic ↔ close-up): Both capture individual lesions under controlled conditions; partial feature transferability.
- $C_{13} = C_{31} = 0.70$ (dermoscopic ↔ overview): Dermoscopy resolves subsurface microstructures; clinical overview provides anatomical context with lesions as small objects. Fundamentally different diagnostic tasks.
- $C_{23} = C_{32} = 0.45$ (close-up ↔ overview): Shared surface-reflectance imaging but different scale and framing.

#### 3.2.5 Epistemic Cost Matrix for Confirmation Type

Ordered by decreasing diagnostic confidence tier, confirmed by empirical malignancy rates (histopathology: 37.38%, serial imaging: 0.00%, expert consensus: 0.29%, single contributor: 0.00%):

$$C_{ij}^{\text{conf}} = \frac{|\text{tier}(i) - \text{tier}(j)|}{3}$$

$$\mathbf{C}_{\text{conf}} = \begin{pmatrix} 0 & 1/3 & 2/3 & 1 \\ 1/3 & 0 & 1/3 & 2/3 \\ 2/3 & 1/3 & 0 & 1/3 \\ 1 & 2/3 & 1/3 & 0 \end{pmatrix} \quad \begin{array}{l} \text{rows/cols: (histopathology,} \\ \quad \text{serial imaging,} \\ \quad \text{expert consensus,} \\ \quad \text{single contributor)} \end{array}$$

The entry $C_{14} = 1.0$ captures the maximum epistemic distance: moving probability mass from histopathology-confirmed labels to single-contributor-assessed labels crosses the full confidence spectrum.

**Numerical verification (Optimal Transport cost, LP solution):**

Consider $P_A = [0.70, 0.15, 0.10, 0.05]$ (histopathology-heavy) vs. $P_B = [0.05, 0.10, 0.15, 0.70]$ (single-contributor-heavy):

$$\text{OT}(P_A, P_B; \mathbf{C}_{\text{conf}}) = \frac{2}{3} \approx 0.6\overline{6}$$

This equals the TV distance times the average cost scaling for this near-antipodal case. The LP-optimal transport plan ships mass along cost-minimizing paths, correctly avoiding the degenerate unit-cost route.

#### 3.2.6 Combined Diversity Score

For client $i$, the **raw diversity** integrates all three OT distances:

$$D_i^{\text{raw}} = w_\varphi \cdot W_1\!\left(\hat{P}_\varphi^{\text{global}}, P_\varphi^{(i)}\right) + w_\tau \cdot \text{OT}\!\left(\hat{P}_\tau^{\text{global}}, P_\tau^{(i)}; \mathbf{C}_{\text{mod}}\right) + w_\lambda \cdot \text{OT}\!\left(\hat{P}_\lambda^{\text{global}}, P_\lambda^{(i)}; \mathbf{C}_{\text{conf}}\right)$$

with default weights $w_\varphi = 0.40$, $w_\tau = 0.35$, $w_\lambda = 0.25$ ($\sum = 1$).

**Normalisation:**

$$\boxed{D_i = \frac{D_i^{\text{raw}}}{\displaystyle\max_{j \in \mathcal{H}} D_j^{\text{raw}}}, \quad D_i \in [0, 1]}$$

**Computed values for representative clients:**

| Client | Type | $W_1^{\text{fitz}}$ | $\text{OT}^{\text{mod}}$ | $\text{OT}^{\text{conf}}$ | $D_i^{\text{raw}}$ | $D_i$ |
|---|---|---|---|---|---|---|
| C1 (Oncology A) | Fair-skin, dermoscopy, histo | 0.8074 | 0.0203 | 0.4615 | 0.4455 | **0.496** |
| C9 (Telemedicine) | Dark-skin, clinical overview, single-contrib | 1.8026 | 0.3286 | 0.2518 | 0.8990 | **1.000** |

C9 holds maximum distributional novelty: its Fitzpatrick distribution is 2.24$\times$ further from the global mean than C1's. This correctly produces $D_9 = 1.000$.

**Critical property:** $D_i$ is a **static prior** — it is a property of the hospital's patient population, not of how many times the server has trained on its data. It does not decay. What decays is the *marginal utility* of training on that manifold, captured by $N_i^{\text{eff}}(t)$ in Section 3.7. This distinction is fundamental.

---

### 3.3 Model Architecture: EfficientNet-B3 + Metadata MLP

#### 3.3.1 EfficientNet-B3 Image Backbone

EfficientNet [5] applies **compound scaling**: simultaneously expanding network width $\phi_w$, depth $\phi_d$, and input resolution $\phi_r$ by a fixed ratio, constrained such that FLOPS $\propto 2^{\phi_d} \cdot (2^{\phi_w})^2 \cdot (2^{\phi_r})^2$. EfficientNet-B3 represents the Pareto-optimal point for accuracy/efficiency at 12M parameters — significantly smaller than ResNet-50 (25M) while outperforming it on ImageNet. This matters for FL because weight deltas of smaller models compress better, reducing communication cost per round.

The backbone processes $\mathbf{x} \in \mathbb{R}^{224 \times 224 \times 3}$ through 26 MBConv blocks (Mobile Inverted Bottleneck + Squeeze-and-Excitation) and global average pooling:

$$\mathbf{e}_{\text{img}} = \text{GlobalAvgPool}\left(\text{EfficientNet-B3}_{\text{backbone}}(\mathbf{x})\right) \in \mathbb{R}^{1536}$$

#### 3.3.2 Metadata MLP

Two-layer projection from the 13-d one-hot metadata encoding to a 32-d semantic embedding:

$$\mathbf{h}_1 = \text{ReLU}\!\left(\text{BN}(\mathbf{W}_1 f(\mathbf{m}_i) + \mathbf{b}_1)\right), \quad \mathbf{W}_1 \in \mathbb{R}^{64 \times 13}$$
$$\mathbf{e}_{\text{meta}} = \text{ReLU}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2) \in \mathbb{R}^{32}, \quad \mathbf{W}_2 \in \mathbb{R}^{32 \times 64}$$

The 32-d output dimension is 2% of the 1568-d fusion space — deliberately asymmetric. The image signal dominates classification; metadata provides a calibrating prior for phenotype-specific features.

#### 3.3.3 Fusion, Logits, and Probabilities

$$\mathbf{z} = [\mathbf{e}_{\text{img}}\ ;\ \mathbf{e}_{\text{meta}}] \in \mathbb{R}^{1568}$$

$$\mathbf{l} = \mathbf{W}_c \cdot \text{Dropout}(\mathbf{z}, p=0.3) + \mathbf{b}_c \in \mathbb{R}^2, \quad \mathbf{W}_c \in \mathbb{R}^{2 \times 1568}$$

The output $\mathbf{l} = [l_0, l_1]^T$ are **logits** — unbounded real-valued pre-probability scores. The softmax transformation yields calibrated (post-scaling) probabilities:

$$\hat{p}_c = \frac{\exp(l_c / T)}{\exp(l_0/T) + \exp(l_1/T)}, \quad c \in \{0, 1\}$$

where $T$ is the learned temperature parameter from calibration (Section 3.11). At inference: $T = T^*$; during training: $T = 1$ (no scaling applied during local updates).

---

### 3.4 Class-Weighted Focal Loss

#### 3.4.1 Failure Modes of Standard Cross-Entropy

Standard cross-entropy $\mathcal{L}_{\text{CE}} = -\log(p_t)$ fails in two compounding ways under the 17:1 class imbalance:

1. **Majority-class domination:** 9,199 benign samples generate 17$\times$ more total loss than 541 malignant samples, pushing the optimizer toward the degenerate all-benign solution.

2. **Easy-example saturation:** Even within the malignant class, well-classified examples ($p_t \approx 0.9$) contribute $-\log(0.9) \approx 0.105$ loss each round — preventing gradient concentration on hard, ambiguous lesions.

#### 3.4.2 Focal Loss (Lin et al., 2017 [6])

Let $p_t = \hat{p}_{y_j}$ (the model's predicted probability for the true class). The focal loss:

$$\mathcal{L}_{\text{focal}}(p_t) = -(1 - p_t)^{\gamma_f} \log(p_t)$$

The modulation factor $(1-p_t)^{\gamma_f}$ at $\gamma_f = 2$: easy examples ($p_t = 0.9$) receive weight $(0.1)^2 = 0.01$ (100$\times$ reduction); hard examples ($p_t = 0.1$) receive weight $(0.9)^2 = 0.81$ (minimal reduction).

#### 3.4.3 Per-Client Class Weighting

Client $i$ computes inverse-frequency weights **locally** — raw class counts are never transmitted:

$$\alpha_c^{(i)} = \frac{n_{\text{total}}^{(i)}}{2 \cdot n_c^{(i)}}, \quad \hat{\alpha}_c^{(i)} = \frac{\alpha_c^{(i)}}{\alpha_0^{(i)} + \alpha_1^{(i)}}$$

The full **Class-Weighted Focal Loss**:

$$\boxed{\mathcal{L}_{\text{CW-Focal}} = -\frac{1}{N}\sum_{j=1}^{N} \hat{\alpha}_{y_j}^{(i)} \cdot (1-p_t^{(j)})^{\gamma_f} \cdot \log(p_t^{(j)})}$$

At 2% local malignancy ($n_0 = 980$, $n_1 = 20$): $\hat{\alpha}_1 \approx 0.98$, amplifying malignant gradient by $\approx 49\times$.

---

### 3.5 Reliability Signal $R_i^{(t)}$

#### 3.5.1 Definition

$$\boxed{R_i^{(t+1)} = \rho \cdot R_i^{(t)} + (1 - \rho) \cdot \delta^{s_i^{(t)}}}$$

with canonical default $\rho = 0.85$, $\delta = 0.90$, $R_i^{(0)} = 1.0$.

#### 3.5.2 Interpretation

$R_i^{(t)}$ is the **exponential moving average of staleness-adjusted recent systems quality**. It is emphatically not a binary completion indicator. It is a continuous signal in $(0, 1]$ that jointly encodes:

- **Whether** the client responded (a failure contributes $\delta^{s_i} = 0$ by setting $s_i \to \infty$ or using a dedicated failure term of 0)
- **How timely** the response was — an on-time update ($s_i = 0$) contributes $\delta^0 = 1$; a stale update ($s_i = 2$) contributes $\delta^2 = 0.81$; a very stale update ($s_i = 5$) contributes $\delta^5 \approx 0.59$

The term $(1-\rho)$ is the **update rate**: at $\rho = 0.85$, each new observation receives weight $1-\rho = 0.15$, meaning recent behavior accounts for $1 - \rho^k$ of the signal over the most recent $k$ rounds.

**On failure:** Define $\delta^{s_i} := 0$ when no update is received (client failed entirely). This makes $R_i^{(t+1)} = \rho R_i^{(t)}$ on failure — a multiplicative contraction preserving the distinction between total failure and partial staleness.

#### 3.5.3 Mathematical Properties

**Boundedness:** $R_i^{(t)} \in [0, 1]$ for all $t$. *Proof:* By induction. $R_i^{(0)} = 1.0$. If $R_i^{(t)} \in [0,1]$ and $\delta^{s_i} \in [0,1]$, then $R_i^{(t+1)} = \rho R_i^{(t)} + (1-\rho)\delta^{s_i} \in [\rho \cdot 0, \rho \cdot 1 + (1-\rho) \cdot 1] = [0, 1]$. $\square$

**Steady-state behavior:** The fixed point $R^* = \rho R^* + (1-\rho)\bar{q}$ yields:

$$R^* = \bar{q} = \mathbb{E}[\delta^{s_i}]$$

where $\bar{q}$ is the long-run average staleness-adjusted quality. Under perfect on-time operation: $\bar{q} = \delta^0 = 1 \Rightarrow R^* = 1$. Under persistent staleness $s_i = 2$: $\bar{q} = 0.81 \Rightarrow R^* = 0.81$. Under persistent failure: $\bar{q} = 0 \Rightarrow R^* = 0$. The steady state is interpretable as the average deployment quality of the client.

**EMA memory horizon:** The effective half-life of past observations under the EMA with decay $\rho$ is:

$$t_{1/2} = \frac{-\log 2}{\log \rho} = \frac{-\log 2}{\log 0.85} \approx 4.27 \text{ rounds}$$

This is calibrated for the FL setting: transient failures (1–2 rounds) have a visible but recoverable effect; persistent failures (5+ rounds) drive $R_i$ below useful thresholds.

**Numerical trajectory after 5 consecutive failures (starting $R_i = 1.0$):**

| Round | Event | $R_i^{(t)}$ |
|---|---|---|
| 1 | Failure ($\delta^s = 0$) | 0.850 |
| 2 | Failure | 0.723 |
| 3 | Failure | 0.614 |
| 4 | Failure | 0.522 |
| 5 | Failure | 0.444 |
| 6 | On-time ($\delta^0 = 1$) | 0.527 |
| 7 | On-time | 0.598 |
| 8 | On-time | 0.658 |
| 9 | On-time | 0.710 |
| 10 | On-time | 0.753 |

A client recovers meaningfully within 5 on-time rounds after a failure streak — critical for not permanently penalising telemedicine nodes that experience infrastructure outages.

#### 3.5.4 Coupling with $\delta$

The **same staleness parameter $\delta$** appears in both $R_i^{(t)}$ (the reliability update) and $N_i^{\text{eff}}(t)$ (the exposure count, Section 3.6). This is not a coincidence — it is a deliberate **systems coupling**. A single $\delta$ governs:

- How much the aggregation server discounts stale updates
- How much the reliability signal penalises stale arrivals
- How much each stale interaction counts toward the exposure counter

Setting $\delta = 0.90$ coherently across all three mechanisms ensures that the system's definition of "staleness" is unified: a round-3 stale update is $\delta^3 = 0.729$ as useful in aggregation, contributes $0.729$ to $R_i$, and increments $N_i^{\text{eff}}$ by $0.729$. This interpretive consistency is a key systems feature.

---

### 3.6 Effective Interaction Count $N_i^{\text{eff}}(t)$

#### 3.6.1 Definition

$$\boxed{N_i^{\text{eff}}(t) = \sum_{\tau=1}^{t} \delta^{s_i^{(\tau)}}}$$

where $s_i^{(\tau)}$ is the staleness of client $i$'s update at round $\tau$ (with $s_i^{(\tau)} = \infty \Rightarrow \delta^{s_i} = 0$ on failure, contributing nothing).

#### 3.6.2 Interpretation

$N_i^{\text{eff}}(t)$ counts the **effective number of successful manifold interactions** — a staleness-discounted accumulation of the optimization "touches" the global model has received from client $i$'s data distribution.

- **On-time update** ($s_i = 0$): $\delta^0 = 1$ — full interaction credit
- **Stale update** ($s_i = 2$): $\delta^2 = 0.81$ — partial credit
- **Very stale update** ($s_i = 5$): $\delta^5 \approx 0.59$ — reduced credit
- **Failure** ($s_i \to \infty$): $\delta^\infty = 0$ — no credit

It is an **exposure count proxy**, not a novelty counter. The client's data distribution $P_i$ has not changed. What accumulates is the optimizer's exposure to the gradients defined by that distribution.

**Comparison with round count:** A client selected in 20 consecutive rounds contributes $N_i^{\text{eff}} = 20$ (all on-time). A client selected in 20 rounds with mixed staleness (10 on-time + 5 stale $s=1$ + 5 stale $s=2$) contributes $N_i^{\text{eff}} = 10 + 5 \times 0.9 + 5 \times 0.81 = 18.55$. The metric is finer-grained than raw selection count.

**Coupling with $\delta$:** The same staleness parameter governs both the aggregation discount and the exposure increment. A stale update that contributes little to the global model also increments $N_i^{\text{eff}}$ by a correspondingly small amount. The system correctly interprets a series of stale interactions as less informative than an equal number of on-time interactions.

---

### 3.7 Marginal Utility Attenuation $U_i^{\text{eff}}(t)$

#### 3.7.1 Definition

$$\boxed{U_i^{\text{eff}}(t) = \frac{D_i}{1 + \lambda \cdot N_i^{\text{eff}}(t)}}$$

with canonical default $\lambda = 0.05$.

#### 3.7.2 Theoretical Grounding: Diminishing Marginal Optimization Returns

The attenuation formula is a zero-communication proxy for a well-documented empirical phenomenon in optimization theory and federated learning: **the marginal benefit of training on a specific data distribution decreases as the global model has already incorporated many updates from that distribution.**

Several theoretical mechanisms jointly support this interpretation:

**Gradient Exhaustion.** In stochastic gradient descent on a smooth loss landscape, the gradient norm $\|\nabla F_i(\mathbf{w}^t)\|$ decreases as training progresses. Near a local optimum of $F_i$, the gradient contribution from client $i$ shrinks. The server cannot observe $\|\nabla F_i(\mathbf{w}^t)\|$ directly without additional communication, but the number of prior interactions $N_i^{\text{eff}}(t)$ is a proxy: more interactions → closer to client-local convergence → smaller marginal gradient.

**Diminishing Marginal Local Loss Reduction.** Define the marginal improvement from one interaction with client $i$ as $\Delta F(\mathbf{w}^t; i) = F(\mathbf{w}^t) - F(\mathbf{w}^{t+1})$. Under the standard convergence analysis [2, 4], $\Delta F$ decreases monotonically along the gradient path. After $N_i^{\text{eff}}$ interactions, the model has moved substantially toward the manifold defined by $P_i$, and subsequent interactions produce diminishing improvement per round.

**Submodular Diminishing Returns.** The function $f(S) = \text{information gained by selecting client set } S$ is known to exhibit submodularity in related active learning settings [see: Krause & Golovin, 2014]. Submodularity directly implies diminishing marginal returns: the value of adding client $i$ to a set $S$ decreases as $|S|$ grows (or, analogously, as the number of prior interactions grows). The attenuation formula $D_i / (1 + \lambda N_i^{\text{eff}})$ is a smooth, computationally tractable proxy for the submodular marginal gain.

**Representation Rehearsal Dynamics.** In continual learning, models trained repeatedly on the same data distribution "rehearse" learned representations without acquiring new information. The optimization trajectory converges to a neighborhood of the manifold defined by client $i$'s loss, and additional training on $\mathcal{D}_i$ produces increasingly small perturbations to the global model.

**Why this is NOT "novelty consumed":** The diversity $D_i$ is a property of the hospital's patient population. It does not change when the server trains on this hospital's data. What changes is the **marginal optimization utility** of that training — how much additional gradient signal the global model can extract from this particular manifold before hitting an optimization plateau. The attenuation captures the latter, not the former.

#### 3.7.3 Mathematical Properties

**Boundedness:** $U_i^{\text{eff}}(t) \in [0, 1]$ for all $t$. Since $D_i \in [0,1]$ and $1 + \lambda N_i^{\text{eff}} \geq 1$:

$$0 \leq U_i^{\text{eff}}(t) = \frac{D_i}{1 + \lambda N_i^{\text{eff}}} \leq \frac{1}{1} = 1$$

**Monotone decrease:** $\partial U_i^{\text{eff}} / \partial N_i^{\text{eff}} = -\lambda D_i / (1 + \lambda N_i^{\text{eff}})^2 < 0$. Marginal utility strictly decreases with cumulative exposure.

**Calibrated via $\lambda = 0.05$:** At $N_i^{\text{eff}} = 20$:

$$U_i^{\text{eff}} = \frac{D_i}{1 + 0.05 \times 20} = \frac{D_i}{2} = 0.5 D_i$$

Marginal utility halves after 20 effective interactions — consistent with the empirical observation that SGD on smooth losses reduces gradient norm by roughly half within $\mathcal{O}(1/\eta L)$ steps. The half-life of 20 rounds matches a typical federated training horizon of $T = 50$ rounds, ensuring the attenuation is active and meaningful throughout training without prematurely collapsing diversity bonuses.

**Heavy tail:** As $N_i^{\text{eff}} \to \infty$, $U_i^{\text{eff}} \to 0$ but never reaches 0. Even a frequently-selected client retains a small positive utility contribution, preventing score collapse to exactly $R_i$.

---

### 3.8 Canonical Selector Score

#### 3.8.1 Definition

$$\boxed{\text{score}_i^{(t)} = R_i^{(t)} \cdot \left(1 + \gamma \cdot \frac{D_i}{1 + \lambda N_i^{\text{eff}}(t)}\right)}$$

with canonical defaults $\gamma = 1.0$, $\lambda = 0.05$, $\rho = 0.85$, $\delta = 0.90$.

#### 3.8.2 Structural Interpretation

The score has a **multiplicative gating structure**: reliability $R_i^{(t)}$ gates the diversity-augmented utility. This is qualitatively different from additive mixture scores.

- **Additive** ($\alpha R_i + \beta D_i$): A high-diversity client with zero reliability can receive a high score purely from $D_i$. The server selects a client that will almost certainly fail to respond.

- **Multiplicative** ($R_i \cdot (1 + \gamma U_i^{\text{eff}})$): If $R_i \approx 0$, the score approaches 0 regardless of diversity. Reliability is a necessary condition for the diversity bonus to matter. This is operationally correct: a client with valuable data that never responds is worthless to the aggregation.

The factor $(1 + \gamma U_i^{\text{eff}})$ is a **bounded utility multiplier** that augments baseline reliability. With $\gamma = 1.0$ and $U_i^{\text{eff}} \in [0,1]$, the multiplier is in $[1, 2]$, giving a score range of $[0, 2]$.

#### 3.8.3 Derivation from First Principles

The score can be derived as an expected utility of selecting client $i$:

$$\mathbb{E}[\text{utility}_i^{(t)}] = \underbrace{P(\text{useful update})}_{\approx R_i^{(t)}} \times \underbrace{\text{optimization value of update}}_{\propto 1 + \gamma U_i^{\text{eff}}(t)}$$

The probability of receiving a useful update is approximated by the reliability EMA $R_i^{(t)}$ — a client that has historically arrived on time with low staleness will likely do so again. The optimization value is the aggregate effect of: (a) the baseline value of any update (normalised to 1), plus (b) the additional diversity bonus $\gamma U_i^{\text{eff}}(t)$ reflecting the marginal utility of incorporating this specific data distribution given current exposure.

This expected-utility interpretation justifies the multiplicative structure: expected value of a random variable factorises as probability × magnitude.

#### 3.8.4 Key Properties

**Monotonicity in $R_i$:** $\partial \text{score}_i / \partial R_i = 1 + \gamma U_i^{\text{eff}} \geq 1 > 0$. Score is strictly increasing in reliability.

**Monotonicity in $D_i$:** $\partial \text{score}_i / \partial D_i = R_i \cdot \gamma / (1 + \lambda N_i^{\text{eff}}) \geq 0$. Score is non-decreasing in diversity (strictly so for $R_i > 0$).

**Monotone decrease in $N_i^{\text{eff}}$:** $\partial \text{score}_i / \partial N_i^{\text{eff}} = -R_i \gamma \lambda D_i / (1 + \lambda N_i^{\text{eff}})^2 \leq 0$. Score is non-increasing in cumulative exposure.

**Boundedness:** $\text{score}_i \in [0, 2]$ since $R_i \in [0,1]$ and the multiplier $\in [1, 2]$.

**Asymptotic behavior:** As $N_i^{\text{eff}} \to \infty$:

$$\text{score}_i^{(t)} \to R_i^{(t)} \cdot \left(1 + \gamma \cdot 0\right) = R_i^{(t)}$$

This is the **exploration-to-exploitation transition**: early in training, clients with high $D_i$ receive substantially elevated scores (diversity bonus up to $+\gamma D_i$), encouraging the model to explore minority-phenotype manifolds. As training progresses and $N_i^{\text{eff}}$ accumulates, the diversity bonus attenuates, and the score converges to pure reliability. The system transitions from **exploration** (finding diverse gradient directions) to **convergence stabilisation** (selecting the most reliably responding, well-proven clients).

This transition is **automatic and parameter-free** — it emerges from the accumulation of $N_i^{\text{eff}}$ without any additional scheduling mechanism. The rate of transition is controlled solely by $\lambda$.

**Sensitivity analysis ($\gamma$):** For $\gamma = 0$ the score collapses to $R_i$ (pure reliability, equivalent to FedLesScan [3]). For $\gamma \to \infty$ the score becomes $R_i \cdot \gamma D_i/(1+\lambda N_i^{\text{eff}})$, overweighting diversity relative to reliability. $\gamma = 1.0$ achieves a balanced range of $[0, 2]$ where diversity can at most double the baseline reliability utility.

**Corner cases:**
- $R_i = 0$, $D_i = 1$, $N_i^{\text{eff}} = 0$: $\text{score} = 0$. A never-responsive client with maximum diversity contributes nothing. Correct — it cannot be selected usefully.
- $R_i = 1$, $D_i = 0$, $N_i^{\text{eff}} = 0$: $\text{score} = 1$. A maximally reliable client with global-average distribution earns baseline score. Correct.
- $R_i = 1$, $D_i = 1$, $N_i^{\text{eff}} = 0$: $\text{score} = 2.0$. Maximum possible score for a fresh, perfectly reliable, maximally novel client. Correct for the first round of training.

**Worked example at $t = 10$:**

| Client | $R_i$ | $D_i$ | $N_i^{\text{eff}}$ | $U_i^{\text{eff}}$ | $\text{score}_i$ |
|---|---|---|---|---|---|
| C1 (Oncology) | 0.960 | 0.496 | 8.00 | 0.354 | $0.960 \times 1.354 = \mathbf{1.300}$ |
| C9 (Telemedicine) | 0.420 | 1.000 | 2.90 | 0.873 | $0.420 \times 1.873 = \mathbf{0.787}$ |

C1 scores higher due to superior reliability despite lower diversity and higher exposure. C9 scores meaningfully (0.787 vs. maximum 2.0) due to high diversity compensating its low reliability — not zero, not silenced.

---

### 3.9 Boltzmann Client Scheduling

#### 3.9.1 Motivation: Why Not Greedy Top-$K$?

Greedy top-$K$ selection (sort by score, take highest $K$) has three critical failure modes in asynchronous FL:

1. **Async stampede:** The same $K$ high-score clients are selected every round. They submit updates simultaneously. The server receives a surge of updates from identical clients while others starve.

2. **Starvation:** Low-but-non-zero score clients — precisely the minority-phenotype telemedicine nodes — are permanently excluded. This is operationally equivalent to not federating with them.

3. **Score ties and instability:** Deterministic selection is brittle under score perturbations. Small changes in $R_i$ (from a single straggler event) cause abrupt selection changes.

Epsilon-greedy (select greedily with prob $1-\varepsilon$, uniformly random with prob $\varepsilon$) partially addresses starvation but decouples exploration probability from score information, selecting a failing client with probability $\varepsilon / K$ regardless of whether it has any useful data.

UCB-style augmentation ($\text{score}_i + c/\sqrt{n_i}$) adds confidence intervals borrowed from bandit literature but requires stationarity assumptions violated by the non-stationary FL loss landscape.

#### 3.9.2 Boltzmann Sampling: Definition

$$\boxed{p_i^{(t)} = \frac{\exp\!\left(\text{score}_i^{(t)} / \tau\right)}{\displaystyle\sum_{j=1}^{K} \exp\!\left(\text{score}_j^{(t)} / \tau\right)}}$$

Select $C = 6$ clients **without replacement** by sampling from this categorical distribution $K$ times, removing each sampled client from the pool before the next draw.

Canonical default: $\tau = 0.20$.

#### 3.9.3 Temperature Calibration

The Boltzmann temperature $\tau$ controls the sharpness of the sampling distribution. Since $\text{score}_i \in [0, 2]$, $\tau$ must be calibrated relative to this scale. For two clients with scores $s_1 > s_2$, the selection probability ratio is:

$$\frac{p_1}{p_2} = \exp\!\left(\frac{s_1 - s_2}{\tau}\right)$$

At $\tau = 0.20$ with a score gap $s_1 - s_2 = 0.8$ (e.g., C1 vs. C9 in the worked example):

$$\frac{p_1}{p_2} = e^{0.8/0.20} = e^4 \approx 54.6$$

The higher-scoring client is approximately 54$\times$ more likely to be selected than the lower-scoring one — **strong exploitation** that respects the score ranking. Yet $p_2 > 0$: the lower-scoring client has non-zero selection probability, guaranteeing no permanent starvation.

At $\tau \to 0$: Boltzmann sampling collapses to greedy top-$K$ (deterministic, all probability mass on the highest-score client). At $\tau \to \infty$: uniform random selection (ignores scores entirely). $\tau = 0.20$ sits in the strong-exploitation regime appropriate for the score scale.

**Boltzmann probabilities for a six-client example at $t = 10$:**

| Client | Score | $\exp(\text{score}/0.20)$ | Probability |
|---|---|---|---|
| C2 (Oncology B) | 1.550 | 2,688 | 0.502 |
| C1 (Oncology A) | 1.300 | 713 | 0.143 |
| C3 (General Derm) | 1.400 | 1,202 | 0.237 |
| C5 (General Derm) | 1.200 | 403 | 0.087 |
| C6 (Primary Care) | 0.900 | 90 | 0.020 |
| C9 (Telemedicine) | 0.787 | 51 | 0.011 |

C9 has probability 0.011 — small but strictly positive. Over 50 training rounds, it is expected to be selected approximately $50 \times 6 \times 0.011 \approx 3.3$ times, ensuring the global model receives gradient signal from the dark-skin, telemedicine manifold. Under deterministic top-$K$, it would receive zero.

#### 3.9.4 Theoretical Justifications

**No starvation:** $\forall i$: $p_i^{(t)} > 0$ since $\exp(\text{score}/\tau) > 0$ for all finite scores. No client is permanently excluded as long as $\text{score}_i > 0$ (equivalently, $R_i > 0$, i.e., the client has responded at least once).

**Score ranking respected:** $\text{score}_i > \text{score}_j \Rightarrow p_i > p_j$. Boltzmann sampling is order-preserving: higher scores receive higher selection probabilities.

**Prevents async stampede:** Under Boltzmann sampling without replacement, the six selected clients for each round are a random subset whose composition varies across rounds. The server receives diverse update combinations, smoothing the gradient aggregation and preventing any fixed cohort from dominating.

**Smooth exploration-exploitation:** As $N_i^{\text{eff}} \to \infty$, $\text{score}_i \to R_i$, and the Boltzmann distribution concentrates on high-reliability clients. The system's scheduling behaviour transitions from diversity-weighted exploration to reliability-weighted exploitation without any explicit annealing schedule — the same automatic transition noted in Section 3.8.4.

**Strongest theoretical grounding:** Boltzmann / softmax sampling is the optimal action-selection strategy under the principle of maximum entropy subject to expected utility constraints [Jaynes, 1957]. Among all distributions over clients that achieve a given expected score, the Boltzmann distribution maximises entropy — it is the least-presumptuous selector that respects score information.

---

### 3.10 Staleness-Weighted Aggregation with Diversity Bonus

#### 3.10.1 FedAvg Baseline

Standard FedAvg [2]: $\mathbf{w}^{t+1} = \mathbf{w}^t + \sum_{i \in \mathcal{A}^t} (n_i/n) \cdot \Delta\mathbf{w}_i$. Synchronous, staleness-blind, diversity-blind.

#### 3.10.2 FedMelanoma Aggregation

Let $\mathcal{A}^t \subseteq \mathcal{S}^t$ be the set of arrived clients (on-time plus buffered stragglers). The update rule:

$$\boxed{\mathbf{w}^{t+1} = \mathbf{w}^t + \sum_{i \in \mathcal{A}^t} c_i \cdot \Delta\mathbf{w}_i, \quad \Delta\mathbf{w}_i = \mathbf{w}_i^{t+1} - \mathbf{w}^t}$$

$$c_i = \frac{\tilde{c}_i}{\sum_{j \in \mathcal{A}^t} \tilde{c}_j}, \qquad \tilde{c}_i = \frac{n_i}{n_{\mathcal{A}}} \cdot \delta^{s_i} \cdot (1 + \gamma D_i)$$

where $n_\mathcal{A} = \sum_{j \in \mathcal{A}^t} n_j$, $s_i = t - t_i^{\text{gen}}$ is staleness, and $D_i$ is the **static diversity prior** from Section 3.2.

**Three factors of $\tilde{c}_i$:**

- **$n_i / n_\mathcal{A}$ (size weight):** Proportional contribution; consistent with the maximum likelihood interpretation of FedAvg [2].
- **$\delta^{s_i}$ (staleness discount):** Reduces contribution of outdated gradients, bounding the staleness-induced convergence error (Section 3.12).
- **$(1 + \gamma D_i)$ (diversity amplification):** Increases weight of clients whose data distribution diverges from the global mean. This is the aggregation-side complement to the selection-side diversity bonus. Note $\gamma$ here is the same parameter as in the selector score, providing consistent diversity weighting across selection and aggregation.

**Normalisation:** $\sum_{i \in \mathcal{A}^t} c_i = 1$ ensures the update is well-scaled regardless of how many clients arrive — a critical property under 40–60% straggler rates.

#### 3.10.3 Straggler Handling: Adaptive Time Window

$$W_t = \max\!\left(\tilde{T}_{\text{med}} + k \cdot \text{MAD}(T_1, \ldots, T_n),\ W_{\min}\right), \quad k = 2.5$$

where $\tilde{T}_{\text{med}}$ and MAD are computed from the 10-round rolling history of per-client completion times. Aggregation triggers on: **(A)** quorum $|\mathcal{A}^t| \geq \rho_q \cdot C$ (default $\rho_q = 0.70$), or **(B)** window expiry $W_t$, whichever comes first.

Late-arriving updates are buffered with staleness $s_i = 1$ for inclusion in round $t+1$; updates not received by $t + S_{\max}$ (default $S_{\max} = 3$) are discarded.

---

### 3.11 Logit Calibration & Clinical Reliability

#### 3.11.1 The Overconfidence Problem

Deep neural networks trained with focal loss are systematically overconfident [14]: the focal modulation suppresses easy-example gradients, causing the model to push predictions toward confidence extremes. When the calibrated model reports $\hat{p}_{\text{malignant}} = 0.92$, a physician using this confidence for triage decisions should understand whether "92% confident" is accurate or inflated.

#### 3.11.2 Temperature Scaling

$$\hat{\mathbf{p}}_{\text{cal}} = \text{softmax}(\mathbf{l} / T^*), \quad T^* = \underset{T > 0}{\arg\min} \ \frac{1}{|\mathcal{D}_{\text{val}}|}\sum_{j \in \mathcal{D}_{\text{val}}} -\log \hat{p}_{y_j}^{\text{cal}}(T)$$

Temperature scaling [14] is post-hoc and affects only confidence magnitudes, not the argmax — predictions are unchanged. $T > 1$ softens overconfident outputs toward 0.5; $T < 1$ sharpens them. The NLL minimisation objective is convex in $T$, admitting exact solution via bisection.

In FedMelanoma, $T^*$ is calibrated at the server every 5 rounds on the held-out validation set, stored alongside the global model checkpoint.

#### 3.11.3 Expected Calibration Error

$$\text{ECE} = \sum_{k=1}^{K_{\text{bin}}} \frac{|\mathcal{B}_k|}{n} \left| \text{acc}(\mathcal{B}_k) - \text{conf}(\mathcal{B}_k) \right|$$

Target post-calibration: ECE $< 0.03$. At ECE $= 0.10$: when the model says "90% confident malignant," it is correct only $\approx 80\%$ of the time — at population scale, this 10% miscalibration affects hundreds of patients per year.

---

### 3.12 Theoretical Convergence Analysis

#### 3.12.1 Assumptions

**A1** ($L$-smoothness): $\|\nabla F_i(\mathbf{u}) - \nabla F_i(\mathbf{v})\| \leq L\|\mathbf{u}-\mathbf{v}\|$ for all $i$.  
**A2** (Bounded variance): $\mathbb{E}\|\nabla F_i(\mathbf{w};\xi) - \nabla F_i(\mathbf{w})\|^2 \leq \sigma^2$.  
**A3** (Bounded Non-IID dissimilarity): $\mathbb{E}\|\nabla F_i(\mathbf{w}) - \nabla F(\mathbf{w})\|^2 \leq \zeta^2$.  
**A4** (Bounded staleness): $s_i \leq S_{\max}$ enforced by the quorum/discard policy.

#### 3.12.2 Convergence Bound

Under Assumptions A1–A4, with learning rate $\eta \leq 1/(4EL)$ and $|\mathcal{A}^t| \geq \tau_{\text{eff}} = \lfloor 0.7 \times 6 \rfloor = 4$:

$$\frac{1}{T}\sum_{t=0}^{T-1}\mathbb{E}\|\nabla F(\mathbf{w}^t)\|^2 \leq \underbrace{\mathcal{O}\!\left(\frac{F(\mathbf{w}^0)-F^*}{\eta\,\tau_{\text{eff}}\,T}\right)}_{\text{Term 1: convergence}} + \underbrace{\mathcal{O}(\eta L\sigma^2)}_{\text{Term 2: noise}} + \underbrace{\mathcal{O}(\eta^2 L^2 E^2 \zeta^2)}_{\text{Term 3: Non-IID drift}} + \underbrace{\mathcal{O}(\gamma^2\eta^2 L^2 E^2\sigma^2)}_{\text{Term 4: diversity overhead}}$$

**Term 4** represents the convergence cost of the diversity bonus $(1 + \gamma D_i)$: for $\gamma = 1.0$, this is $\mathcal{O}(\eta^2 L^2 E^2 \sigma^2)$ — a constant additive overhead on stochastic variance. This is the formal price of minority-phenotype preservation. Given $D_i \in [0,1]$, the effective contribution of Term 4 is bounded and does not grow with $T$.

**Boltzmann vs. deterministic selection:** Stochastic Boltzmann sampling introduces additional variance in $\tau_{\text{eff}}$ (the realised number of arrived clients per round). However, since $p_i > 0$ for all $i$ and scores are bounded in $[0,2]$, the expected effective quorum satisfies $\mathbb{E}[\tau_{\text{eff}}] \geq C \cdot p_{\min}(1-\delta_{\text{fail}})$ where $p_{\min} = \min_i p_i > 0$ and $\delta_{\text{fail}}$ is the per-client failure rate. This expectation is bounded away from zero, ensuring Term 1 remains $\mathcal{O}(1/T)$.

---

---

## 4. End-to-End Implementation Strategy

### Phase 1: Data Pipeline

**1.1 Preprocessing.** The dataset is fully clean (no missing values). The `isic_id` column maps to ISIC archive image files. All three metadata axes are complete for all 9,740 records.

**1.2 Metadata Encoding.** One-hot encoding to 13 dimensions: Fitzpatrick (6), modality (3), confirmation (4). Encoding order is fixed globally — any client-server mismatch would corrupt Metadata MLP weights.

**1.3 Image Preprocessing.** Resize to 224×224. Normalize with ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). DullRazor morphological hair removal for dermoscopic images. Training augmentation: horizontal/vertical flip ($p=0.5$), rotation ($\pm15°$), color jitter (brightness/contrast $\pm0.2$, saturation $\pm0.1$), random erasing ($p=0.1$).

**1.4 Stratified Split.** 70/15/15 split, stratified by `label` and `image_type`. Clinical overview (54.78% malignancy) must be stratified to prevent test-set contamination.

---

### Phase 2: Client Simulation / Hospital Setup

**2.1 Non-IID Partitioning.** K=10 partitions via softmax-temperature assignment ($\tau_{\text{soft}} = 0.5$, $\mathbf{W} \in \mathbb{R}^{10 \times 3}$). Expected client sizes: ~300 (telemedicine) to ~2,500 (tertiary oncology). Expected per-client malignancy range: ~0–2% to ~30–35%.

**2.2 Straggler Model.** Per-client straggler behavior sampled from a **Beta distribution** per round:

$$p_{\text{complete}}^{(i)} \sim \text{Beta}(a_i, b_i)$$

| Tier | Clients | $(a_i, b_i)$ | Mean | Compute delay |
|---|---|---|---|---|
| Reliable | C1–C5 | $(19, 1)$ | 0.95 | Log-normal $\mu=1.39$, $\sigma=0.3$ sec |
| Moderate | C6–C8 | $(7, 3)$ | 0.70 | Log-normal $\mu=2.77$, $\sigma=0.5$ sec |
| High-Straggler | C9–C10 | $(2, 3)$ | 0.40 | Log-normal $\mu=4.09$, $\sigma=0.8$ sec |

The log-normal distribution correctly models right-skewed compute delays (occasional extreme outliers from server contention).

**2.3 Client Isolation.** Each client is an independent Python subprocess. No inter-client communication. Clients transmit only: weight deltas $\Delta\mathbf{w}_i$, dataset size $n_i$, and timing metadata. No raw images, labels, or patient records leave the hospital.

---

### Phase 3: Training Loop

**3.1 Round lifecycle (five stages):**

1. **Selection.** Server computes scores $\text{score}_i^{(t)}$ for all $K$ clients, constructs Boltzmann probabilities, samples $C=6$ clients without replacement.

2. **Broadcast.** Server transmits $\mathbf{w}^t$ (zstd-compressed) to all $i \in \mathcal{S}^t$ via mTLS gRPC. Non-selected clients receive nothing.

3. **Local training.** Each $i \in \mathcal{S}^t$ resets optimizer state, trains for $E=5$ epochs with class-weighted focal loss and AdamW ($\eta=10^{-4}$, weight decay $10^{-4}$). Computes $\Delta\mathbf{w}_i = \mathbf{w}_i^{t+1} - \mathbf{w}^t$.

4. **Collection.** Straggler detector monitors against $W_t$. Trigger: quorum $|\mathcal{A}^t| \geq 4$ or window expiry, whichever first.

5. **Aggregation and state update.** Compute $c_i$ weights, aggregate, checkpoint. Update $R_i^{(t+1)}$ and $N_i^{\text{eff}}(t+1)$ for all clients in $\mathcal{S}^t$.

**3.2 State update after each round:**

For each selected client $i \in \mathcal{S}^t$:
$$R_i^{(t+1)} = \rho R_i^{(t)} + (1-\rho)\delta^{s_i^{(t)}}$$
$$N_i^{\text{eff}}(t+1) = N_i^{\text{eff}}(t) + \delta^{s_i^{(t)}}$$

For non-selected clients ($i \notin \mathcal{S}^t$): $R_i^{(t+1)} = R_i^{(t)}$, $N_i^{\text{eff}}(t+1) = N_i^{\text{eff}}(t)$ (no change — they did not interact).

**3.3 Optimizer reset.** AdamW momentum and variance buffers are **reset at the start of each round's local training.** This prevents stale first and second moment estimates from prior rounds accumulating artificial drift.

**3.4 Rationale for $E=5$.** Term 3 of the convergence bound grows as $\mathcal{O}(\eta^2 E^2 \zeta^2)$. Li et al. [4] identify $E=5$ as the empirically stable setting under high Non-IID heterogeneity. Smaller $E$ approaches FedSGD (expensive communication); larger $E$ risks divergence.

---

### Phase 4: Communication

**4.1 gRPC over Mutual TLS.** HTTP/2 multiplexing enables concurrent bidirectional streaming from all $C=6$ selected clients simultaneously. Protocol Buffers v3 serializes float32 tensors with minimal overhead. mTLS provides: AES-256-GCM encryption (weight deltas must not be interceptable due to gradient inversion risk [se e Zhu et al., 2019 [16]]), mutual X.509 authentication (no fake hospital injections), and certificate revocation.

**4.2 Compression.** zstd compression at Level 3 achieves ~3.5× ratio on float32 tensors ($\approx 45$ MB model → $\approx 13$ MB delta). Bandwidth-adaptive levels: Level 9 for $< 10$ Mbps, Level 19 for $< 1$ Mbps (telemedicine). Total server ingress over $T=50$ rounds: $\approx 6 \times 13 \times 50 = 3.9$ GB.

**4.3 Delta-only transmission.** Transmitting $\Delta\mathbf{w}_i$ rather than $\mathbf{w}_i^{t+1}$ enables better compression (deltas are sparser) and reduces bandwidth by $\approx 50\%$ in practice.

---

### Phase 5: Aggregation (Step-by-Step)

1. Collect $\{(\Delta\mathbf{w}_i, n_i, s_i)\}_{i \in \mathcal{A}^t}$ including buffered straggler updates (staleness incremented by 1).
2. Retrieve $D_i$ (static, computed at registration; not recomputed during training).
3. Compute $\tilde{c}_i = (n_i / n_\mathcal{A}) \cdot \delta^{s_i} \cdot (1 + \gamma D_i)$.
4. Normalise: $c_i = \tilde{c}_i / \sum_j \tilde{c}_j$. Verify $\sum c_i = 1$.
5. Update: $\mathbf{w}^{t+1} = \mathbf{w}^t + \sum_i c_i \Delta\mathbf{w}_i$.
6. If $t \bmod 5 = 0$: calibrate $T^*$ on validation set; checkpoint $(\mathbf{w}^{t+1}, T^*)$.
7. Update $R_i$ and $N_i^{\text{eff}}$ for all $i \in \mathcal{S}^t$ (on-time or straggler).

---

### Phase 6: Selector State Maintenance

At the start of round $t$:

1. For each $i \in \mathcal{H}$: compute $\text{score}_i^{(t)}$ from current $R_i^{(t)}$, static $D_i$, current $N_i^{\text{eff}}(t)$.
2. Compute Boltzmann probabilities $p_i^{(t)} = \exp(\text{score}_i^{(t)}/\tau) / Z$.
3. Sample $C=6$ clients without replacement.
4. Transmit global model to selected clients. Begin round timer.

$D_i$ values are computed once at client registration using the OT pipeline against the initial global distribution centroid, and **remain fixed throughout training**. They are static properties of the hospital's patient population distribution, not a function of training progress. The global centroid is recomputed if a new client registers or if distribution drift is detected.

---

### Phase 7: Deployment

**Hospital nodes.** Docker containers with the FedMelanoma client software access local DICOM archives via a data access adapter (HL7 FHIR/DICOM SR/flat-file). Minimum hardware: 8 GB RAM, 50 GB storage, 1 Mbps sustained network (CPU fallback supported). Recommended: NVIDIA RTX 3060 or equivalent.

**Telemedicine nodes.** Offline training mode: local epochs run during idle periods, updates transmitted opportunistically. zstd Level 19 compression for sub-1 Mbps nodes. Training state checkpointed every 100 batches for resumable operation after power loss.

**Registration protocol.** Client transmits: $n_i$ (dataset size), $[P_\varphi^{(i)}, P_\tau^{(i)}, P_\lambda^{(i)}]$ (three distribution vectors, 13 floats total), and malignancy rate (scalar). Server computes $D_i$, $M_i$ (coverage), and assigns initial $R_i^{(0)} = 1.0$, $N_i^{\text{eff}}(0) = 0$. No raw patient records transmitted.

---

### Phase 8: Monitoring

**Performance monitoring.** Every 5 rounds: AUROC, AUPRC, Sensitivity, Specificity, F1 (malignant), Balanced Accuracy, ECE, Brier Score on the server's held-out test set. Alerts on: AUROC drop $>0.03$, Sensitivity $<0.75$, ECE $>0.05$.

**Fairness monitoring.** Per-round Fitzpatrick fairness gap:

$$\Delta_{\text{fair}}^{(t)} = \max_\varphi \text{Sensitivity}(\varphi) - \min_\varphi \text{Sensitivity}(\varphi)$$

A non-increasing $\Delta_{\text{fair}}^{(t)}$ validates diversity-aware selection. Sudden increases trigger investigation of straggler rates for minority-phenotype clients.

**Selector state monitoring.** Per-client: $R_i^{(t)}$ trajectory, $N_i^{\text{eff}}(t)$, $\text{score}_i^{(t)}$, $p_i^{(t)}$, and selection frequency vs. Boltzmann expectation. A client's selection frequency significantly below $p_i \times t$ indicates unlucky sampling and can be monitored for accumulating bias.

**Distribution drift detection.** KS test (Fitzpatrick) and chi-squared goodness-of-fit (modality, confirmation) per client per 10-round sliding window. Detected drift at $p < 0.05$ triggers $D_i$ recomputation.

**Three-tier logging.** Operational (round metadata, no weights), mathematical (scores, $R_i$, $N_i^{\text{eff}}$, loss curves), and audit (SHA-256 model hashes, participation certificates, write-once tamper-evident).

---

## 5. Evaluation & Validation Framework

### 5.1 Primary Classification Metrics

**AUROC** — Area Under the ROC Curve. Probabilistic interpretation: $P(\hat{p}_1^{(\text{mal})} > \hat{p}_1^{(\text{ben})})$. Threshold-independent; appropriate as the primary metric for 5.55% prevalence. A degenerate all-benign predictor achieves AUROC = 0.5.

**AUPRC** — Area Under the Precision-Recall Curve [17]. More informative than AUROC under high imbalance. Baseline AUPRC ≈ 0.0555 (prevalence). The degenerate all-benign predictor achieves AUPRC = 0, correctly penalised unlike AUROC.

**Sensitivity** $= \text{TP}/(\text{TP}+\text{FN})$. Primary clinical objective: missed melanoma (false negative) is a potentially fatal failure mode. Target: Sensitivity $\geq 0.85$.

**Specificity** $= \text{TN}/(\text{TN}+\text{FP})$. Secondary clinical objective: unnecessary biopsies are costly but not life-threatening. Target: Specificity $\geq 0.70$.

**F1 (malignant)** $= 2\text{PR}/(P+R)$. Harmonic mean penalises extreme precision-recall imbalance. Computed only for the malignant class.

**Balanced Accuracy** $= (\text{Sensitivity}+\text{Specificity})/2$. Imbalance-invariant single-number summary. Degenerate predictor scores 0.5.

### 5.2 Calibration Metrics

**ECE** $= \sum_k (|\mathcal{B}_k|/n)|\text{acc}(\mathcal{B}_k) - \text{conf}(\mathcal{B}_k)|$ with $K_{\text{bin}} = 10$.

**Brier Score** $= (1/n)\sum_j (\hat{p}_{1,j} - y_j)^2$. Proper scoring rule; decomposes into calibration + resolution.

**Brier Skill Score** $= 1 - \text{BS}/\text{BS}_{\text{clim}}$ where $\text{BS}_{\text{clim}} = 0.0555 \times 0.9445 = 0.0524$.

### 5.3 Federated System Metrics

**Global-local performance gap:** $\Delta_{\text{gap}}^{(i)} = \text{AUROC}_{\text{global}} - \text{AUROC}_{\text{local}}^{(i)}$. Negative gap at a client indicates local overfitting and is a candidate for proximal regularization.

**Fitzpatrick fairness gap:** $\Delta_{\text{fair}} = \max_\varphi \text{Sensitivity}(\varphi) - \min_\varphi \text{Sensitivity}(\varphi)$. Should be non-increasing over training rounds.

**Worst-group accuracy:** $\text{WGA} = \min_{g \in \mathcal{G}} \text{AUROC}(g)$ where $\mathcal{G}$ partitions the test set by client type.

**Selection entropy:** $H(\mathbf{p}^{(t)}) = -\sum_i p_i^{(t)} \log p_i^{(t)}$. Tracks how concentrated selection has become. A collapsing entropy indicates premature convergence to a small client subset.

**Communication efficiency:** $\eta_{\text{comm}} = \text{AUROC at round } T / \text{total bytes transmitted}$.

---

## 6. Comparison with Existing FL Methods

### 6.1 FedAvg (McMahan et al., 2017 [2])

**What it does:** Synchronous weighted averaging of local updates. No staleness handling, no diversity awareness, uniform random selection. Foundational FL algorithm.

**Limitations in our setting:** (1) Synchronous: 40–60% straggler rates block every round. (2) No staleness discount: buffered updates from telemedicine nodes (2–3 rounds stale) carry equal weight to fresh oncology updates. (3) Uniform random selection: minority-phenotype clients selected with probability $1/K$ regardless of distributional value.

**FedMelanoma generalises FedAvg** as the special case $\delta = 1.0$, $\gamma = 0$, and uniform selection ($\tau \to \infty$).

### 6.2 FedAsync / FedBuff (Xie et al., 2019 [7]; Nguyen et al., 2022)

**What it does:** Asynchronous FL with per-update staleness discounting. FedBuff extends this with server-side buffering of multiple updates before aggregation.

**Limitations:** Neither provides diversity-aware aggregation or selection. Minority-phenotype straggler clients are penalised by staleness discount without compensation. No framework for marginal utility tracking.

**FedMelanoma extends FedAsync** with the diversity bonus $(1 + \gamma D_i)$ in aggregation and the Boltzmann selector with exposure-aware attenuation.

### 6.3 FedProx (Li et al., 2020 [4])

**What it does:** Adds proximal regularization $(\mu/2)\|\mathbf{w} - \mathbf{w}^t\|^2$ to the local objective, bounding client drift under Non-IID conditions. Proven convergence under bounded heterogeneity.

**Limitations:** Optimization-side only — constrains local training but does not address aggregation weighting, straggler handling, or client selection diversity. The proximal term bounds Term 3 of the convergence analysis but does nothing about Terms 1, 2, or 4.

**FedMelanoma is complementary to FedProx:** adding the proximal term to local training would reduce $\zeta^2$ (Term 3) while FedMelanoma's aggregation handles staleness and diversity.

### 6.4 FedLesScan (Elzohairy, 2022 [3])

**What it does:** Straggler-aware client selection using reliability EMA only ($\text{score}_i = R_i$). Nearest prior work to FedMelanoma.

**Limitations:** (1) Pure reliability: high-straggler clients with maximum distributional value (dark-skin telemedicine nodes) are progressively deprioritised. (2) No diversity awareness in selection or aggregation. (3) No marginal utility tracking. (4) Greedy top-$K$ selection: starvation of minority-distribution clients.

**FedMelanoma strictly generalises FedLesScan** as the case $\gamma = 0$, $\lambda = 0$, $\tau \to 0$ (greedy deterministic, reliability-only).

### 6.5 Comparison Summary

| Property | FedAvg | FedAsync | FedProx | FedLesScan | **FedMelanoma v4** |
|---|---|---|---|---|---|
| Non-IID convergence | Partial | Partial | ✓ | Partial | ✓ |
| Asynchronous aggregation | ✗ | ✓ | ✗ | ✓ | **✓** |
| Staleness discount (aggregation) | ✗ | ✓ | ✗ | ✗ | **✓** |
| Diversity-weighted aggregation | ✗ | ✗ | ✗ | ✗ | **✓** |
| OT-based diversity prior | ✗ | ✗ | ✗ | ✗ | **✓** |
| Marginal utility attenuation | ✗ | ✗ | ✗ | ✗ | **✓** |
| Stochastic (non-greedy) selection | ✗ | ✗ | ✗ | ✗ | **✓** |
| Minority starvation prevention | ✗ | ✗ | ✗ | ✗ | **✓** |
| Post-hoc calibration | ✗ | ✗ | ✗ | ✗ | **✓** |
| Exploration→exploitation transition | ✗ | ✗ | ✗ | ✗ | **✓** |

---

## 7. Ablation Studies

The following ablations isolate the contribution of each architectural component. All ablations share the same EfficientNet-B3 + Metadata MLP backbone, focal loss, and aggregation protocol.

| Variant | Change from v4 | Expected Effect |
|---|---|---|
| **A0 (Full v4)** | — | Baseline for comparison |
| **A1 — No diversity bonus** | $\gamma = 0$ in selector and aggregation | Reduced minority-phenotype sensitivity; increased $\Delta_{\text{fair}}$ |
| **A2 — No attenuation** | $\lambda = 0$ (no $N_i^{\text{eff}}$ decay) | Overselection of early-explored clients in late rounds; convergence plateau |
| **A3 — Greedy selection** | $\tau \to 0$ (top-$K$ deterministic) | Starvation of low-score minority clients; stampede effect |
| **A4 — Binary reliability** | $R_i \leftarrow \mathbb{1}[\text{on-time}]$ instead of staleness-adjusted EMA | Ignores staleness quality; volatile selection behaviour |
| **A5 — Static reliability** | $R_i^{(t)} = 1 \ \forall i, t$ | No straggler penalty; unreliable clients selected at full diversity rate |
| **A6 — TV diversity** | Replace OT with total variation distance | Underestimates clinical overview novelty; poorer minority preservation |
| **A7 — No calibration** | $T^* = 1$ (no temperature scaling) | Elevated ECE; clinically unsafe confidence estimates |
| **A8 — Synchronous** | Wait for all $C$ clients (no straggler buffer) | Round delays of $W_{\max} = 600$s every round with high-straggler nodes |

**Key ablation predictions:**

- A1 vs. A0: Fitzpatrick fairness gap $\Delta_{\text{fair}}$ should be substantially larger in A1, demonstrating that the diversity bonus is the primary mechanism for minority-phenotype preservation.
- A2 vs. A0: Late-round performance should plateau earlier in A2, as the system continues heavily selecting early-explored clients whose marginal utility has saturated.
- A3 vs. A0: A3 should show higher worst-group accuracy variance and occasional starvation of C9/C10, visible in per-client contribution weight $c_i$ tracking.
- A6 vs. A0: OT-computed diversity scores should produce better minority-client selection frequency than TV, particularly for clinical overview (OT correctly penalises the large quality gap from dermoscopy; TV does not).

---

## 8. Limitations

**Dataset scope.** The 9,740-sample preprocessed dataset is a curated subset of the ISIC archive. Real deployment requires validation on external cohorts, particularly for Fitzpatrick Types V–VI where the current dataset has 0.00% confirmed malignancy — a data collection artifact, not a biological property.

**Simulated straggler dynamics.** The Beta distribution straggler model is a reasonable approximation of real hospital network behaviour, but does not capture correlated outages (all rural nodes failing simultaneously during a regional network incident), scheduled downtime, or adversarial behaviour.

**Static diversity prior.** $D_i$ is computed at client registration and held fixed. In live deployment, hospital populations shift over time (demographic changes, updated confirmation protocols, new imaging equipment). The drift detection mechanism (KS test, chi-squared) provides a reactive correction, but proactive online diversity updating would be more robust.

**Single-modality client model.** All clients train the same EfficientNet-B3 architecture. In deployment, some hospitals use different imaging systems or resolution standards. Heterogeneous model architectures (personalised FL) would require federated distillation rather than parameter averaging.

**No differential privacy.** Weight deltas can leak patient information via gradient inversion [16]. Full differential privacy guarantees (Gaussian mechanism, $(\varepsilon, \delta)$-DP) would require noise injection that degrades convergence — a fundamental tradeoff not currently resolved in this architecture.

**Boltzmann temperature stability.** $\tau = 0.20$ is calibrated for $\text{score} \in [0, 2]$. If hyperparameter changes cause scores to deviate significantly from this range, $\tau$ requires recalibration. Future work should explore adaptive temperature scheduling.

---

## 9. Future Work

### 9.1 Adaptive Hyperparameter Tuning

The five canonical hyperparameters $(\rho, \delta, \lambda, \gamma, \tau)$ are currently fixed. All can be framed as a joint optimisation problem across rounds:

- **Bayesian optimisation** over the hyperparameter space, with AUROC on the validation set as the objective function. Practical given the small dimensionality (5 parameters).
- **Meta-learning (MAML-style)** [Finn et al., 2017]: learn hyperparameters that generalise across FL tasks with similar heterogeneity structure.
- **Online regret minimisation**: model hyperparameter selection as a multi-armed bandit problem; algorithms such as EXP3 provide sublinear regret guarantees in the non-stationary setting.
- **Reinforcement learning scheduler**: full RL formulation where the state includes $\{R_i^{(t)}, N_i^{\text{eff}}(t), D_i\}$, actions are hyperparameter values, and rewards are round-level validation improvements.

### 9.2 Learnable Diversity Prior

The current OT-based $D_i$ computation uses handcrafted cost matrices $\mathbf{C}_{\text{mod}}$ and $\mathbf{C}_{\text{conf}}$ with manually specified semantic distances. A more principled approach:

- **Embedding-space diversity**: replace metadata distribution comparison with comparison in the learned embedding space of the Metadata MLP. Clients whose metadata embeddings are far from the global centroid (in $\ell^2$ or optimal transport distance in embedding space) receive high $D_i$.
- **Gradient-based diversity**: measure diversity as the angle between client $i$'s gradient direction and the current global gradient, as proposed in [Zhao et al., 2020]. This requires additional communication (gradient norms) but is strictly more informative.
- **End-to-end learned OT weights**: treat $w_\varphi, w_\tau, w_\lambda$ as learnable parameters updated via meta-gradient through the selection objective.

### 9.3 Dynamic Metadata Distribution Updates

The static $D_i$ assumption breaks when hospital populations evolve. Extensions:

- **Continual FL formulation**: clients transmit updated distribution statistics (not raw data) at configurable intervals (e.g., every 10 rounds). Server recomputes $D_i$ and updates Boltzmann probabilities.
- **Online OT tracking**: maintain a sliding-window estimate of each client's current distribution, using reservoir sampling of metadata statistics from recent batches.
- **Drift detection integration**: the existing KS test / chi-squared drift detector triggers $D_i$ recomputation automatically when distribution shift is detected.

### 9.4 Personalised Client Utility Modeling

The canonical hyperparameters $(\rho, \delta, \lambda)$ are global constants applied uniformly to all clients. In heterogeneous deployments:

$$\text{score}_i^{(t)} = R_i^{(t)} \cdot \left(1 + \gamma_i \cdot \frac{D_i}{1 + \lambda_i N_i^{\text{eff}}(t)}\right)$$

with per-client $(\gamma_i, \lambda_i, \delta_i)$. A reliably fast oncology client should have lower $\lambda_i$ (marginal utility decays more slowly — it is always fresh and has limited data to exhaust). A telemedicine client should have higher $\delta_i$ (staleness is expected; penalise it less). Learning these per-client parameters is a natural personalised FL extension.

### 9.5 Reinforcement Learning Scheduler

Formulate the entire selection problem as a Markov Decision Process:

- **State** $\mathcal{S}^t$: $\{R_i^{(t)}, N_i^{\text{eff}}(t), D_i, \text{round } t\}$ for all $K$ clients
- **Action**: select a subset of $C$ clients (or equivalently, a Boltzmann temperature $\tau^{(t)}$)
- **Reward**: improvement in global AUROC per round

Deep Q-learning or policy gradient methods could discover selection strategies that outperform the closed-form Boltzmann policy, particularly in non-stationary environments where the optimal exploration-exploitation tradeoff evolves over training.

### 9.6 Queue-Aware Systems Scheduling

Extend the selector to incorporate real-time infrastructure signals:

- Estimated **GPU availability** and **queue depth** at each hospital node
- **Bandwidth measurements** from recent communication rounds
- **Wall-clock latency predictions** from completion time history

These signals can augment $R_i^{(t)}$ or replace it entirely with a more precise systems-quality predictor, removing the EMA lag for sudden infrastructure changes.

### 9.7 Calibration-Aware Diversity

Current diversity $D_i$ measures distribution mismatch in metadata space. An alternative: weight clients by their **calibration improvement** contribution. A client whose updates historically improved the global model's ECE should be prioritised over one whose updates increased overconfidence, even if their $D_i$ values are similar.

### 9.8 Privacy-Preserving Diversity Estimation

The current metadata profile transmission (13 floats) has negligible privacy leakage. However, in sensitive demographics, even the distribution over Fitzpatrick types could be considered quasi-identifying. Extensions:

- **Local differential privacy (LDP)**: clients add calibrated Laplace or Gaussian noise to distribution vectors before transmission. Server-side OT computation proceeds on noisy distributions; the $D_i$ estimates degrade gracefully.
- **Federated histogram estimation**: use secure aggregation protocols (e.g., Bonawitz et al., 2017) to compute global distribution centroids without revealing per-client distributions.

### 9.9 Theoretical Regret Bounds

The current convergence analysis (Section 3.12) addresses optimization convergence. A complementary theoretical contribution: **stochastic scheduling regret analysis**. Define regret as the gap between the Boltzmann scheduler's cumulative reward and the oracle best-fixed policy. Recent work on combinatorial bandits [Kveton et al., 2014] and online learning provides tools for bounding this regret under the non-stationary FL loss landscape.

### 9.10 Multi-Objective Selector

The current score is a scalar. A true multi-objective formulation would jointly optimise:

- **Convergence utility** (gradient signal quality)
- **Fairness** (proportional representation of demographic groups)
- **Robustness** (Byzantine-fault tolerance)
- **Communication efficiency** (bytes per AUROC improvement)

Pareto-optimal selection in this multi-objective space, using evolutionary algorithms or multi-objective Bayesian optimisation, could produce selection policies that dominate the scalar selector on the fairness-efficiency frontier.

---

## 10. System Configuration Schema

```yaml
# ============================================================
# FedMelanoma v4 — Canonical Configuration
# All canonical hyperparameters are LOCKED at their proven defaults.
# ============================================================

system:
  version: "4.0.0"
  seed: 42
  device: "cuda"
  checkpoint_dir: "models/checkpoints/"
  checkpoint_interval: 5
  audit_log_dir: "logs/audit/"

federated:
  total_rounds: 50
  num_clients: 10
  clients_per_round: 6          # C
  quorum_fraction: 0.70         # ρ_q
  time_window_k: 2.5            # k: W_t = T̃_med + k·MAD
  min_time_window_seconds: 60
  max_time_window_seconds: 600
  max_staleness_rounds: 3       # S_max

# ── CANONICAL HYPERPARAMETERS (LOCKED) ──────────────────────
selector:
  # Reliability EMA decay
  rho: 0.85                     # ρ: EMA half-life ≈ 4.27 rounds
  
  # Staleness discount — shared across reliability, N_eff, aggregation
  delta: 0.90                   # δ: δ^5 ≈ 0.59 (still useful); δ^20 ≈ 0.12 (negligible)
  
  # Exposure attenuation
  lambda_exposure: 0.05         # λ: marginal utility halves at N_eff = 20
  
  # Diversity multiplier
  gamma: 1.0                    # γ: score ∈ [0, 2]; diversity can double reliability utility
  
  # Boltzmann temperature (calibrated to score scale [0,2])
  tau_boltzmann: 0.20           # τ: at gap=0.8, ratio = e^4 ≈ 54 (strong exploitation)
  
  # Initial client state
  R_init: 1.0
  N_eff_init: 0.0

# ── AGGREGATION ──────────────────────────────────────────────
aggregation:
  delta: 0.90                   # Same δ as selector — unified staleness definition
  gamma: 1.0                    # Same γ as selector — consistent diversity weighting

# ── OT DIVERSITY COMPUTATION ──────────────────────────────────
diversity:
  method: "optimal_transport"
  w_fitzpatrick: 0.40           # w_φ: Fitzpatrick (1D Wasserstein)
  w_modality: 0.35              # w_τ: modality (LP OT with C_mod)
  w_confirmation: 0.25          # w_λ: confirmation (LP OT with C_conf)
  
  # Cost matrix: modality (3×3, order: dermoscopic, close-up, overview)
  cost_matrix_modality:
    - [0.00, 0.30, 0.70]
    - [0.30, 0.00, 0.45]
    - [0.70, 0.45, 0.00]
  
  # Cost matrix: confirmation (4×4, order: histo, serial, consensus, single)
  cost_matrix_confirmation:
    - [0.000, 0.333, 0.667, 1.000]
    - [0.333, 0.000, 0.333, 0.667]
    - [0.667, 0.333, 0.000, 0.333]
    - [1.000, 0.667, 0.333, 0.000]

# ── DATASET ──────────────────────────────────────────────────
dataset:
  csv_path: "data/preprocessed_data.csv"
  images_dir: "data/images/"
  total_samples: 9740
  malignancy_rate: 0.0555
  total_meta_input_dim: 13      # 6+3+4, no anatomic site
  
  metadata_axes:
    - {name: skin_type,   categories: [I, II, III, IV, V, VI],   encoding_dim: 6}
    - {name: image_type,  categories: [dermoscopic, "clinical: close-up", "clinical: overview"], encoding_dim: 3}
    - {name: confirm_type, categories:
        - histopathology
        - "serial imaging showing no change"
        - "single image expert consensus"
        - "single contributor clinical assessment"
      encoding_dim: 4}
  
  split: {train: 0.70, val: 0.15, test: 0.15}
  stratify_by: [label, image_type]

# ── MODEL ────────────────────────────────────────────────────
model:
  backbone: efficientnet-b3
  pretrained: true
  image_size: 224
  img_embedding_dim: 1536
  metadata_mlp:
    input_dim: 13
    hidden_dim: 64
    output_dim: 32
    dropout: 0.20
  fusion_dim: 1568
  fusion_dropout: 0.30
  num_classes: 2

# ── TRAINING ────────────────────────────────────────────────
training:
  optimizer: AdamW
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
  lr_schedule: cosine_annealing
  local_epochs: 5               # E
  batch_size: 32
  grad_clip_norm: 1.0
  reset_optimizer_each_round: true
  loss:
    type: class_weighted_focal
    gamma_f: 2.0
    reduction: mean

# ── CALIBRATION ─────────────────────────────────────────────
calibration:
  enabled: true
  method: temperature_scaling
  calibration_set: validation
  num_bins_ece: 10
  recalibrate_every_n_rounds: 5

# ── COMMUNICATION ────────────────────────────────────────────
communication:
  compressor: zstd
  compression_level_default: 3
  compression_level_low_bw: 9
  compression_level_very_low_bw: 19
  bandwidth_low_mbps: 10.0
  bandwidth_very_low_mbps: 1.0
  transmit_delta_only: true
  mtls_enabled: true

# ── STRAGGLER SIMULATION ──────────────────────────────────────
straggler:
  high:    {clients: [C9, C10], beta_params: [2,3],   lognormal: [4.09, 0.8]}
  moderate:{clients: [C6,C7,C8],beta_params: [7,3],   lognormal: [2.77, 0.5]}
  reliable:{clients: [C1,C2,C3,C4,C5],beta_params:[19,1],lognormal:[1.39,0.3]}

# ── EVALUATION ──────────────────────────────────────────────
evaluation:
  eval_every_n_rounds: 5
  metrics:
    [auroc, auprc, sensitivity, specificity, f1_malignant, balanced_accuracy,
     ece, brier_score, fitzpatrick_fairness_gap, worst_group_accuracy,
     selection_entropy, communication_efficiency]
  alert_thresholds:
    auroc_drop: 0.03
    sensitivity_min: 0.75
    ece_max: 0.05
    fairness_gap_increase: 0.05

# ── MONITORING ──────────────────────────────────────────────
monitoring:
  drift_detection: {enabled: true, ks_alpha: 0.05, chi2_alpha: 0.05, window: 10}
  logging: {operational: "logs/op/", mathematical: "logs/math/", audit: "logs/audit/"}
```

---

## 11. Repository Architecture

```
fedmelanoma/
│
├── server/
│   ├── server.py              # Flower ServerApp; round lifecycle; background asyncio loop
│   ├── aggregation.py         # FedMelanomaStrategy: c_i = (n/n_A)·δ^s·(1+γD_i); Δw aggregation
│   ├── selector.py            # score_i = R_i·(1+γ·D_i/(1+λN_i)); Boltzmann sampling
│   ├── straggler_detector.py  # W_t = T̃_med + k·MAD; quorum trigger; staleness buffer
│   └── calibrator.py          # T* via NLL minimization; ECE computation
│
├── client/
│   ├── client.py              # FedMelanomaClient: Flower NumPyClient wrapper
│   ├── local_trainer.py       # AdamW, cosine LR, E=5 epochs, optimizer reset
│   ├── dataset.py             # ISICDataset: image loading, 13-d one-hot encoding, augmentation
│   └── focal_loss.py          # ClassWeightedFocalLoss: γ_f=2, local α_c
│
├── models/
│   ├── fusion_model.py        # EfficientNet-B3 (1536-d) + MetadataMLP(13→64→32) → 1568-d
│   └── checkpoints/           # global_model_rNNN.npz + calibration_T_rNNN.json
│
├── math/
│   ├── ot_solver.py           # LP-based OT + Sinkhorn; W1 1D Wasserstein
│   ├── cost_matrices.py       # C_mod (3×3), C_conf (4×4), validation utilities
│   ├── diversity_prior.py     # D_i computation pipeline; global centroid maintenance
│   └── selector_math.py       # R_i EMA, N_eff, U_eff, score_i; Boltzmann sampling
│
├── data/
│   ├── preprocessed_data.csv  # 9,740 records: isic_id|label|skin_type|image_type|confirm_type
│   ├── images/
│   └── partition/             # K=10 Non-IID splits (generated at runtime)
│
├── data_prep/
│   ├── partitioner.py         # Softmax-temperature Non-IID (W ∈ ℝ^{10×3})
│   └── encoder.py             # One-hot (13-d); cross-client consistency validation
│
├── evaluation/
│   ├── metrics.py             # AUROC, AUPRC, Sensitivity, ECE, Brier, Δ_fair, WGA, H(p)
│   ├── calibration_eval.py    # Reliability diagram; ECE per confidence bin
│   ├── federated_metrics.py   # Local-global gap; selection entropy; comm. efficiency
│   └── baselines.py           # FedAvg, FedAsync, FedProx (μ=0.01), FedLesScan
│
├── monitoring/
│   ├── logger.py              # Three-tier logging (operational/mathematical/audit)
│   ├── drift_detector.py      # KS + chi-squared; triggers D_i recomputation
│   └── alerts.py              # Threshold-based alerts; Slack/email integration
│
├── gui/
│   ├── app.py                 # Flask inference; calibrated confidence output
│   ├── gradcam.py             # Grad-CAM on EfficientNet-B3 _blocks.25._depthwise_conv
│   └── templates/
│
├── configs/
│   └── fedmelanoma_v4.yaml    # Canonical configuration (Section 10)
│
├── tests/
│   ├── test_ot_solver.py      # LP optimality; Sinkhorn convergence; cost matrix symmetry
│   ├── test_selector_math.py  # R_i bounds; N_eff monotonicity; score range; asymptotic
│   ├── test_boltzmann.py      # Probability sum=1; no-starvation; order-preserving
│   ├── test_aggregation.py    # c_i normalisation; staleness decay correctness
│   ├── test_focal_loss.py     # Loss values; class weight computation
│   └── test_calibration.py    # ECE computation; T* monotonicity
│
├── docker/
│   ├── Dockerfile.server
│   ├── Dockerfile.client
│   └── docker-compose.yml
│
├── requirements.txt
└── README.md
```

---

## 12. References

[1] American Cancer Society, "Cancer Facts & Figures 2023," Atlanta, GA, 2023.

[2] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, "Communication-Efficient Learning of Deep Networks from Decentralized Data," in *Proc. AISTATS*, 2017, pp. 1273–1282. *(FedAvg — foundational FL algorithm; size-proportional aggregation)*

[3] M. Elzohairy, "Mitigation of Stragglers in Serverless Federated Learning (FedLesScan)," M.Sc. Thesis, Technical University of Munich, 2022. *(Reliability-only straggler-aware selection; direct prior work)*

[4] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Smola, and V. Smith, "Federated Optimization in Heterogeneous Networks (FedProx)," in *Proc. MLSys*, 2020. *(Proximal regularization; $E=5$ local epoch recommendation)*

[5] M. Tan and Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," in *Proc. ICML*, 2019, pp. 6105–6114. *(Model backbone; compound scaling)*

[6] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal Loss for Dense Object Detection," in *Proc. ICCV*, 2017, pp. 2980–2988. *(Focal loss; $\gamma_f = 2$ default; class imbalance)*

[7] C. Xie, S. Koyejo, and I. Gupta, "Asynchronous Federated Optimization," in *NeurIPS Optimization Workshop*, 2019. *(FedAsync; staleness-weighted aggregation; convergence under async FL)*

[8] J. Nguyen, K. Malik, H. Zhan, A. Yousefpour, M. Rabbat, M. Malek, and D. Huba, "FEDERATED LEARNING WITH BUFFERED ASYNCHRONOUS AGGREGATION," in *Proc. AISTATS*, 2022. *(FedBuff; server-side buffering for async FL)*

[9] P. Kairouz et al., "Advances and Open Problems in Federated Learning," *Foundations and Trends in Machine Learning*, vol. 14, no. 1–2, pp. 1–210, 2021. *(Comprehensive FL theory; Non-IID analysis; open problems)*

[10] S. Caldas et al., "LEAF: A Benchmark for Federated Settings," in *NeurIPS Workshop on Federated Learning*, 2019. *(Non-IID simulation; temperature-based partitioning)*

[11] C. Villani, *Optimal Transport: Old and New*, Grundlehren der mathematischen Wissenschaften, vol. 338, Springer, 2008. *(Wasserstein distances; Kantorovich duality; theoretical foundation for OT)*

[12] L. V. Kantorovich, "On a Problem of Monge," *Uspekhi Mat. Nauk*, vol. 3, pp. 225–226, 1948. *(Original Kantorovich OT formulation)*

[13] M. Cuturi, "Sinkhorn Distances: Lightspeed Computation of Optimal Transport Distances," in *Proc. NeurIPS*, 2013, pp. 2292–2300. *(Entropy-regularized OT; Sinkhorn algorithm; practical OT computation)*

[14] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On Calibration of Modern Neural Networks," in *Proc. ICML*, 2017, pp. 1321–1330. *(Temperature scaling; ECE metric; overconfidence in deep networks)*

[15] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, "ImageNet: A Large-Scale Hierarchical Image Database," in *Proc. CVPR*, 2009. *(Normalisation statistics; pretrained EfficientNet-B3 initialisation)*

[16] L. Zhu, Z. Liu, and S. Han, "Deep Leakage from Gradients," in *Proc. NeurIPS*, 2019. *(Gradient inversion attacks; motivation for mTLS in FL)*

[17] J. Davis and M. Goadrich, "The Relationship Between Precision-Recall and ROC Curves," in *Proc. ICML*, 2006, pp. 233–240. *(AUPRC theoretical justification for imbalanced classification)*

[18] A. Krause and D. Golovin, "Submodular Function Maximization," in *Tractability: Practical Approaches to Hard Problems*, Cambridge University Press, 2014, ch. 3. *(Submodular diminishing returns; greedy approximation guarantees)*

[19] R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," in *Proc. ICCV*, 2017. *(Grad-CAM clinical interpretability)*

[20] International Skin Imaging Collaboration (ISIC), "ISIC Archive," [Online]. Available: https://www.isic-archive.com. *(Source dataset)*

[21] E. T. Jaynes, "Information Theory and Statistical Mechanics," *Physical Review*, vol. 106, no. 4, pp. 620–630, 1957. *(Maximum entropy principle; theoretical foundation for Boltzmann sampling)*

[22] B. Kveton, Z. Wen, A. Ashkan, and C. Szepesvari, "Tight Regret Bounds for Stochastic Combinatorial Semi-Bandits," in *Proc. AISTATS*, 2015. *(Combinatorial bandit theory; regret bounds for subset selection)*

[23] R. M. Dudley, *Real Analysis and Probability*, 2nd ed., Cambridge University Press, 2002. *(Wasserstein metric; measure theory; EMA convergence analysis)*
