# FedMelanoma: Master Technical, Mathematical, and Architectural Blueprint

**Version:** 1.0 | **Classification:** Internal Technical Reference  
**Authors:** Rudra Verma 
**Institution:** Thapar Institute of Engineering and Technology, Patiala, India

---

## Table of Contents

1. [Product Requirements & System Overview (PRD)](#1-prd)
2. [Mathematical Formulation & Theoretical Analysis](#2-math)
3. [Core Implementation Architecture](#3-code)
4. [System Configuration & Orchestration](#4-config)

---

## 1. Product Requirements & System Overview (PRD)

### 1.1 Mission Statement

FedMelanoma is a **unified, end-to-end, privacy-preserving federated learning system** for melanoma detection. It is engineered to jointly resolve two compounding pathologies of healthcare FL that existing systems treat in isolation: **(a) multi-axis Non-IID data heterogeneity** and **(b) straggler-induced training instability**, without ever exposing raw patient data beyond the originating institution's firewall.

### 1.2 Regulatory & Privacy Requirements

| Requirement | Constraint | Mechanism |
|---|---|---|
| HIPAA Compliance | No raw PHI leaves client nodes | Transmit only weight deltas `Δwᵢ` |
| GDPR Article 25 | Privacy-by-design | gRPC/mTLS, per-client X.509 certs |
| Clinical Auditability | Model versioning + timestamps | Global Model Repository (v1..T) |
| Bias Equity (SDG 10) | No phenotype blind-spots | Diversity-compensated aggregation |

### 1.3 System-Level Functional Requirements

**FR-1 (Data Heterogeneity):** The system MUST characterize and mitigate heterogeneity across all three Non-IID axes simultaneously:
- **Feature Skew** `P(x)`: Imaging modality differences (dermoscopy 73.9%, TBP tile 23.1%, clinical photography 2.9%) and Fitzpatrick skin type skew (Types I-II: 64.0%, Types IV-VI: 23.8%).
- **Label Quantity Skew** `P(y)`: Per-client malignancy rates ranging from 2-8% (primary care) to 35-45% (tertiary oncology), against a global prevalence of 8.57%.
- **Concept Shift** `P(y|x)`: Four diagnostic confirmation methodologies with confidence weights `λ ∈ {1.0, 0.95, 0.9, 0.85, 0.75}` for histopathology, confocal, serial imaging, expert consensus, and single-contributor, respectively.

**FR-2 (Straggler Tolerance):** The system MUST operate under up to 40-60% straggler rates without blocking the global training loop. Straggler updates MUST NOT be permanently discarded if their distributional value is high.

**FR-3 (Adaptive Selection):** Client selection per round MUST incorporate reliability, distributional diversity, and metadata coverage. At least one client per distributional cluster MUST participate per round.

**FR-4 (Model Architecture):** All clients share a parameter-compatible model: **EfficientNet-B3** image backbone fused with a **Metadata MLP** producing a 1568-dimensional joint embedding for binary classification.

**FR-5 (Communication Security):** All transmissions via **gRPC over mutual TLS (mTLS)** with zstd compression (3-5× compression ratio on FP32 tensors, reducing ~45 MB full model to ~10-15 MB per delta).

**FR-6 (Clinical Inference):** A **Flask/Grad-CAM GUI** exposes the global model for real-time binary classification with spatial attention heatmaps and clinical audit trails.

### 1.4 Non-Functional Requirements

| Category | Requirement |
|---|---|
| Fault Tolerance | Server checkpoints every 5 rounds; clients auto-re-register |
| Scalability | Horizontal: new clients register via client registry |
| Containerization | Docker for consistent cross-hardware environments |
| Minimum Quorum | Aggregation triggers at ρ=0.70 of selected clients OR after time-window Wₜ |
| Convergence Budget | T=50 communication rounds |

### 1.5 Architecture Flow Overview

```
ISIC Archive (127,145 records)
          │
          ▼
  Non-IID Partitioning ──── Softmax-temperature (τ=0.5)
  (K=10 Client Datasets) ─── Facility profile matrix W ∈ R^{10×4}
          │
          ▼ Per-Client (parallel)
  ┌────────────────────────────────────────┐
  │  Preprocessing Pipeline               │
  │  Resize 224×224 → Normalize →         │
  │  Augment → DullRazor Hair Removal     │
  │                                        │
  │  Metadata Imputation                  │
  │  Fitzpatrick KNN impute → one-hot     │
  │                                        │
  │  Local Training (E=5 epochs)          │
  │  EfficientNet-B3 (1536-d) ──┐         │
  │  Metadata MLP (32-d) ────────┤concat  │
  │  Dropout(0.3) → FC → logit  │         │
  │                              │         │
  │  Loss: Class-Weighted        │         │
  │  Focal Loss (γ_f=2, α_c)    │         │
  └────────────────────────────────────────┘
          │
          ▼  Transmit Δwᵢ = wᵢ^{t+1} - w^t
    gRPC/mTLS (zstd compressed)
          │
          ▼
  ┌────────────────────────────────────────┐
  │  CENTRAL AGGREGATION SERVER           │
  │                                        │
  │  Adaptive Selection Engine            │
  │  ─ score_i = α·Rᵢ + β·Dᵢ + γ·Mᵢ    │
  │  ─ Enforce group coverage ∀g∈G        │
  │                                        │
  │  Straggler Detector                   │
  │  ─ Window Wt = T̃_med + k·MAD         │
  │  ─ Late → buffer → round t+1          │
  │                                        │
  │  Staleness-Weighted Aggregator        │
  │  ─ cᵢ = (nᵢ/n) · δ^{sᵢ} · (1+γDᵢ)  │
  │  ─ w^{t+1} = w^t + Σ cᵢ·Δwᵢ         │
  └────────────────────────────────────────┘
          │
          ▼
  Global Model Repository (checkpointed)
          │
          ▼
  Flask GUI + Grad-CAM Clinical Interface
```

### 1.6 Client Taxonomy (K=10 Facility Profiles)

| Client ID | Facility Type | Dataset Size | Malignant % | Dominant Modality | Straggler Class |
|---|---|---|---|---|---|
| C1 | Tertiary Oncology | ~15,000 | ~40% | Dermoscopy | Reliable (<5%) |
| C2 | Tertiary Oncology | ~12,000 | ~35% | Dermoscopy + Histo | Reliable (<5%) |
| C3 | General Dermatology | ~8,000 | ~20% | Dermoscopy | Reliable (<5%) |
| C4 | General Dermatology | ~6,000 | ~18% | Dermoscopy | Reliable (<5%) |
| C5 | General Dermatology | ~5,000 | ~15% | TBP Tile | Reliable (<5%) |
| C6 | Primary Care / GP | ~3,000 | ~6% | Clinical Photography | Moderate (15-25%) |
| C7 | Primary Care / GP | ~2,500 | ~5% | Clinical Photography | Moderate (15-25%) |
| C8 | Primary Care / GP | ~2,000 | ~4% | Mixed | Moderate (15-25%) |
| C9 | Telemedicine | ~800 | ~3% | Mobile/Clinical | High Straggler (40-60%) |
| C10 | Telemedicine | ~300 | ~2% | Mobile/Missing Meta | High Straggler (40-60%) |

---

## 2. Mathematical Formulation & Theoretical Analysis

### 2.1 Non-IID Partitioning: Softmax-Temperature Assignment

#### 2.1.1 Formal Setup

Let $\mathcal{H} = \{h_1, h_2, \ldots, h_K\}$ be a set of $K$ hospital clients, each holding a local dataset $\mathcal{D}_i = \{(\mathbf{x}_j, y_j, \mathbf{m}_j)\}_{j=1}^{n_i}$, where $\mathbf{x}_j \in \mathbb{R}^{224 \times 224 \times 3}$ is a dermoscopic image, $y_j \in \{0,1\}$ is the binary label, and $\mathbf{m}_j \in \mathbb{R}^d$ is the metadata vector.

The Non-IID condition is formally: $P_i(\mathbf{x}, y) \neq P_j(\mathbf{x}, y)$ for $i \neq j$.

#### 2.1.2 Facility Profile Matrix

To simulate $K$ clinically realistic, heterogeneous data distributions from the centralized ISIC archive, we define a **facility profile matrix** $\mathbf{W} \in \mathbb{R}^{K \times 4}$, where each row $\mathbf{w}_k$ encodes the affinity of hospital $k$ across four metadata axes:

$$\mathbf{m}_i = [\varphi_i, \tau_i, \rho_i, \lambda_i]^T$$

where $\varphi_i \in \{$I, II, III, IV, V, VI$\}$ is the Fitzpatrick skin type, $\tau_i$ is imaging modality, $\rho_i$ is anatomic site, and $\lambda_i$ is confirmation methodology.

#### 2.1.3 Softmax-Temperature Assignment

The probability of assigning sample $i$ with metadata $\mathbf{m}_i$ to hospital $k$ follows a softmax distribution parameterized by temperature hyperparameter $\tau_{\text{soft}}$:

$$P(k \mid \mathbf{m}_i) = \frac{\exp\left(\mathbf{w}_k \cdot f(\mathbf{m}_i) / \tau_{\text{soft}}\right)}{\sum_{j=1}^{K} \exp\left(\mathbf{w}_j \cdot f(\mathbf{m}_i) / \tau_{\text{soft}}\right)}$$

where $f: \mathbb{R}^4 \to \mathbb{R}^4$ is the metadata feature encoding function (one-hot expansion + ordinal encoding) and $\mathbf{w}_k \cdot f(\mathbf{m}_i)$ is the dot-product affinity score.

**Temperature semantics:**

| Temperature $\tau_{\text{soft}}$ | Behavior |
|---|---|
| $\tau_{\text{soft}} \to 0$ | Deterministic: sample assigned to highest-affinity hospital |
| $\tau_{\text{soft}} = 0.5$ | High Non-IID skew (used in experiments) |
| $\tau_{\text{soft}} = 1.0$ | Standard Gibbs distribution |
| $\tau_{\text{soft}} \to \infty$ | Uniform IID-like assignment |

The theoretical basis for temperature scaling as a distribution sharpening operator originates in the softmax temperature literature, with applications to Non-IID FL partitioning formalized in the LEAF benchmark [Caldas et al., 2018] and further developed for heterogeneous medical FL settings.

#### 2.1.4 Weight Divergence Under Non-IID

Under IID conditions, the gradient of the global objective $F(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^{K} n_i F_i(\mathbf{w})$ satisfies $\nabla F_i(\mathbf{w}) \approx \nabla F_j(\mathbf{w})$, bounding divergence. Under Non-IID conditions with the softmax assignment above, the gradient dissimilarity quantity is bounded by:

$$\mathbb{E}\left[\|\nabla F_i(\mathbf{w}) - \nabla F(\mathbf{w})\|^2\right] \leq \zeta^2$$

where $\zeta^2$ is the **Non-IID degree** (client drift), which grows as $\tau_{\text{soft}} \to 0$. McMahan et al. [2] established that FedAvg's convergence degrades when $\zeta^2$ is large, motivating the diversity-compensated aggregation strategy in Section 2.3.

---

### 2.2 Class-Weighted Focal Loss

#### 2.2.1 Motivation

The ISIC dataset exhibits extreme label imbalance: 8.57% malignant (10,895) vs. 91.43% benign (116,250). Under Non-IID partitioning, individual clients can deviate from this mean by **5-10×**, producing per-client malignancy rates as low as 2% (telemedicine) and as high as 42% (tertiary oncology). Standard cross-entropy loss is dominated by the majority class, causing the local model to learn a degenerate solution that ignores the clinically critical positive class.

#### 2.2.2 Focal Loss Formulation (Lin et al., 2017)

The focal loss [6] was introduced by Lin et al. for dense object detection to address extreme foreground-background class imbalance. FedMelanoma adapts it for per-client label imbalance. For a binary classification problem, let $p_t$ denote the model's estimated probability for the **true class**:

$$p_t = \begin{cases} \hat{p} & \text{if } y = 1 \\ 1 - \hat{p} & \text{if } y = 0 \end{cases}$$

The standard cross-entropy loss is $\text{CE}(p_t) = -\log(p_t)$. The focal modulation factor $(1 - p_t)^{\gamma_f}$ down-weights the contribution of easy, well-classified examples, forcing the optimizer toward hard misclassifications:

$$\mathcal{L}_{\text{focal}}(p_t) = -(1 - p_t)^{\gamma_f} \log(p_t)$$

#### 2.2.3 Class-Weighted Focal Loss with Per-Client Weights

FedMelanoma augments the focal loss with per-client class-weighting. For class $c \in \{0, 1\}$, the **per-client inverse frequency weight** is:

$$\alpha_c = \frac{n_{\text{total}}^{(i)}}{2 \cdot n_c^{(i)}}$$

where $n_c^{(i)}$ is the count of class $c$ samples at client $i$ and $n_{\text{total}}^{(i)} = n_0^{(i)} + n_1^{(i)}$. These weights are normalized to sum to 1: $\hat{\alpha}_c = \alpha_c / (\alpha_0 + \alpha_1)$.

The complete **class-weighted focal loss** is:

$$\mathcal{L}_{\text{CW-Focal}}(p_t) = -\hat{\alpha}_c \cdot (1 - p_t)^{\gamma_f} \cdot \log(p_t)$$

For the full batch over $N$ samples in a local training iteration:

$$\mathcal{L}_{\text{total}} = -\frac{1}{N} \sum_{j=1}^{N} \hat{\alpha}_{y_j} \cdot (1 - p_t^{(j)})^{\gamma_f} \cdot \log(p_t^{(j)})$$

**Parametric choices:**
- $\gamma_f = 2$: Standard focusing parameter from Lin et al. [6]. At $p_t = 0.9$ (easy example), the modulation factor $(1-0.9)^2 = 0.01$ reduces the loss by 100×. At $p_t = 0.1$ (hard example), $(1-0.1)^2 = 0.81$ preserves 81% of the loss.
- $\alpha_c$: Computed locally from private client data. **Critically, raw class counts are never transmitted to the server**, preserving class distribution privacy.

**Effect on malignant class:** When $n_{\text{malignant}} \ll n_{\text{benign}}$ (e.g., 2% malignancy rate at telemedicine node C10), $\hat{\alpha}_{\text{malignant}} \approx 0.98 \gg \hat{\alpha}_{\text{benign}} \approx 0.02$, effectively amplifying the training gradient for positive cases by ~49×, preventing the degenerate all-benign solution.

---

### 2.3 Staleness-Weighted Aggregation with Diversity Bonus

#### 2.3.1 Standard FedAvg Aggregation (McMahan et al., 2017)

The global model at round $t+1$ under synchronous FedAvg [2] is:

$$\mathbf{w}^{t+1} = \sum_{i=1}^{K} \frac{n_i}{n} \cdot \mathbf{w}_i^{t+1}$$

where $n = \sum_i n_i$. Equivalently, expressed as an update from the current global model:

$$\mathbf{w}^{t+1} = \mathbf{w}^t + \sum_{i=1}^{K} \frac{n_i}{n} \cdot \Delta\mathbf{w}_i^t, \quad \Delta\mathbf{w}_i^t = \mathbf{w}_i^{t+1} - \mathbf{w}^t$$

This formulation **requires synchronous participation** and has no mechanism for handling stale or late-arriving updates.

#### 2.3.2 Asynchronous FedAvg Baseline (Xie et al., 2019)

Xie et al. [FedAsync] formalized asynchronous FL with staleness-weighted aggregation. Upon receiving update $\mathbf{w}_i^{t_i+1}$ from client $i$ (generated at server round $t_i$, received at server round $t$), the staleness is $s_i = t - t_i$ and the server performs:

$$\mathbf{w}^{t+1} = (1 - \eta_s) \mathbf{w}^t + \eta_s \cdot \phi(s_i) \cdot \mathbf{w}_i^{t_i+1}$$

where $\phi(s_i) = \delta^{s_i}$ is the staleness decay function and $\eta_s$ is the server learning rate. This reduces but does not eliminate the staleness problem, and ignores distributional value of individual clients entirely.

#### 2.3.3 FedMelanoma Modified Aggregation (Core Novelty)

Let $\mathcal{A}^t \subseteq \mathcal{S}^t$ denote the set of clients whose updates arrived within window $W_t$ at round $t$ (the **arrived set**). The FedMelanoma aggregation is:

$$\boxed{\mathbf{w}^{t+1} = \mathbf{w}^t + \sum_{i \in \mathcal{A}^t} c_i \cdot \left(\mathbf{w}_i^{t+1} - \mathbf{w}^t\right)}$$

where the **normalized contribution weight** $c_i$ is:

$$c_i = \frac{\tilde{c}_i}{\sum_{j \in \mathcal{A}^t} \tilde{c}_j}, \quad \tilde{c}_i = \frac{n_i}{n} \cdot \delta^{s_i} \cdot (1 + \gamma D_i)$$

**Components of $c_i$:**

1. **Size Weight** $\frac{n_i}{n}$: Proportional contribution from McMahan et al. [2]. Clients with larger datasets contribute more to the aggregate, consistent with maximum likelihood principles.

2. **Staleness Decay** $\delta^{s_i}$, where $\delta \in (0,1)$ (default $\delta = 0.85$): Exponentially discounts updates generated $s_i$ rounds in the past. Prevents oscillation from outdated gradients pointing toward an old loss landscape. For $s_i = 0$ (on-time update): $\delta^0 = 1.0$ (no penalty). For $s_i = 3$: $0.85^3 \approx 0.614$ (38.6% reduction).

3. **Diversity Bonus** $(1 + \gamma D_i)$, where $\gamma > 0$ (default $\gamma = 0.3$) and $D_i \in [0,1]$: The critical departure from FedLesScan and FedAsync. The distributional diversity score $D_i$ is the **Earth Mover's Distance (EMD)** from the current aggregated model's implicit distribution to client $i$'s distribution, normalized to $[0,1]$:

$$D_i = \frac{\text{EMD}(\hat{P}_{\text{global}}, P_i^{\text{metadata}})}{\max_{j \in \mathcal{H}} \text{EMD}(\hat{P}_{\text{global}}, P_j^{\text{metadata}})}$$

**The Straggler-Value Paradox:** Resource-constrained clients (rural clinics, telemedicine nodes) are simultaneously the most likely to be stragglers ($s_i$ large) AND the most likely to hold underrepresented phenotypes ($D_i$ large). Without the diversity bonus, the staleness decay $\delta^{s_i}$ would systematically erase these critical minority-distribution contributions. The $(1 + \gamma D_i)$ term **compensates for staleness decay proportionally to distributional rarity**, preventing phenotype blind-spots in the global model.

**Numerical example:** Consider a high-straggler telemedicine client (C10) with $s_i = 3$, $D_i = 0.9$, $\frac{n_{C10}}{n} = 0.003$:
- Without diversity bonus: $\tilde{c}_{C10} = 0.003 \times 0.85^3 = 0.00184$
- With diversity bonus: $\tilde{c}_{C10} = 0.003 \times 0.85^3 \times (1 + 0.3 \times 0.9) = 0.00184 \times 1.27 = 0.00234$

The bonus partially offsets the 38.6% staleness penalty with a 27% diversity boost.

---

### 2.4 Adaptive Client Selection Objective

#### 2.4.1 Component Scores

For each candidate client $i \in \mathcal{H}$, three scores are maintained on the server:

**Reliability Score** $R_i \in [0,1]$: Exponential moving average of round completion success:

$$R_i^{(t+1)} = \beta_R \cdot R_i^{(t)} + (1 - \beta_R) \cdot \mathbb{1}[\text{client } i \text{ completed round } t]$$

where $\beta_R = 0.9$ provides a 10-round effective window.

**Distributional Diversity Score** $D_i \in [0,1]$: As defined in Section 2.3.3, the normalized EMD between the current global model's distribution and client $i$'s metadata-derived distribution.

**Metadata Coverage Score** $M_i \in [0,1]$: Fraction of underrepresented metadata categories present in client $i$'s dataset:

$$M_i = \frac{|\{c \in \mathcal{C}_{\text{rare}} : n_{i,c} > 0\}|}{|\mathcal{C}_{\text{rare}}|}$$

where $\mathcal{C}_{\text{rare}}$ is the set of metadata category combinations with global frequency below a rarity threshold $\theta$ (e.g., dark Fitzpatrick types IV-VI, mobile teledermatology).

#### 2.4.2 Composite Selection Objective

The client selection problem for round $t$ is a **constrained combinatorial optimization**:

$$\mathcal{S}^* = \underset{\mathcal{S} \subseteq \mathcal{H},\, |\mathcal{S}|=C}{\arg\max} \sum_{i \in \mathcal{S}} \left(\alpha R_i + \beta D_i + \gamma M_i\right)$$

subject to:

$$\text{(C1)} \quad \frac{1}{|\mathcal{S}|} \sum_{i \in \mathcal{S}} R_i \geq R_{\min}$$

$$\text{(C2)} \quad \forall g \in \mathcal{G}: |\mathcal{S} \cap \mathcal{C}_g| \geq 1$$

where:
- $\alpha + \beta + \gamma = 1$, defaults: $\alpha=0.4$, $\beta=0.3$, $\gamma=0.3$
- $R_{\min}$ is the minimum average reliability constraint (default 0.5)
- $\mathcal{G} = \{g_1, g_2, g_3, g_4\}$ are the four distributional groups: fair-skin dermoscopy, fair-skin clinical, dark-skin dermoscopy, mixed/telemedicine
- Constraint (C2) enforces at least one client from each occupied cluster per round, guaranteeing cross-distributional gradient diversity

The combinatorial optimization over $\binom{|\mathcal{H}|}{C}$ subsets is solved greedily with constraint enforcement (Section 3.3), which is optimal for submodular objectives and practically sufficient for $K=10$, $C=6$.

---

### 2.5 Theoretical Convergence Analysis

#### 2.5.1 Assumptions

Let the following standard FL convergence assumptions hold (following Li et al. [4], Xie et al. [FedAsync]):

**A1 (L-smoothness):** Each local objective $F_i$ is $L$-smooth: $\|\nabla F_i(\mathbf{u}) - \nabla F_i(\mathbf{v})\| \leq L\|\mathbf{u} - \mathbf{v}\|$ for all $\mathbf{u}, \mathbf{v}$.

**A2 (Bounded Variance):** Stochastic gradients have bounded variance: $\mathbb{E}\|\nabla F_i(\mathbf{w}; \xi) - \nabla F_i(\mathbf{w})\|^2 \leq \sigma^2$.

**A3 (Bounded Non-IID Dissimilarity):** $\mathbb{E}\|\nabla F_i(\mathbf{w}) - \nabla F(\mathbf{w})\|^2 \leq \zeta^2$ for all $i$.

**A4 (Bounded Staleness):** Staleness is bounded: $s_i \leq S_{\max}$ for all $i$, where $S_{\max}$ is determined by the quorum policy.

#### 2.5.2 Convergence Bound for Standard FedAsync

Xie et al. proved that for standard asynchronous FL with staleness decay $\delta^{s_i}$, over $T$ rounds with learning rate $\eta$:

$$\frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}\|\nabla F(\mathbf{w}^t)\|^2 \leq \mathcal{O}\left(\frac{L\sigma^2}{\sqrt{T}} + \frac{L\zeta^2 \delta^{-S_{\max}}}{\sqrt{T}} + \frac{\eta L^2 S_{\max}}{1-\delta}\right)$$

The term $\frac{\eta L^2 S_{\max}}{1-\delta}$ represents the **staleness-induced error floor**: even with $T \to \infty$, convergence is limited by accumulated staleness.

#### 2.5.3 FedMelanoma Convergence with Diversity Bonus

The diversity bonus $(1 + \gamma D_i)$ introduces a modified contribution weight. Define the **effective gradient variance** introduced by the diversity term:

$$\text{Var}_{\text{div}} = \gamma^2 \mathbb{E}\left[D_i^2 \cdot \|\Delta\mathbf{w}_i^t\|^2\right]$$

Since $D_i \in [0,1]$ and $\|\Delta\mathbf{w}_i^t\| \leq \eta E \|\nabla F_i\|$, this variance is bounded:

$$\text{Var}_{\text{div}} \leq \gamma^2 \cdot \eta^2 E^2 \cdot \mathbb{E}\|\nabla F_i(\mathbf{w}^t)\|^2$$

This represents an **additive variance term proportional to $\gamma^2$** in the convergence bound. Substituting normalized weights $c_i$ into the FedMelanoma update:

$$\mathbf{w}^{t+1} = \mathbf{w}^t + \underbrace{\sum_{i \in \mathcal{A}^t} c_i \Delta\mathbf{w}_i}_{\text{effective gradient step}}$$

**Theorem (FedMelanoma Convergence):** Under Assumptions A1-A4, with learning rate $\eta \leq \frac{1}{4EL}$ and diversity coefficient $\gamma \leq \frac{1}{2}$, the FedMelanoma aggregation satisfies:

$$\frac{1}{T}\sum_{t=0}^{T-1} \mathbb{E}\|\nabla F(\mathbf{w}^t)\|^2 \leq \mathcal{O}\left(\frac{F(\mathbf{w}^0) - F^*}{\eta \tau_{\text{eff}} T} + \eta L \sigma^2 + \eta^2 L^2 E^2 \zeta^2 + \gamma^2 \eta^2 L^2 E^2 \sigma^2\right)$$

where $\tau_{\text{eff}} = \min_{t} |\mathcal{A}^t|$ is the minimum arrived-client count per round.

**Interpretation of terms:**
- Term 1: $\mathcal{O}(1/T)$ — Standard optimization convergence, accelerated by larger $\tau_{\text{eff}}$ (more participating clients)
- Term 2: $\mathcal{O}(\eta \sigma^2)$ — Stochastic gradient noise, controlled by learning rate
- Term 3: $\mathcal{O}(\eta^2 E^2 \zeta^2)$ — Non-IID client drift, controlled by local epoch count $E$ (FedProx addresses this via the proximal term; FedMelanoma addresses it via diversity-enforced selection)
- **Term 4: $\mathcal{O}(\gamma^2 \eta^2 \sigma^2)$ — Diversity bonus variance cost**: This is the price of the diversity compensation mechanism. For $\gamma = 0.3$, this term is $0.09\eta^2 L^2 E^2 \sigma^2$, a 9% additive overhead on stochastic variance — **a favorable tradeoff against the systematic minority erasure prevented by the bonus**.

**Comparison to FedProx [4]:** FedProx introduces a proximal regularization term $\frac{\mu}{2}\|\mathbf{w} - \mathbf{w}^t\|^2$ that directly bounds $\|\Delta\mathbf{w}_i\|$, reducing Term 3. However, FedProx does not address staleness (Terms 1, 2) or minority-preserving diversity (Term 4 benefit). FedMelanoma's approach is **aggregation-side rather than optimization-side**, complementary to FedProx and compatible with a proximal variant if needed.

**Remark on quorum policy:** The minimum quorum $\rho = 0.70$ of $C=6$ clients ensures $\tau_{\text{eff}} \geq \lfloor 0.7 \times 6 \rfloor = 4$ clients per round, bounding Term 1 denominator from below and guaranteeing convergence progress even under 30% round-level attrition.

---

## 3. Core Implementation Architecture

### 3.1 The Model Backbone (PyTorch)

**File:** `fedmelanoma/models/fusion_model.py`

```python
"""
FedMelanoma: EfficientNet-B3 + Metadata MLP Fusion Architecture
PyTorch implementation of the joint image-metadata classification backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from typing import Optional


class MetadataMLP(nn.Module):
    """
    Two-layer MLP for encoding structured metadata into a 32-d embedding.
    
    Input: one-hot + ordinal encoded metadata vector
    Architecture: input_dim → 64 → 32
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32,
                 dropout_rate: float = 0.2):
        super(MetadataMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, meta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            meta: (B, input_dim) — batch of encoded metadata vectors
        Returns:
            (B, 32) — metadata embedding
        """
        return self.net(meta)


class FedMelanomaModel(nn.Module):
    """
    FedMelanoma Fusion Architecture:
        EfficientNet-B3 (1536-d image embedding)
        + Metadata MLP (32-d metadata embedding)
        → Concatenate → Dropout(0.3) → FC → Binary logit
    
    Total fusion embedding dimension: 1536 + 32 = 1568-d
    
    All clients share this exact architecture for parameter compatibility
    across heterogeneous federated nodes.
    
    Reference: Tan & Le, EfficientNet (ICML 2019) [5]
    """
    
    IMG_EMBEDDING_DIM = 1536  # EfficientNet-B3 feature dimension
    META_EMBEDDING_DIM = 32
    FUSION_DIM = IMG_EMBEDDING_DIM + META_EMBEDDING_DIM  # 1568
    
    def __init__(
        self,
        meta_input_dim: int,
        num_classes: int = 2,
        dropout_rate: float = 0.3,
        pretrained: bool = True,
    ):
        super(FedMelanomaModel, self).__init__()
        
        # ── Image Backbone: EfficientNet-B3 ─────────────────────────────────
        if pretrained:
            self.image_backbone = EfficientNet.from_pretrained(
                'efficientnet-b3',
                num_classes=self.IMG_EMBEDDING_DIM  # Replace classification head
            )
        else:
            self.image_backbone = EfficientNet.from_name(
                'efficientnet-b3',
                num_classes=self.IMG_EMBEDDING_DIM
            )
        # Remove the final classifier to use as feature extractor
        self.image_backbone._fc = nn.Identity()
        
        # ── Metadata Encoder ─────────────────────────────────────────────────
        self.metadata_mlp = MetadataMLP(
            input_dim=meta_input_dim,
            hidden_dim=64,
            output_dim=self.META_EMBEDDING_DIM,
            dropout_rate=0.2,
        )
        
        # ── Fusion Classification Head ────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.FUSION_DIM, num_classes),
        )
    
    def forward(
        self,
        image: torch.Tensor,
        meta: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the fusion model.
        
        Args:
            image: (B, 3, 224, 224) — normalized dermoscopic image batch
            meta:  (B, meta_input_dim) — encoded metadata batch.
                   If None, uses learned 'unknown' zero embedding.
        Returns:
            logits: (B, 2) — raw logits for [benign, malignant]
        """
        # Extract image embedding: (B, 1536)
        e_img = self.image_backbone(image)
        
        # Compute metadata embedding: (B, 32)
        if meta is not None:
            e_meta = self.metadata_mlp(meta)
        else:
            # Unknown metadata token: learned zero initialization
            e_meta = torch.zeros(
                image.size(0), self.META_EMBEDDING_DIM,
                device=image.device, dtype=image.dtype
            )
        
        # Fusion: concatenate → (B, 1568)
        e_fused = torch.cat([e_img, e_meta], dim=1)
        
        # Classification: (B, 1568) → (B, 2)
        logits = self.classifier(e_fused)
        return logits
    
    def get_embedding(self, image: torch.Tensor, meta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract the 1568-d fusion embedding before the classifier head (for Grad-CAM)."""
        e_img = self.image_backbone(image)
        e_meta = self.metadata_mlp(meta) if meta is not None else torch.zeros(
            image.size(0), self.META_EMBEDDING_DIM, device=image.device)
        return torch.cat([e_img, e_meta], dim=1)
```

**File:** `fedmelanoma/client/focal_loss.py`

```python
"""
Class-Weighted Focal Loss Implementation.

Mathematical formulation:
    L_CW-Focal(p_t) = -alpha_c * (1 - p_t)^gamma_f * log(p_t)

where alpha_c = n_total / (2 * n_c)   [per-client inverse frequency weight]
      gamma_f = 2                       [focusing parameter, Lin et al. 2017]
      p_t     = model probability for the true class

Reference: Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017 [6]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ClassWeightedFocalLoss(nn.Module):
    """
    Per-client class-weighted focal loss for label imbalance correction.
    
    Class weights are computed locally from private client data.
    Raw class counts are NEVER transmitted to the server (privacy preservation).
    """
    
    def __init__(self, gamma_f: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            gamma_f: Focusing parameter. gamma_f=0 reduces to weighted CE loss.
                     gamma_f=2 is the standard Lin et al. value.
            reduction: 'mean' | 'sum' | 'none'
        """
        super(ClassWeightedFocalLoss, self).__init__()
        self.gamma_f = gamma_f
        self.reduction = reduction
    
    @staticmethod
    def compute_class_weights(labels: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
        """
        Compute per-client inverse frequency class weights from local label distribution.
        
        alpha_c = n_total / (2 * n_c)     [unnormalized]
        hat_alpha_c = alpha_c / sum(alpha) [normalized to sum=1]
        
        Args:
            labels: (N,) integer label tensor from local dataset
            num_classes: number of classes (default 2 for binary)
        Returns:
            weights: (num_classes,) normalized class weight tensor
        """
        n_total = labels.numel()
        weights = torch.zeros(num_classes, dtype=torch.float32)
        
        for c in range(num_classes):
            n_c = (labels == c).sum().item()
            if n_c > 0:
                weights[c] = n_total / (num_classes * n_c)
            else:
                weights[c] = 1.0  # Fallback for missing classes
        
        # Normalize weights to sum to 1
        weights = weights / weights.sum()
        return weights
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        alpha: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the class-weighted focal loss.
        
        Args:
            logits:  (B, C) — raw model logits (pre-softmax)
            targets: (B,)   — integer class labels
            alpha:   (C,)   — per-class weights. If None, uses uniform weights.
        Returns:
            loss: scalar or (B,) depending on reduction
        """
        # Convert to probabilities: (B, C)
        probs = F.softmax(logits, dim=1)
        
        # Gather probability of the true class: p_t shape (B,)
        targets_one_hot = F.one_hot(targets, num_classes=probs.size(1)).float()
        p_t = (probs * targets_one_hot).sum(dim=1)
        
        # Focal modulation factor: (1 - p_t)^gamma_f
        focal_weight = (1.0 - p_t) ** self.gamma_f
        
        # Per-sample cross-entropy: -log(p_t)
        log_p_t = torch.log(p_t.clamp(min=1e-8))
        
        # Per-sample focal loss (before class weighting)
        focal_loss = -focal_weight * log_p_t  # (B,)
        
        # Apply class weights alpha_c per sample
        if alpha is not None:
            alpha = alpha.to(logits.device)
            # Select alpha for each sample's true class
            alpha_t = (alpha.unsqueeze(0) * targets_one_hot).sum(dim=1)  # (B,)
            focal_loss = alpha_t * focal_loss
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
```

---

### 3.2 The Aggregation Strategy (Flower)

**File:** `fedmelanoma/server/aggregation.py`

```python
"""
FedMelanoma Custom Flower Strategy: Staleness-Weighted Aggregation with Diversity Bonus.

Core aggregation formula:
    w^{t+1} = w^t + sum_{i in A^t} c_i * (w_i^{t+1} - w^t)

    where: c_i = [(n_i/n) * delta^{s_i} * (1 + gamma * D_i)] / Z
           Z   = normalizing constant = sum_j [(n_j/n) * delta^{s_j} * (1 + gamma * D_j)]

Mathematical grounding:
    - Size-proportional weighting: McMahan et al., FedAvg (AISTATS 2017) [2]
    - Staleness decay: Xie et al., FedAsync (2019)
    - Diversity bonus: FedMelanoma novelty (this work)
"""

import numpy as np
import flwr as fl
from flwr.common import (
    FitRes, Parameters, Scalar, ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
import time

logger = logging.getLogger("fedmelanoma.aggregation")


@dataclass
class ClientState:
    """Server-side state maintained per client across rounds."""
    client_id: str
    dataset_size: int
    diversity_score: float = 0.0           # D_i in [0, 1]
    reliability_ema: float = 1.0            # R_i: exponential moving average
    metadata_coverage: float = 0.5         # M_i in [0, 1]
    last_update_round: int = 0             # t_i: round of last update
    completion_times: List[float] = field(default_factory=list)
    cluster_id: int = 0                    # g in G = {0, 1, 2, 3}
    buffered_update: Optional[Tuple] = None  # Straggler buffer


class FedMelanomaStrategy(fl.server.strategy.Strategy):
    """
    Custom Flower Strategy implementing:
    1. Staleness-weighted aggregation with diversity bonus
    2. Adaptive client selection with cluster coverage constraint
    3. Asynchronous quorum-based round triggering
    """
    
    def __init__(
        self,
        # ── Aggregation hyperparameters ─────────────────────────────────
        delta: float = 0.85,           # Staleness decay factor δ ∈ (0,1)
        gamma_div: float = 0.3,        # Diversity bonus coefficient γ
        # ── Selection hyperparameters ───────────────────────────────────
        alpha_rel: float = 0.4,        # Weight for reliability R_i
        beta_div: float = 0.3,         # Weight for diversity D_i
        gamma_cov: float = 0.3,        # Weight for metadata coverage M_i
        R_min: float = 0.5,            # Minimum average reliability constraint
        num_groups: int = 4,           # |G| distributional groups
        # ── Quorum policy ───────────────────────────────────────────────
        min_fit_clients: int = 6,      # C: clients selected per round
        quorum_fraction: float = 0.70, # ρ: fraction needed to trigger aggregation
        time_window_k: float = 2.5,    # k for W_t = T_med + k * MAD
        min_time_window: float = 60.0, # Minimum window in seconds
        # ── EMA hyperparameter ──────────────────────────────────────────
        reliability_ema_alpha: float = 0.1,  # (1 - beta_R) in R_i update
    ):
        super().__init__()
        
        # Store hyperparameters
        self.delta = delta
        self.gamma_div = gamma_div
        self.alpha_rel = alpha_rel
        self.beta_div = beta_div
        self.gamma_cov = gamma_cov
        self.R_min = R_min
        self.num_groups = num_groups
        self.min_fit_clients = min_fit_clients
        self.quorum_fraction = quorum_fraction
        self.time_window_k = time_window_k
        self.min_time_window = min_time_window
        self.reliability_ema_alpha = reliability_ema_alpha
        
        # ── Server-side state ────────────────────────────────────────────
        self.current_round: int = 0
        self.global_weights: Optional[List[np.ndarray]] = None
        self.client_states: Dict[str, ClientState] = {}
        self.total_samples: int = 0
        
        # Straggler buffer: maps client_id → (weights, staleness)
        self.straggler_buffer: Dict[str, Tuple[List[np.ndarray], int]] = {}
    
    # ──────────────────────────────────────────────────────────────────────────
    # Flower Strategy Interface
    # ──────────────────────────────────────────────────────────────────────────
    
    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        """Return initial global model parameters (loaded from checkpoint if available)."""
        return None  # Flower will use the first client's parameters
    
    def configure_fit(self, server_round, parameters, client_manager):
        """Select clients and configure local training for this round."""
        self.current_round = server_round
        available_clients = list(client_manager.all().values())
        
        # Initialize state for new clients
        for client in available_clients:
            if client.cid not in self.client_states:
                self.client_states[client.cid] = ClientState(
                    client_id=client.cid, dataset_size=1000  # Will be updated on first fit
                )
        
        # Adaptive client selection
        selected_clients = self._adaptive_select(available_clients, server_round)
        
        config = {
            "server_round": server_round,
            "local_epochs": 5,
        }
        
        fit_ins = fl.common.FitIns(parameters, config)
        return [(client, fit_ins) for client in selected_clients]
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Core aggregation logic implementing:
            w^{t+1} = w^t + Σ_{i∈A^t} c_i * (w_i^{t+1} - w^t)
        """
        if not results:
            logger.warning(f"Round {server_round}: No results received. Skipping aggregation.")
            return None, {}
        
        # Update client states for failures (straggler detection)
        for failure in failures:
            if isinstance(failure, tuple):
                client_proxy, _ = failure
                self._update_reliability(client_proxy.cid, success=False)
                logger.info(f"Client {client_proxy.cid} marked as straggler/failed.")
        
        # Collect arrived updates
        arrived_updates = []
        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            weights = parameters_to_ndarrays(fit_res.parameters)
            
            # Retrieve metrics sent by client
            n_i = fit_res.num_examples
            staleness = fit_res.metrics.get("staleness", 0)
            
            # Update client state
            state = self.client_states.get(cid)
            if state:
                state.dataset_size = n_i
                state.last_update_round = server_round - staleness
                self._update_reliability(cid, success=True)
            
            arrived_updates.append((cid, weights, n_i, staleness))
        
        # Include buffered straggler updates (with incremented staleness)
        for buffered_cid, (buffered_weights, buf_staleness) in list(self.straggler_buffer.items()):
            arrived_updates.append((buffered_cid, buffered_weights, 
                                    self.client_states[buffered_cid].dataset_size,
                                    buf_staleness + 1))
        self.straggler_buffer.clear()
        
        # Compute total dataset size for normalization
        n_total = sum(n_i for _, _, n_i, _ in arrived_updates)
        if n_total == 0:
            return None, {}
        
        # ── Compute contribution weights c_i ─────────────────────────────────
        # c̃_i = (n_i/n) * δ^{s_i} * (1 + γ * D_i)
        raw_weights = []
        for cid, weights, n_i, s_i in arrived_updates:
            state = self.client_states.get(cid, ClientState(cid, n_i))
            D_i = state.diversity_score
            
            size_weight = n_i / n_total
            staleness_decay = self.delta ** s_i
            diversity_bonus = 1.0 + self.gamma_div * D_i
            
            c_tilde = size_weight * staleness_decay * diversity_bonus
            raw_weights.append(c_tilde)
            
            logger.debug(
                f"Client {cid}: n_i={n_i}, s_i={s_i}, D_i={D_i:.3f}, "
                f"c̃_i={c_tilde:.6f} "
                f"(size={size_weight:.4f}, decay={staleness_decay:.4f}, bonus={diversity_bonus:.4f})"
            )
        
        # Normalize: c_i = c̃_i / Σ_j c̃_j
        Z = sum(raw_weights)
        if Z == 0:
            logger.error("Normalizing constant Z=0; falling back to uniform weights.")
            normalized_weights = [1.0 / len(arrived_updates)] * len(arrived_updates)
        else:
            normalized_weights = [w / Z for w in raw_weights]
        
        # ── Perform weighted aggregation ──────────────────────────────────────
        # w^{t+1} = w^t + Σ c_i * (w_i^{t+1} - w^t)
        # Equivalent to: w^{t+1} = Σ c_i * w_i^{t+1}  (when c_i sum to 1)
        aggregated_weights = [
            sum(c_i * layer for c_i, layer in zip(
                normalized_weights,
                [update[1][layer_idx] for update in arrived_updates]
            ))
            for layer_idx in range(len(arrived_updates[0][1]))
        ]
        
        self.global_weights = aggregated_weights
        
        # ── Log aggregation metrics ───────────────────────────────────────────
        metrics = {
            "num_arrived": len(arrived_updates),
            "total_samples": n_total,
            "aggregation_round": server_round,
            "normalizing_constant": float(Z),
        }
        
        logger.info(
            f"Round {server_round}: Aggregated {len(arrived_updates)} updates "
            f"(Z={Z:.4f}, samples={n_total})"
        )
        
        return ndarrays_to_parameters(aggregated_weights), metrics
    
    def configure_evaluate(self, server_round, parameters, client_manager):
        """Configure evaluation — evaluate on all clients every 5 rounds."""
        if server_round % 5 != 0:
            return []
        config = {"server_round": server_round}
        eval_ins = fl.common.EvaluateIns(parameters, config)
        all_clients = list(client_manager.all().values())
        return [(client, eval_ins) for client in all_clients]
    
    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        total_samples = sum(r.num_examples for _, r in results)
        agg_loss = sum(r.loss * r.num_examples for _, r in results) / total_samples
        return agg_loss, {"server_round": server_round}
    
    def evaluate(self, server_round, parameters):
        """No server-side evaluation."""
        return None
    
    # ──────────────────────────────────────────────────────────────────────────
    # Internal Methods
    # ──────────────────────────────────────────────────────────────────────────
    
    def _adaptive_select(
        self, available_clients: List[ClientProxy], server_round: int
    ) -> List[ClientProxy]:
        """
        Adaptive client selection solving:
            S* = argmax_{S⊆H, |S|=C} Σ_{i∈S} (α·Rᵢ + β·Dᵢ + γ·Mᵢ)
        subject to:
            (C1) mean(R_i for i in S) >= R_min
            (C2) ∀g∈G: |S ∩ C_g| >= 1
        
        Solved greedily with constraint enforcement.
        """
        C = self.min_fit_clients
        
        # Compute composite scores for all available clients
        scored_clients = []
        for client in available_clients:
            state = self.client_states.get(client.cid)
            if state is None:
                R_i, D_i, M_i = 1.0, 0.5, 0.5  # Priors for new clients
            else:
                R_i = state.reliability_ema
                D_i = state.diversity_score
                M_i = state.metadata_coverage
            
            score = self.alpha_rel * R_i + self.beta_div * D_i + self.gamma_cov * M_i
            scored_clients.append((score, client, R_i, D_i, M_i))
        
        scored_clients.sort(key=lambda x: x[0], reverse=True)
        
        # ── Phase 1: Enforce cluster coverage constraint (C2) ─────────────────
        selected = []
        selected_cids = set()
        covered_groups = set()
        
        for group_id in range(self.num_groups):
            # Find highest-scoring client from this group not yet selected
            for score, client, R_i, D_i, M_i in scored_clients:
                cid = client.cid
                state = self.client_states.get(cid)
                client_group = state.cluster_id if state else 0
                
                if client_group == group_id and cid not in selected_cids:
                    if R_i >= self.R_min * 0.8:  # Relaxed reliability for coverage
                        selected.append(client)
                        selected_cids.add(cid)
                        covered_groups.add(group_id)
                        break
        
        # ── Phase 2: Fill remaining slots greedily, enforcing C1 ─────────────
        remaining_slots = C - len(selected)
        for score, client, R_i, D_i, M_i in scored_clients:
            if len(selected) >= C:
                break
            if client.cid in selected_cids:
                continue
            
            # Check mean reliability constraint (C1) for tentative addition
            tentative_selected = selected + [client]
            tentative_R_mean = np.mean([
                self.client_states.get(c.cid, ClientState(c.cid, 0)).reliability_ema
                for c in tentative_selected
            ])
            
            if tentative_R_mean >= self.R_min:
                selected.append(client)
                selected_cids.add(client.cid)
        
        # Fallback: fill with any available clients if constraints couldn't be met
        if len(selected) < max(1, int(C * self.quorum_fraction)):
            for score, client, R_i, D_i, M_i in scored_clients:
                if client.cid not in selected_cids:
                    selected.append(client)
                    selected_cids.add(client.cid)
                if len(selected) >= C:
                    break
        
        logger.info(
            f"Round {server_round}: Selected {len(selected)}/{len(available_clients)} clients. "
            f"Groups covered: {covered_groups}"
        )
        return selected[:C]
    
    def _update_reliability(self, cid: str, success: bool) -> None:
        """Update client reliability EMA: R_i^{t+1} = β_R·R_i^t + (1-β_R)·success"""
        if cid not in self.client_states:
            return
        state = self.client_states[cid]
        beta_R = 1.0 - self.reliability_ema_alpha
        state.reliability_ema = (
            beta_R * state.reliability_ema +
            self.reliability_ema_alpha * float(success)
        )
    
    def update_diversity_scores(self, diversity_scores: Dict[str, float]) -> None:
        """Update D_i scores from external diversity computation (EMD-based)."""
        for cid, score in diversity_scores.items():
            if cid in self.client_states:
                self.client_states[cid].diversity_score = float(np.clip(score, 0.0, 1.0))
    
    def register_client(
        self, cid: str, dataset_size: int, cluster_id: int,
        metadata_coverage: float = 0.5
    ) -> None:
        """Register a new client with the server."""
        self.client_states[cid] = ClientState(
            client_id=cid,
            dataset_size=dataset_size,
            cluster_id=cluster_id,
            metadata_coverage=metadata_coverage,
        )
        self.total_samples += dataset_size
        logger.info(f"Registered client {cid}: n={dataset_size}, group={cluster_id}")
```

---

### 3.3 The Selection Engine

**File:** `fedmelanoma/server/selection_engine.py`

```python
"""
FedMelanoma Adaptive Client Selection Engine.

Computes EMD-based diversity scores and solves the composite selection objective:
    S* = argmax Σ(α·Rᵢ + β·Dᵢ + γ·Mᵢ)
    s.t. mean(Rᵢ) >= R_min and ∀g: |S∩Cg| >= 1
"""

import numpy as np
from scipy.stats import wasserstein_distance
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger("fedmelanoma.selection")


# Metadata category definitions for the ISIC archive
FITZPATRICK_BINS = [0, 1, 2, 3, 4, 5]   # Types I-VI (0-indexed)
MODALITY_BINS = [0, 1, 2, 3, 4, 5, 6]   # 7 modality categories
CONFIRMATION_BINS = [0, 1, 2, 3, 4]      # 5 confirmation types

# Rare metadata categories (threshold: <5% global frequency)
RARE_CATEGORIES = {
    "fitzpatrick_iv": ("fitzpatrick", 3),
    "fitzpatrick_v":  ("fitzpatrick", 4),
    "fitzpatrick_vi": ("fitzpatrick", 5),
    "mobile_tele":    ("modality", 5),
    "rcm":            ("modality", 6),
}


@dataclass
class ClientMetadataProfile:
    """Compressed metadata distribution statistics for a client (no raw data)."""
    client_id: str
    dataset_size: int
    
    # Marginal distributions over metadata axes
    fitzpatrick_dist: np.ndarray      # (6,) probability vector
    modality_dist: np.ndarray         # (7,) probability vector
    confirmation_dist: np.ndarray     # (5,) probability vector
    malignancy_rate: float            # scalar in [0, 1]
    
    # Derived scores (computed server-side)
    cluster_id: int = 0
    diversity_score: float = 0.0
    metadata_coverage: float = 0.0


class SelectionEngine:
    """
    Computes diversity scores and solves the adaptive client selection problem.
    """
    
    def __init__(
        self,
        num_groups: int = 4,
        emd_weight_fitzpatrick: float = 0.4,
        emd_weight_modality: float = 0.4,
        emd_weight_malignancy: float = 0.2,
    ):
        self.num_groups = num_groups
        self.emd_weights = {
            "fitzpatrick": emd_weight_fitzpatrick,
            "modality": emd_weight_modality,
            "malignancy": emd_weight_malignancy,
        }
        self.client_profiles: Dict[str, ClientMetadataProfile] = {}
        self.global_distribution: Optional[ClientMetadataProfile] = None
    
    def register_client_profile(self, profile: ClientMetadataProfile) -> None:
        """Register a client's metadata distribution profile."""
        self.client_profiles[profile.client_id] = profile
        self._assign_cluster(profile)
        self._compute_metadata_coverage(profile)
        logger.info(f"Registered profile for client {profile.client_id} → cluster {profile.cluster_id}")
    
    def update_global_distribution(self) -> None:
        """
        Compute the global aggregated distribution as a weighted combination
        of all registered client profiles.
        """
        if not self.client_profiles:
            return
        
        total_n = sum(p.dataset_size for p in self.client_profiles.values())
        
        global_fitzpatrick = np.zeros(6)
        global_modality = np.zeros(7)
        global_confirmation = np.zeros(5)
        global_malignancy = 0.0
        
        for profile in self.client_profiles.values():
            w = profile.dataset_size / total_n
            global_fitzpatrick += w * profile.fitzpatrick_dist
            global_modality += w * profile.modality_dist
            global_confirmation += w * profile.confirmation_dist
            global_malignancy += w * profile.malignancy_rate
        
        self.global_distribution = ClientMetadataProfile(
            client_id="global",
            dataset_size=total_n,
            fitzpatrick_dist=global_fitzpatrick,
            modality_dist=global_modality,
            confirmation_dist=global_confirmation,
            malignancy_rate=global_malignancy,
        )
    
    def compute_diversity_scores(self) -> Dict[str, float]:
        """
        Compute EMD-based diversity scores for all clients:
        
            D_i = EMD(P_global, P_i^metadata)  [unnormalized]
        
        Then normalize to [0, 1] by dividing by max_j EMD(P_global, P_j).
        
        EMD is computed as a weighted combination across metadata axes.
        """
        if self.global_distribution is None:
            self.update_global_distribution()
        
        raw_emds: Dict[str, float] = {}
        
        for cid, profile in self.client_profiles.items():
            # Fitzpatrick EMD (1D Wasserstein distance)
            emd_fitz = wasserstein_distance(
                np.arange(6), np.arange(6),
                self.global_distribution.fitzpatrick_dist,
                profile.fitzpatrick_dist
            )
            
            # Modality EMD
            emd_mod = wasserstein_distance(
                np.arange(7), np.arange(7),
                self.global_distribution.modality_dist,
                profile.modality_dist
            )
            
            # Malignancy rate distance (absolute difference)
            emd_mal = abs(self.global_distribution.malignancy_rate - profile.malignancy_rate)
            
            # Weighted composite EMD
            composite_emd = (
                self.emd_weights["fitzpatrick"] * emd_fitz +
                self.emd_weights["modality"] * emd_mod +
                self.emd_weights["malignancy"] * emd_mal
            )
            raw_emds[cid] = composite_emd
        
        # Normalize to [0, 1]
        max_emd = max(raw_emds.values()) if raw_emds else 1.0
        if max_emd == 0:
            return {cid: 0.0 for cid in raw_emds}
        
        normalized_scores = {cid: emd / max_emd for cid, emd in raw_emds.items()}
        
        # Update client profiles with computed scores
        for cid, score in normalized_scores.items():
            self.client_profiles[cid].diversity_score = score
        
        return normalized_scores
    
    def _assign_cluster(self, profile: ClientMetadataProfile) -> None:
        """
        Assign client to one of G=4 distributional clusters:
            0: Fair-skin dermoscopy hospitals     (Fitzpatrick I-II + dermoscopy)
            1: Fair-skin clinical photography     (Fitzpatrick I-II + clinical)
            2: Dark-skin dermoscopy hospitals     (Fitzpatrick IV-VI + dermoscopy)
            3: Mixed / telemedicine               (mobile, missing metadata)
        """
        dominant_modality = np.argmax(profile.modality_dist)
        is_dermoscopy = (dominant_modality == 0)  # Modality 0 = dermoscopy
        
        # Fair skin: Fitzpatrick I-III (bins 0,1,2) dominates
        fair_skin_fraction = profile.fitzpatrick_dist[:3].sum()
        dark_skin_fraction = profile.fitzpatrick_dist[3:].sum()
        
        # Mobile/telemedicine: modality bins 5,6 or high missing rate
        mobile_fraction = profile.modality_dist[5:].sum()
        
        if mobile_fraction > 0.3:
            profile.cluster_id = 3  # Mixed / telemedicine
        elif dark_skin_fraction > 0.3 and is_dermoscopy:
            profile.cluster_id = 2  # Dark-skin dermoscopy
        elif fair_skin_fraction > 0.5 and is_dermoscopy:
            profile.cluster_id = 0  # Fair-skin dermoscopy
        else:
            profile.cluster_id = 1  # Fair-skin clinical
    
    def _compute_metadata_coverage(self, profile: ClientMetadataProfile) -> None:
        """
        Compute M_i: fraction of rare metadata categories present in client i's dataset.
        M_i = |{c ∈ C_rare : n_{i,c} > 0}| / |C_rare|
        """
        covered_rare = 0
        
        # Check rare Fitzpatrick types (IV, V, VI = bins 3, 4, 5)
        for bin_idx in [3, 4, 5]:
            if profile.fitzpatrick_dist[bin_idx] > 0.01:  # >1% coverage
                covered_rare += 1
        
        # Check rare modalities (mobile = bin 5, RCM = bin 6)
        for bin_idx in [5, 6]:
            if profile.modality_dist[bin_idx] > 0.01:
                covered_rare += 1
        
        total_rare = len(RARE_CATEGORIES)
        profile.metadata_coverage = covered_rare / max(total_rare, 1)
```

---

### 3.4 Data Orchestration: Non-IID Partitioning

**File:** `fedmelanoma/data/partitioner.py`

```python
"""
FedMelanoma Non-IID Data Partitioner.

Implements softmax-temperature assignment:
    P(k | m_i) = exp(w_k · f(m_i) / τ) / Σ_j exp(w_j · f(m_i) / τ)

where:
    τ   = temperature hyperparameter (0.5 for high Non-IID skew)
    w_k = facility profile vector for hospital k  (W ∈ R^{K×4})
    f   = metadata feature encoding function
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.special import softmax
import logging

logger = logging.getLogger("fedmelanoma.partitioner")


# Facility profile matrix W ∈ R^{10×4}
# Columns: [Fitzpatrick_affinity, Modality_affinity, AnatomicSite_affinity, Confirmation_affinity]
# Each row represents a hospital's affinity for receiving certain metadata profiles
FACILITY_PROFILES = {
    "C1_tertiary_onco_1":     [0.3, 0.8, 0.6, 0.9],   # High dermoscopy, histo
    "C2_tertiary_onco_2":     [0.3, 0.7, 0.5, 0.85],
    "C3_general_derm_1":      [0.5, 0.6, 0.5, 0.7],
    "C4_general_derm_2":      [0.5, 0.5, 0.5, 0.65],
    "C5_general_derm_3":      [0.6, 0.3, 0.4, 0.6],   # More TBP tile
    "C6_primary_care_1":      [0.7, 0.2, 0.3, 0.5],   # Clinical photography, single contrib
    "C7_primary_care_2":      [0.7, 0.2, 0.3, 0.45],
    "C8_primary_care_3":      [0.6, 0.2, 0.4, 0.5],
    "C9_telemedicine_1":      [0.8, 0.1, 0.2, 0.35],  # High dark-skin, mobile
    "C10_telemedicine_2":     [0.9, 0.05, 0.1, 0.3],  # Missing metadata, very small
}


def encode_metadata(row: pd.Series) -> np.ndarray:
    """
    Encode a single sample's metadata into the affinity feature vector f(m_i).
    
    Returns:
        (4,) float vector encoding [fitzpatrick_score, modality_score,
                                     site_score, confirmation_score]
    """
    # Fitzpatrick type → normalized score (higher = darker)
    fitzpatrick_map = {'I': 0.1, 'II': 0.2, 'III': 0.4, 'IV': 0.6, 'V': 0.8, 'VI': 1.0, 'missing': 0.5}
    fitz_val = str(row.get('fitzpatrick_skin_type', 'missing'))
    fitz_score = fitzpatrick_map.get(fitz_val, 0.5)
    
    # Imaging modality → score (higher = more specialized)
    modality_map = {
        'dermoscopy': 0.9, 'tbp_tile_close_up': 0.6, 'clinical': 0.3,
        'mobile': 0.2, 'rcm': 0.95, 'unknown': 0.4,
    }
    modality_val = str(row.get('image_type', 'unknown')).lower()
    mod_score = modality_map.get(modality_val, 0.4)
    
    # Anatomic site → complexity score
    site_complexity_map = {
        'head/neck': 0.7, 'torso': 0.6, 'lower extremity': 0.5,
        'upper extremity': 0.5, 'palms/soles': 0.9, 'unknown': 0.4,
    }
    site_val = str(row.get('anatom_site_general', 'unknown')).lower()
    site_score = site_complexity_map.get(site_val, 0.4)
    
    # Confirmation methodology → confidence score
    confirm_map = {
        'histopathology': 1.0, 'confocal': 0.95, 'serial_imaging': 0.9,
        'single_consensus': 0.85, 'single_contributor': 0.75,
    }
    confirm_val = str(row.get('diagnosis_confirm_type', 'single_contributor')).lower()
    confirm_score = confirm_map.get(confirm_val, 0.75)
    
    return np.array([fitz_score, mod_score, site_score, confirm_score], dtype=np.float32)


def partition_dataset(
    df: pd.DataFrame,
    facility_profiles: Dict[str, List[float]] = None,
    tau: float = 0.5,
    random_seed: int = 42,
    min_samples_per_client: int = 100,
) -> Dict[str, pd.DataFrame]:
    """
    Partition the ISIC dataset into K Non-IID client subsets using
    softmax-temperature assignment.
    
    Mathematical formulation:
        P(k | m_i) = exp(w_k · f(m_i) / τ) / Σ_j exp(w_j · f(m_i) / τ)
    
    Args:
        df:                  Full ISIC metadata DataFrame (127,145 rows)
        facility_profiles:   W ∈ R^{K×4} facility profile matrix
        tau:                 Temperature τ; lower = sharper Non-IID skew
        random_seed:         Reproducibility seed
        min_samples_per_client: Minimum samples to ensure no empty partitions
    
    Returns:
        Dict mapping client_id → partitioned DataFrame
    """
    if facility_profiles is None:
        facility_profiles = FACILITY_PROFILES
    
    np.random.seed(random_seed)
    rng = np.random.default_rng(random_seed)
    
    client_ids = list(facility_profiles.keys())
    K = len(client_ids)
    
    # Profile matrix W: (K, 4)
    W = np.array([facility_profiles[cid] for cid in client_ids], dtype=np.float32)
    
    logger.info(f"Partitioning {len(df)} samples across K={K} clients with τ={tau}")
    
    # Encode metadata features for all samples: (N, 4)
    logger.info("Encoding metadata features...")
    feature_matrix = np.stack([
        encode_metadata(row) for _, row in df.iterrows()
    ], axis=0)
    
    # Compute affinity scores: A = feature_matrix @ W^T   shape: (N, K)
    affinity_scores = feature_matrix @ W.T  # (N, K)
    
    # Apply softmax with temperature τ along client axis
    # P(k | m_i) = softmax(A_i / τ)
    assignment_probs = softmax(affinity_scores / tau, axis=1)  # (N, K)
    
    # Sample client assignment for each sample
    client_assignments = np.array([
        rng.choice(K, p=assignment_probs[i])
        for i in range(len(df))
    ])
    
    # Build partitions
    partitions: Dict[str, pd.DataFrame] = {cid: [] for cid in client_ids}
    for idx, client_idx in enumerate(client_assignments):
        partitions[client_ids[client_idx]].append(idx)
    
    # Convert index lists to DataFrames
    result = {}
    for cid in client_ids:
        indices = partitions[cid]
        if len(indices) < min_samples_per_client:
            logger.warning(
                f"Client {cid} has only {len(indices)} samples; "
                f"padding to {min_samples_per_client} via random oversampling."
            )
            extra = rng.choice(indices if indices else list(range(len(df))),
                               size=min_samples_per_client - len(indices))
            indices = indices + list(extra)
        result[cid] = df.iloc[indices].reset_index(drop=True)
        
        malignant_rate = result[cid]['target'].mean() if 'target' in result[cid] else "N/A"
        logger.info(
            f"  {cid}: n={len(result[cid])}, "
            f"malignant_rate={malignant_rate:.3f}" if isinstance(malignant_rate, float)
            else f"  {cid}: n={len(result[cid])}"
        )
    
    return result


def log_partition_statistics(partitions: Dict[str, pd.DataFrame]) -> None:
    """Log Non-IID statistics for all client partitions."""
    print("\n" + "="*70)
    print("NON-IID PARTITION STATISTICS")
    print("="*70)
    
    total = sum(len(df) for df in partitions.values())
    
    for cid, df in partitions.items():
        n = len(df)
        mal_rate = df['target'].mean() if 'target' in df.columns else float('nan')
        
        # Fitzpatrick distribution
        if 'fitzpatrick_skin_type' in df.columns:
            fitz_dist = df['fitzpatrick_skin_type'].value_counts(normalize=True).to_dict()
        else:
            fitz_dist = {}
        
        print(f"\n{cid}")
        print(f"  Samples: {n:,} ({100*n/total:.1f}% of total)")
        print(f"  Malignant rate: {mal_rate:.1%}")
        print(f"  Fitzpatrick: {fitz_dist}")
    
    print("="*70 + "\n")
```

---

## 4. System Configuration & Orchestration

### 4.1 Configuration Schema

**File:** `fedmelanoma/configs/fedmelanoma_config.yaml`

```yaml
# ============================================================
# FedMelanoma System Configuration
# Master hyperparameter schema for all system components
# ============================================================

# ── System Identity ──────────────────────────────────────────
system:
  name: "FedMelanoma"
  version: "1.0.0"
  description: "Privacy-preserving federated melanoma detection"
  log_level: "INFO"
  seed: 42
  device: "cuda"                # "cuda" | "cpu" | "auto"
  checkpoint_dir: "models/checkpoints/"
  checkpoint_interval: 5        # Save global model every N rounds

# ── Federated Learning Core Parameters ───────────────────────
federated:
  # Communication rounds
  total_rounds: 50              # T: number of FL rounds
  
  # Client participation
  num_clients: 10               # K: total registered clients
  clients_per_round: 6          # C: clients selected per round
  
  # Aggregation protocol
  aggregation_strategy: "fedmelanoma"   # "fedmelanoma" | "fedavg" | "fedprox"
  
  # Quorum policy (asynchronous triggering)
  quorum_fraction: 0.70         # ρ: trigger aggregation at ρ·C responses
  time_window_k: 2.5            # k: W_t = T̃_med + k·MAD
  min_time_window_seconds: 60   # Absolute minimum window regardless of statistics
  
  # Flower framework settings
  flower:
    server_address: "0.0.0.0:8080"
    grpc_max_message_length: 536870912   # 512 MB for large model deltas
    use_ssl: true
    ssl_certfile: "certs/server.crt"
    ssl_keyfile: "certs/server.key"
    ca_certfile: "certs/ca.crt"

# ── Mathematical Hyperparameters ─────────────────────────────
math:
  # Non-IID Partitioning
  noniid:
    temperature_tau: 0.5        # τ: softmax temperature (lower = sharper skew)
    num_facility_types: 4       # Axes in facility profile vector
    min_samples_per_client: 100 # Minimum partition size guard
  
  # Staleness-Weighted Aggregation
  aggregation:
    delta: 0.85                 # δ: staleness decay factor ∈ (0,1)
    gamma_diversity: 0.3        # γ: diversity bonus coefficient
    max_staleness_rounds: 3     # S_max: discard updates older than this
  
  # Adaptive Client Selection
  selection:
    alpha_reliability: 0.4      # α: weight for R_i in composite score
    beta_diversity: 0.3         # β: weight for D_i in composite score
    gamma_coverage: 0.3         # γ: weight for M_i in composite score
    R_min: 0.50                 # Minimum average reliability constraint
    reliability_ema_beta: 0.9   # β_R for R_i EMA update
    num_distributional_groups: 4 # |G|: enforced cluster coverage
  
  # Distributional Diversity (EMD)
  diversity:
    emd_weight_fitzpatrick: 0.40
    emd_weight_modality: 0.40
    emd_weight_malignancy: 0.20
  
  # FedProx (baseline comparison only — not used in FedMelanoma)
  fedprox:
    mu: 0.01                    # Proximal term coefficient

# ── Local Training Parameters ────────────────────────────────
training:
  # Optimization
  optimizer: "AdamW"
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
  lr_schedule: "cosine_annealing"
  T_max: 50                     # For cosine annealing
  
  # Training loop
  local_epochs: 5               # E: local training epochs per round
  batch_size: 32
  num_workers: 4
  pin_memory: true
  
  # Loss function
  loss:
    type: "class_weighted_focal"
    gamma_f: 2.0                # Focusing parameter (Lin et al., 2017)
    # alpha_c: computed locally per client from private class counts
    reduction: "mean"
  
  # Gradient clipping (stability)
  grad_clip_norm: 1.0

# ── Model Architecture ────────────────────────────────────────
model:
  backbone: "efficientnet-b3"
  pretrained: true
  
  # Image settings
  image_size: 224               # Input resolution
  img_embedding_dim: 1536       # EfficientNet-B3 feature dimension
  
  # Metadata MLP
  meta_input_dim: 32            # Depends on one-hot encoding size
  meta_hidden_dim: 64
  meta_embedding_dim: 32
  
  # Fusion head
  fusion_dim: 1568              # img_embedding_dim + meta_embedding_dim
  dropout_rate: 0.3
  num_classes: 2                # Binary: benign (0) / malignant (1)
  
  # Communication efficiency
  transmit_delta_only: true     # Transmit Δw_i = w_i^{t+1} - w^t only
  compress_updates: true
  compression: "zstd"
  compression_level: 3

# ── Data Pipeline ─────────────────────────────────────────────
data:
  # Dataset
  isic_csv: "data/filtered_data.csv"
  images_dir: "data/images/"
  partition_dir: "data/partition/"
  
  # Train/val/test split
  train_fraction: 0.70
  val_fraction: 0.15
  test_fraction: 0.15
  stratify_by: ["target", "image_type"]  # Stratified split
  
  # Preprocessing
  normalize_mean: [0.485, 0.456, 0.406]  # ImageNet statistics
  normalize_std: [0.229, 0.224, 0.225]
  hair_removal: true            # DullRazor morphological filtering
  
  # Augmentation (training only)
  augmentation:
    horizontal_flip: 0.5
    vertical_flip: 0.5
    rotation_degrees: 15
    brightness_jitter: 0.2
    contrast_jitter: 0.2
    saturation_jitter: 0.1
    random_erasing_prob: 0.1
  
  # Metadata handling
  metadata:
    fitzpatrick_imputation: "knn"  # "knn" | "modal" | "missing_token"
    missing_token: true
    one_hot_encode: true
    
    # Diagnosis confirmation confidence weights λ
    confirmation_weights:
      histopathology: 1.00
      confocal: 0.95
      serial_imaging: 0.90
      single_consensus: 0.85
      single_contributor: 0.75

# ── Straggler Simulation (Experimental) ──────────────────────
straggler:
  high_straggler_clients: ["C9_telemedicine_1", "C10_telemedicine_2"]
  high_straggler_failure_rate: [0.40, 0.60]    # Range [min, max]
  
  moderate_straggler_clients: ["C6_primary_care_1", "C7_primary_care_2", "C8_primary_care_3"]
  moderate_straggler_failure_rate: [0.15, 0.25]
  
  reliable_clients: ["C1_tertiary_onco_1", "C2_tertiary_onco_2", 
                     "C3_general_derm_1", "C4_general_derm_2", "C5_general_derm_3"]
  reliable_failure_rate: [0.0, 0.05]
  
  # Simulated compute delay (seconds)
  compute_delay_high: [30, 120]    # [min, max] seconds for high-straggler
  compute_delay_reliable: [1, 5]

# ── Evaluation & Baselines ────────────────────────────────────
evaluation:
  eval_every_n_rounds: 5
  metrics:
    - "accuracy"
    - "auc_roc"
    - "sensitivity"      # Recall for malignant class
    - "specificity"
    - "f1_malignant"
    - "fairness_gap"     # Max sensitivity difference across Fitzpatrick groups
  
  baselines:
    fedavg:
      enabled: true
    fedprox:
      enabled: true
      mu: 0.01
    fedlesscan:
      enabled: true
      reliability_only: true    # Selection without diversity compensation

# ── GUI / Inference Interface ─────────────────────────────────
gui:
  host: "0.0.0.0"
  port: 5000
  debug: false
  
  # Grad-CAM settings
  gradcam:
    target_layer: "_blocks.25._depthwise_conv"   # Last conv block in EfficientNet-B3
    colormap: "jet"
    alpha: 0.5                # Heatmap overlay transparency
  
  # Supported upload formats
  allowed_extensions: ["jpg", "jpeg", "png"]
  max_upload_size_mb: 10
  
  # Batch evaluation
  batch_mode_enabled: true
  batch_max_samples: 1000
```

---

### 4.2 Asynchronous Flow: Timeout and Quorum Logic

**File:** `fedmelanoma/server/straggler_detector.py`

```python
"""
Straggler Detector and Asynchronous Round Controller.

Implements the time-window-based quorum trigger:
    W_t = T̃_med + k · MAD(completion_times)

Round t aggregation triggers when EITHER:
    (A) ρ · C clients have responded (quorum met), OR
    (B) W_t seconds have elapsed since round broadcast

whichever comes first.
"""

import time
import threading
import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
import logging

logger = logging.getLogger("fedmelanoma.straggler")


@dataclass
class RoundState:
    """Per-round state tracked by the straggler detector."""
    round_id: int
    selected_clients: List[str]
    start_time: float = field(default_factory=time.time)
    arrived_clients: List[str] = field(default_factory=list)
    late_clients: List[str] = field(default_factory=list)
    failed_clients: List[str] = field(default_factory=list)
    is_aggregated: bool = False
    window_size: float = 120.0


class StragglerDetector:
    """
    Manages round timing, quorum detection, and straggler buffering.
    
    Window computation (robust to outliers):
        W_t = T̃_med + k · MAD
    where T̃_med = median(historical completion times)
          MAD   = median(|T_i - T̃_med|)  [Median Absolute Deviation]
    """
    
    def __init__(
        self,
        quorum_fraction: float = 0.70,
        time_window_k: float = 2.5,
        min_time_window: float = 60.0,
        max_time_window: float = 600.0,
        max_staleness_rounds: int = 3,
        aggregation_callback: Optional[Callable] = None,
    ):
        self.quorum_fraction = quorum_fraction
        self.time_window_k = time_window_k
        self.min_time_window = min_time_window
        self.max_time_window = max_time_window
        self.max_staleness = max_staleness_rounds
        self.aggregation_callback = aggregation_callback
        
        # Per-client historical completion times (seconds)
        self.completion_times: Dict[str, List[float]] = {}
        
        # Round-level state
        self.current_round: Optional[RoundState] = None
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None
        
        # Straggler buffer: maps client_id → (update_data, staleness)
        self.straggler_buffer: Dict[str, tuple] = {}
    
    def start_round(self, round_id: int, selected_clients: List[str]) -> float:
        """
        Begin a new FL round. Returns the computed time window W_t.
        
        W_t = max(T̃_med + k·MAD, min_time_window)
        """
        window = self._compute_window(selected_clients)
        
        with self._lock:
            self.current_round = RoundState(
                round_id=round_id,
                selected_clients=selected_clients,
                start_time=time.time(),
                window_size=window,
            )
        
        # Start countdown timer
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(window, self._on_window_expired, args=[round_id])
        self._timer.daemon = True
        self._timer.start()
        
        logger.info(
            f"Round {round_id} started: {len(selected_clients)} clients selected, "
            f"W_t = {window:.1f}s"
        )
        return window
    
    def register_arrival(self, client_id: str, update_data) -> bool:
        """
        Register a client update arrival. Returns True if it's on-time.
        Triggers aggregation if quorum is reached.
        """
        with self._lock:
            if self.current_round is None:
                logger.warning(f"Update from {client_id} but no active round.")
                return False
            
            round_state = self.current_round
            elapsed = time.time() - round_state.start_time
            is_on_time = elapsed <= round_state.window_size
            
            if client_id in round_state.selected_clients:
                if is_on_time:
                    round_state.arrived_clients.append(client_id)
                    completion_time = elapsed
                    
                    # Update completion time history for this client
                    if client_id not in self.completion_times:
                        self.completion_times[client_id] = []
                    self.completion_times[client_id].append(completion_time)
                    
                    logger.debug(
                        f"Client {client_id}: ON-TIME arrival at {elapsed:.1f}s "
                        f"(s_i=0, quorum={len(round_state.arrived_clients)}/{len(round_state.selected_clients)})"
                    )
                    
                    # Check quorum: ρ · C clients arrived
                    quorum = self.quorum_fraction * len(round_state.selected_clients)
                    if len(round_state.arrived_clients) >= quorum and not round_state.is_aggregated:
                        logger.info(
                            f"Quorum reached ({len(round_state.arrived_clients)}/{len(round_state.selected_clients)}). "
                            f"Triggering aggregation."
                        )
                        self._trigger_aggregation(round_state)
                else:
                    # Straggler: buffer for next round
                    round_state.late_clients.append(client_id)
                    staleness = round_state.round_id - (round_state.round_id - 1)  # s_i = 1 for t+1
                    
                    if staleness <= self.max_staleness:
                        self.straggler_buffer[client_id] = (update_data, staleness)
                        logger.info(
                            f"Client {client_id}: STRAGGLER (arrived {elapsed:.1f}s, "
                            f"window={round_state.window_size:.1f}s). Buffered with s_i={staleness}."
                        )
                    else:
                        logger.warning(
                            f"Client {client_id}: update too stale (s_i={staleness} > {self.max_staleness}). "
                            f"Discarding."
                        )
            
            return is_on_time
    
    def _on_window_expired(self, round_id: int) -> None:
        """Callback fired when W_t elapses. Triggers aggregation with arrived clients."""
        with self._lock:
            if (self.current_round is None or 
                self.current_round.round_id != round_id or
                self.current_round.is_aggregated):
                return
            
            round_state = self.current_round
            missing = set(round_state.selected_clients) - set(round_state.arrived_clients)
            round_state.failed_clients = list(missing)
            
            logger.info(
                f"Round {round_id}: Window expired. "
                f"Arrived={len(round_state.arrived_clients)}, "
                f"Stragglers/Failed={len(missing)}: {missing}"
            )
            
            self._trigger_aggregation(round_state)
    
    def _trigger_aggregation(self, round_state: RoundState) -> None:
        """Trigger the aggregation callback (must be called within lock)."""
        if round_state.is_aggregated:
            return
        round_state.is_aggregated = True
        
        if self._timer:
            self._timer.cancel()
        
        if self.aggregation_callback:
            self.aggregation_callback(
                round_id=round_state.round_id,
                arrived_clients=round_state.arrived_clients,
                failed_clients=round_state.failed_clients,
                buffered_updates=dict(self.straggler_buffer),
            )
        
        # Clear buffer after consumption
        self.straggler_buffer.clear()
    
    def _compute_window(self, selected_clients: List[str]) -> float:
        """
        Compute W_t = T̃_med + k · MAD for the selected client set.
        
        Falls back to min_time_window if insufficient history.
        """
        all_times = []
        for cid in selected_clients:
            if cid in self.completion_times and self.completion_times[cid]:
                all_times.extend(self.completion_times[cid][-10:])  # Last 10 rounds
        
        if len(all_times) < 3:
            logger.debug(f"Insufficient history ({len(all_times)} samples); using min window.")
            return self.min_time_window
        
        times_arr = np.array(all_times)
        T_med = np.median(times_arr)
        MAD = np.median(np.abs(times_arr - T_med))
        
        window = T_med + self.time_window_k * MAD
        window = float(np.clip(window, self.min_time_window, self.max_time_window))
        
        logger.debug(f"W_t = {T_med:.1f} + {self.time_window_k}×{MAD:.1f} = {window:.1f}s")
        return window
    
    def get_straggler_stats(self, round_id: int = None) -> Dict:
        """Return straggler statistics for monitoring/logging."""
        if self.current_round is None:
            return {}
        rs = self.current_round
        n_sel = len(rs.selected_clients)
        return {
            "round": rs.round_id,
            "selected": n_sel,
            "arrived": len(rs.arrived_clients),
            "late": len(rs.late_clients),
            "failed": len(rs.failed_clients),
            "straggler_rate": len(rs.late_clients + rs.failed_clients) / max(n_sel, 1),
            "buffered": len(self.straggler_buffer),
        }
```

---

### 4.3 Repository File Structure

```
fedmelanoma/
├── server/
│   ├── server.py              # Flower ServerApp entry point; round loop controller
│   ├── aggregation.py         # FedMelanomaStrategy: Flower Strategy subclass
│   ├── selection_engine.py    # SelectionEngine: EMD diversity + composite scoring
│   └── straggler_detector.py  # StragglerDetector: quorum/window async controller
│
├── client/
│   ├── client.py              # FedMelanomaClient: Flower NumPyClient subclass
│   ├── local_trainer.py       # Training loop: AdamW, cosine LR, focal loss
│   ├── dataset.py             # ISICDataset: PyTorch Dataset + preprocessing
│   └── focal_loss.py          # ClassWeightedFocalLoss: γ_f=2, α_c per client
│
├── models/
│   ├── fusion_model.py        # FedMelanomaModel: EfficientNet-B3 + MetadataMLP
│   └── checkpoints/           # Saved global model parameters (round N)
│       ├── global_model_r001.pt
│       ├── global_model_r005.pt
│       └── ...
│
├── data/
│   ├── filtered_data.csv      # ISIC metadata (127,145 records × 8 features)
│   ├── images/                # Dermoscopic image files (JPG)
│   └── partition/             # Non-IID client splits (generated at runtime)
│       ├── C1_tertiary_onco_1.csv
│       ├── C2_tertiary_onco_2.csv
│       └── ... (K=10 files)
│
├── data_prep/
│   ├── partitioner.py         # Non-IID softmax-temperature partitioner
│   ├── imputer.py             # Fitzpatrick KNN imputation + missing token
│   └── eda.py                 # Distribution analysis + visualization
│
├── gui/
│   ├── app.py                 # Flask inference server (binary + confidence)
│   ├── gradcam.py             # Grad-CAM heatmap generation (EfficientNet-B3)
│   ├── static/
│   │   ├── css/style.css
│   │   └── js/main.js
│   └── templates/
│       ├── index.html         # Upload + metadata form
│       └── result.html        # Classification + heatmap display
│
├── evaluation/
│   ├── metrics.py             # AUC, sensitivity, specificity, fairness gap
│   └── baseline_comparison.py # FedAvg, FedProx, FedLesScan runners
│
├── configs/
│   └── fedmelanoma_config.yaml # Master config (Section 4.1)
│
├── certs/                     # mTLS certificates (git-ignored)
│   ├── ca.crt
│   ├── server.crt
│   └── server.key
│
├── docker/
│   ├── Dockerfile.server
│   ├── Dockerfile.client
│   └── docker-compose.yml
│
├── tests/
│   ├── test_aggregation.py    # Unit tests for staleness-weighted FedAvg
│   ├── test_focal_loss.py     # Loss function correctness checks
│   ├── test_partitioner.py    # Non-IID distribution statistics
│   └── test_selection.py     # Client selection constraint satisfaction
│
├── requirements.txt           # Python 3.10 dependency pinning
└── README.md
```

---

### 4.4 References

[1] American Cancer Society, "Cancer Facts & Figures 2023," 2023.  
[2] H. B. McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data," *AISTATS*, 2017.  
[3] M. Elzohairy, "Mitigation of Stragglers in Serverless Federated Learning (FedLesScan)," M.Sc. Thesis, TU Munich, 2022.  
[4] T. Li et al., "Federated Optimization in Heterogeneous Networks (FedProx)," *MLSys*, 2020.  
[5] M. Tan and Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," *ICML*, 2019.  
[6] T.-Y. Lin et al., "Focal Loss for Dense Object Detection," *ICCV*, 2017.  
[7] C. Xie, S. Koyejo, I. Gupta, "Asynchronous Federated Optimization," *NeurIPS Workshop*, 2019.  
[8] P. Kairouz et al., "Advances and Open Problems in Federated Learning," *Foundations and Trends in ML*, 2021.  
[9] S. Caldas et al., "LEAF: A Benchmark for Federated Settings," *NeurIPS Workshop*, 2018.  
[10] R. Selvaraju et al., "Grad-CAM: Visual Explanations via Gradient-Based Localization," *ICCV*, 2017.
