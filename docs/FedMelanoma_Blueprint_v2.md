# FedMelanoma: Master Technical, Mathematical & Architectural Blueprint
## Version 3.0 — Research-Grade + Production-Grade System Specification

**Dataset-Aligned | OT-Based EMD | Logit Calibration | Complete Evaluation Framework**

> *"The most dangerous failure mode in medical AI is not a model that is wrong for everyone — it is a model that is quietly wrong for the people who need it most."*

---

## Table of Contents

1. [Product Requirements & System Overview (PRD)](#1-prd)
2. [Dataset Specification & Statistical Grounding](#2-dataset)
3. [Mathematical Formulation & Theoretical Analysis](#3-math)
   - 3.1 Non-IID Partitioning: Softmax-Temperature Assignment
   - 3.2 Earth Mover's Distance (EMD): Deep Explanation
   - 3.3 Model Architecture: Step-by-Step Mathematical Derivation
   - 3.4 Class-Weighted Focal Loss: Full Derivation
   - 3.5 Staleness-Weighted Aggregation with Diversity Bonus
   - 3.6 The Staleness (Ageing) Factor: Analysis & Conclusion
   - 3.7 Adaptive Client Selection: Full System
   - 3.8 Theoretical Convergence Analysis
   - **3.9 Cost-Matrix Optimal Transport Formulation** *(New in v3)*
   - **3.10 Logit Calibration & Clinical Reliability** *(New in v3)*
4. [End-to-End Implementation Strategy](#4-implementation)
   - Phase 1: Data Pipeline
   - Phase 2: Client Simulation / Hospital Setup *(Extended in v3)*
   - Phase 3: Training Loop *(Extended in v3)*
   - Phase 4: Communication *(Extended in v3)*
   - Phase 5: Aggregation *(Extended in v3)*
   - Phase 6: Selection Engine *(Extended in v3)*
   - Phase 7: Deployment *(Extended in v3)*
   - Phase 8: Monitoring *(Extended in v3)*
5. **[Evaluation & Validation Framework](#5-eval)** *(New in v3)*
6. **[Comparison with Existing FL Methods](#6-comparison)** *(New in v3)*
7. [System Configuration Schema](#7-config) *(Updated in v3)*
8. [Repository Architecture](#8-repo) *(Updated in v3)*
9. [References](#9-refs) *(Extended in v3)*

---

## 1. Product Requirements & System Overview (PRD)

### 1.1 Mission Statement

FedMelanoma is a **unified, end-to-end, privacy-preserving federated learning system** for melanoma detection from dermoscopic and clinical imaging. It is engineered to jointly resolve two compounding pathologies of healthcare federated learning (FL) that existing systems treat in isolation:

- **(a) Multi-axis Non-IID data heterogeneity** arising from differences in imaging modality, skin phenotype, and diagnostic labeling standards across hospital nodes.
- **(b) Straggler-induced training instability**, wherein resource-constrained clients — which are simultaneously the most clinically valuable — delay or fail to return updates, threatening both convergence and equity.

All of this is achieved **without ever transmitting raw patient data beyond the originating institution's boundary**, satisfying HIPAA and GDPR requirements at the architectural level.

### 1.2 Why This Problem Matters: The Clinical Stakes

Melanoma constitutes roughly 1% of skin cancer diagnoses yet is responsible for the overwhelming majority of skin cancer deaths. Early detection is the single most impactful intervention: five-year survival for localized melanoma exceeds 99%, while metastatic melanoma drops to 27%. Deep learning models trained on diverse, multi-institutional datasets can match or exceed dermatologist-level performance — but assembling such datasets is legally and logistically impossible without privacy-preserving infrastructure.

The challenge deepens because the populations with the highest diagnostic difficulty are precisely those most underrepresented in centralized training pipelines. Darker skin tones (Fitzpatrick IV–VI), resource-limited settings, and non-specialist confirmation methodologies are all systematically undersampled — not due to clinical insignificance, but due to infrastructure inequality. FedMelanoma is designed to correct this asymmetry.

### 1.3 Three Non-IID Axes: Formal Definition

Let $\mathcal{H} = \{h_1, h_2, \ldots, h_K\}$ denote $K$ hospital clients. Each client $h_i$ holds a private dataset $\mathcal{D}_i = \{(\mathbf{x}_j, y_j, \mathbf{m}_j)\}_{j=1}^{n_i}$, where $\mathbf{x}_j$ is a dermoscopic image, $y_j \in \{0,1\}$ is the binary label (benign/malignant), and $\mathbf{m}_j$ is the metadata vector.

The Non-IID condition holds when $P_i(\mathbf{x}, y) \neq P_j(\mathbf{x}, y)$ for $i \neq j$. In FedMelanoma this manifests across three clinically grounded axes — all encoded within the three-dimensional metadata feature space now in use:

**Feature Skew** $P(\mathbf{x})$: Imaging hardware heterogeneity produces statistically distinct pixel-level distributions. The preprocessed dataset contains three modality classes — dermoscopic (84.45%), clinical close-up (14.37%), and clinical overview (1.18%) — each with fundamentally different spatial frequency profiles, color histograms, and artifact patterns. A CNN trained predominantly on dermoscopic images learns subsurface structural features; one trained on clinical overview learns surface-level morphology. These representations are not interchangeable.

**Label Quantity Skew** $P(y)$: Facility type determines the malignant-to-benign ratio encountered by each client. The preprocessed dataset reveals a global malignancy prevalence of **5.55%** (541 malignant out of 9,740 samples). Under Non-IID partitioning, this deviates dramatically: clients with tertiary oncology profiles and high histopathology confirmation receive malignancy rates up to 37.4% (consistent with the 37.38% malignancy rate in histopathology-confirmed samples in the dataset), while primary care clients with single-contributor assessment receive near-zero malignancy rates (0.0% in the dataset for this confirmation type). This 37×+ difference induces **weight divergence**: locally optimized gradients point in conflicting directions.

**Concept Shift** $P(y \mid \mathbf{x})$: The conditional probability of a label given an image is not fixed across institutions. The dataset encodes four confirmation methodologies: histopathology (37.38% malignancy rate), single image expert consensus (0.29%), serial imaging showing no change (0.00%), and single contributor clinical assessment (0.00%). A borderline lesion image may receive opposite labels at two institutions depending on whether it was confirmed by biopsy or by a single clinician's visual assessment — creating irreconcilable label semantic inconsistency.

**Critical dataset observation:** Skin types V and VI show 0.00% malignancy in this dataset. This is not a biological claim — it reflects a data collection artifact (underrepresentation of dark-skin melanoma in standard research archives). It makes these groups *precisely the most important to preserve* in the global model. They represent rare, high-value, potentially unconfirmed cases. Any aggregation strategy that systematically de-weights clients carrying these phenotypes (e.g., pure staleness decay without diversity compensation) would produce a globally biased model.

### 1.4 System-Level Functional Requirements

| Req. | Specification |
|---|---|
| FR-1 | Characterize and mitigate all three Non-IID axes jointly |
| FR-2 | Tolerate 40–60% straggler rates without blocking the training loop |
| FR-3 | Adaptive client selection incorporating $R_i$, $D_i$, $M_i$ with cluster coverage enforcement |
| FR-4 | All clients share parameter-compatible architecture: EfficientNet-B3 + Metadata MLP |
| FR-5 | gRPC/mTLS communication with zstd compression |
| FR-6 | Flask/Grad-CAM GUI for clinical inference |
| FR-7 | Every hospital must be selected at least once within a configurable fairness window $W_{\text{fair}}$ |

### 1.5 Architecture Flow Overview

```
Preprocessed ISIC Dataset (9,740 records)
    Columns: isic_id | label | skin_type | image_type | confirm_type
                │
                ▼
    Non-IID Partitioning
    Softmax-temperature (τ=0.5) over W ∈ ℝ^{K×3}
    Feature axes: φ (Fitzpatrick) | τ_mod (Modality) | λ (Confirmation)
                │
    ┌───────────┴───────────┐
    K=10 Client Partitions (heterogeneous size, class ratio, metadata)
                │
    ▼ Per-Client (parallel, isolated)
    ┌──────────────────────────────────────┐
    │  Image Preprocessing                 │
    │  → Resize 224×224                    │
    │  → ImageNet Normalize                │
    │  → Augment (flip, rotate, jitter)    │
    │  → DullRazor hair removal            │
    │                                      │
    │  Metadata Encoding                   │
    │  → One-hot: 6 + 3 + 4 = 13 dims     │
    │                                      │
    │  Local Training (E=5 epochs)         │
    │  EfficientNet-B3 → 1536-d            │
    │  Metadata MLP (13→64→32)             │
    │  Concatenate → 1568-d                │
    │  Dropout(0.3) → FC → logit           │
    │  Class-Weighted Focal Loss           │
    └──────────────────────────────────────┘
                │
    Transmit Δwᵢ = wᵢ^{t+1} − w^t   (zstd compressed, mTLS)
                │
                ▼
    ┌──────────────────────────────────────┐
    │  CENTRAL AGGREGATION SERVER          │
    │                                      │
    │  Adaptive Selection Engine           │
    │  → score_i = α·Rᵢ + β·Dᵢ + γ·Mᵢ   │
    │  → aging penalty for unvisited       │
    │  → cluster coverage constraint       │
    │                                      │
    │  Straggler Detector                  │
    │  → Wt = T̃_med + k·MAD               │
    │  → Late → buffer with staleness sᵢ   │
    │                                      │
    │  Staleness-Weighted Aggregator       │
    │  → cᵢ = (nᵢ/n)·δ^{sᵢ}·(1+γDᵢ)      │
    │  → w^{t+1} = w^t + Σ cᵢ·Δwᵢ         │
    └──────────────────────────────────────┘
                │
    Global Model (checkpointed every 5 rounds)
                │
                ▼
    Flask GUI + Grad-CAM Clinical Inference Interface
```

---

## 2. Dataset Specification & Statistical Grounding

### 2.1 Column Schema

The preprocessed dataset (`preprocessed_data.csv`) has **9,740 samples** and **5 columns**:

| Column | Type | Description |
|---|---|---|
| `isic_id` | string | Unique ISIC sample identifier |
| `label` | int (0/1) | Binary target: 0 = Benign, 1 = Malignant |
| `skin_type` | string | Fitzpatrick phototype: I, II, III, IV, V, VI |
| `image_type` | string | Imaging modality |
| `confirm_type` | string | Diagnostic confirmation methodology |

**No missing values.** All columns are fully populated, eliminating the need for imputation strategies that were present in the v1 blueprint.

### 2.2 Label Distribution

$$n_{\text{total}} = 9{,}740, \quad n_{\text{malignant}} = 541 \ (5.55\%), \quad n_{\text{benign}} = 9{,}199 \ (94.45\%)$$

The benign-to-malignant ratio is approximately **17:1**, representing extreme label imbalance that directly motivates the class-weighted focal loss formulation.

### 2.3 Feature Distributions

**Fitzpatrick Skin Type:**

| Type | Count | Fraction | Malignancy Rate |
|---|---|---|---|
| I | 2,599 | 26.68% | 2.85% |
| II | 3,549 | 36.44% | 11.44% |
| III | 1,182 | 12.14% | 4.57% |
| IV | 844 | 8.67% | 0.83% |
| V | 815 | 8.37% | **0.00%** |
| VI | 751 | 7.71% | **0.00%** |

The 0.00% malignancy rates for Types V and VI are a critical dataset artifact. These clients carry rare, diverse distributional information and must not be erased by staleness decay.

**Imaging Modality:**

| Modality | Count | Fraction | Malignancy Rate |
|---|---|---|---|
| Dermoscopic | 8,225 | 84.45% | 5.08% |
| Clinical: close-up | 1,400 | 14.37% | 4.29% |
| Clinical: overview | 115 | 1.18% | **54.78%** |

The clinical overview modality has a 54.78% malignancy rate — almost 10× the global average. This is highly informative for tertiary oncology clients.

**Confirmation Type:**

| Confirm Type | Count | Fraction | Malignancy Rate |
|---|---|---|---|
| Single contributor clinical assessment | 4,879 | 50.09% | 0.00% |
| Single image expert consensus | 1,752 | 17.99% | 0.29% |
| Serial imaging showing no change | 1,675 | 17.20% | 0.00% |
| Histopathology | 1,434 | 14.72% | 37.38% |

Histopathology-confirmed samples carry a 37.38% malignancy rate — they are selectively high-confidence, high-malignancy samples. Clients with high histopathology fraction carry the most confident positive labels in the entire system.

### 2.4 The Three Metadata Axes (Updated — Anatomic Site Removed)

FedMelanoma now operates on exactly **three metadata axes**:

$$\mathbf{m}_i = [\varphi_i, \tau_i^{\text{mod}}, \lambda_i]^T$$

where:
- $\varphi_i \in \{\text{I, II, III, IV, V, VI}\}$ — Fitzpatrick skin phototype
- $\tau_i^{\text{mod}} \in \{\text{dermoscopic, clinical:close-up, clinical:overview}\}$ — imaging modality
- $\lambda_i \in \{\text{histopathology, serial imaging, single consensus, single contributor}\}$ — confirmation methodology

**Anatomic site is completely removed from all formulations.** The dataset does not contain this field, and all prior references to $\rho_i$ are eliminated throughout.

### 2.5 Metadata Encoding Dimensions

The one-hot encoding of the metadata vector $\mathbf{m}_i$ produces a 13-dimensional input to the Metadata MLP:

$$f(\mathbf{m}_i) \in \mathbb{R}^{13} = \underbrace{\mathbb{R}^6}_{\text{Fitzpatrick one-hot}} \oplus \underbrace{\mathbb{R}^3}_{\text{modality one-hot}} \oplus \underbrace{\mathbb{R}^4}_{\text{confirmation one-hot}}$$

The metadata MLP therefore has $\text{input\_dim} = 13$.

---

## 3. Mathematical Formulation & Theoretical Analysis

### 3.1 Non-IID Partitioning: Softmax-Temperature Assignment

#### 3.1.1 Facility Profile Matrix (Updated to K×3)

To simulate $K=10$ clinically realistic, heterogeneous distributions from the centralized dataset, we define a **facility profile matrix** $\mathbf{W} \in \mathbb{R}^{K \times 3}$, where each row $\mathbf{w}_k \in \mathbb{R}^3$ encodes the affinity of hospital $k$ across the **three** metadata axes:

$$\mathbf{w}_k = [w_k^{\varphi}, \ w_k^{\tau}, \ w_k^{\lambda}]$$

- $w_k^{\varphi}$: affinity for darker-skin phenotypes (higher = prefers Fitzpatrick IV–VI)
- $w_k^{\tau}$: affinity for specialized imaging (higher = prefers dermoscopy over clinical photography)
- $w_k^{\lambda}$: affinity for high-confidence confirmation (higher = prefers histopathology)

This is a **reduction from K×4 to K×3** compared to v1, reflecting the removal of anatomic site.

#### 3.1.2 Metadata Feature Encoding Function

The metadata encoding function $f: \mathcal{M} \to \mathbb{R}^3$ maps a sample's raw metadata to a three-dimensional scalar affinity vector:

$$f(\mathbf{m}_i) = [f_\varphi(\varphi_i), \ f_\tau(\tau_i^{\text{mod}}), \ f_\lambda(\lambda_i)]^T$$

where:

$$f_\varphi(\varphi_i) = \begin{cases} 0.10 & \text{Type I} \\ 0.20 & \text{Type II} \\ 0.40 & \text{Type III} \\ 0.60 & \text{Type IV} \\ 0.80 & \text{Type V} \\ 1.00 & \text{Type VI} \end{cases}, \quad f_\tau(\tau_i) = \begin{cases} 0.95 & \text{dermoscopic} \\ 0.60 & \text{clinical: close-up} \\ 0.30 & \text{clinical: overview} \end{cases}, \quad f_\lambda(\lambda_i) = \begin{cases} 1.00 & \text{histopathology} \\ 0.90 & \text{serial imaging} \\ 0.85 & \text{expert consensus} \\ 0.75 & \text{single contributor} \end{cases}$$

*Intuition:* $f_\varphi$ is an ordinal scale — darker skin types score higher because tertiary oncology centers with advanced dermoscopy equipment see disproportionately fair-skin patients, while resource-limited telemedicine nodes see more diverse phenotypes. $f_\tau$ reflects imaging modality quality. $f_\lambda$ reflects confirmation confidence, which is empirically validated by the dataset: histopathology-confirmed samples have 37.38% malignancy vs. 0.00% for single-contributor assessment.

#### 3.1.3 Softmax-Temperature Assignment

The probability of assigning sample $i$ with metadata $\mathbf{m}_i$ to hospital $k$ is:

$$\boxed{P(k \mid \mathbf{m}_i) = \frac{\exp\!\left(\mathbf{w}_k \cdot f(\mathbf{m}_i) \;/\; \tau_{\text{soft}}\right)}{\displaystyle\sum_{j=1}^{K} \exp\!\left(\mathbf{w}_j \cdot f(\mathbf{m}_i) \;/\; \tau_{\text{soft}}\right)}}$$

The dot product $\mathbf{w}_k \cdot f(\mathbf{m}_i) = w_k^\varphi \cdot f_\varphi(\varphi_i) + w_k^\tau \cdot f_\tau(\tau_i) + w_k^\lambda \cdot f_\lambda(\lambda_i)$ is a **scalar affinity score**: how well-matched is sample $i$ to hospital $k$'s profile? The softmax converts these scores into a probability distribution over hospitals.

**Temperature $\tau_{\text{soft}}$ controls the degree of Non-IID heterogeneity:**

| $\tau_{\text{soft}}$ | Assignment Behavior | Non-IID Degree |
|---|---|---|
| $\to 0$ | Deterministic: sample always goes to highest-affinity hospital | Maximum skew |
| $0.5$ | Strong preference for top hospital; others receive little | **High skew (used in experiments)** |
| $1.0$ | Standard Gibbs distribution | Moderate skew |
| $\to \infty$ | Uniform random assignment | IID (no skew) |

With $\tau_{\text{soft}} = 0.5$, a dermoscopic histopathology-confirmed sample from a Type II patient has a probability vector strongly concentrated on the tertiary oncology client profiles, while a clinical close-up from a Type V patient is directed primarily to telemedicine and primary care nodes.

#### 3.1.4 Weight Divergence Under Non-IID Conditions

The fundamental problem that Non-IID partitioning creates is **client drift**. When each client $h_i$ minimizes its local objective $F_i(\mathbf{w})$ over $E$ local epochs, its parameters move toward a local optimum that is inconsistent with the global optimum. Formally, the gradient dissimilarity is measured by the quantity $\zeta^2$:

$$\zeta^2 = \mathbb{E}\left[\left\|\nabla F_i(\mathbf{w}) - \nabla F(\mathbf{w})\right\|^2\right]$$

McMahan et al. [2] showed that FedAvg's convergence guarantee breaks down when $\zeta^2$ is large, because the aggregated gradient becomes a poor approximation of the true global gradient. The softmax temperature $\tau_{\text{soft}} = 0.5$ produces large $\zeta^2$ — this is intentional: it tests the system under realistic clinical heterogeneity. FedMelanoma's diversity-compensated aggregation is designed to remain convergent even under this regime.

---

### 3.2 Earth Mover's Distance (EMD): Deep Explanation

#### 3.2.1 What Is EMD, Really?

The **Earth Mover's Distance** (also called the **Wasserstein-1 distance**) is a measure of the "work" required to transform one probability distribution into another. To build the right intuition: imagine two distributions are each represented as a pile of dirt spread across a number line. EMD measures the minimum total effort — where effort equals (amount of dirt moved) × (distance moved) — needed to reshape the first pile into the second.

This is fundamentally different from other divergence measures:

- **KL Divergence** $D_{\text{KL}}(P \| Q) = \sum_x P(x) \log(P(x)/Q(x))$ fails when $Q(x) = 0$ at points where $P(x) > 0$ (it diverges to infinity). In our metadata distributions, some hospital client profiles may have zero probability mass over certain Fitzpatrick types or modalities — KL divergence would be undefined.

- **Total Variation Distance** $d_{\text{TV}}(P,Q) = \frac{1}{2}\sum_x |P(x) - Q(x)|$ treats all category differences as equal. Moving probability mass between adjacent Fitzpatrick types (e.g., Type II → Type III) is treated the same as moving between maximally distant types (Type I → Type VI). This ignores the ordinal structure of the feature space.

- **EMD (Wasserstein-1)** respects the geometry of the underlying space. Moving probability mass from Type II to Type III costs less than moving it from Type I to Type VI, because those categories are closer together on the ordinal scale. This makes EMD the **metrically correct** measure for comparing distributions over structured medical metadata.

#### 3.2.2 Formal Definition

For two discrete probability distributions $P = (p_1, \ldots, p_n)$ and $Q = (q_1, \ldots, q_n)$ over ordered categories $\{c_1, \ldots, c_n\}$ with distances $d(c_i, c_j) = |i - j|$ (assuming unit spacing), the **1D Wasserstein distance** (which equals EMD in 1D) is:

$$\text{EMD}(P, Q) = W_1(P, Q) = \sum_{k=1}^{n-1} \left| \sum_{j=1}^{k} (p_j - q_j) \right|$$

This is the **cumulative distribution function (CDF) area formula**: the EMD between two 1D distributions equals the $L^1$ distance between their CDFs.

For **unordered categorical distributions** (such as confirmation type, where no natural ordering exists), the formula reverts to total variation distance on the best-case coupling. In practice, for unordered categories, we use the 1D Wasserstein distance on an embedding (ordinal scoring) or the simpler total variation distance.

#### 3.2.3 Step-by-Step EMD Computation

**Example: Fitzpatrick distribution comparison**

Consider comparing a global dataset distribution against a hypothetical Tertiary Oncology client (Client A) and a Telemedicine client (Client B):

| Fitzpatrick | Global $P$ | Client A $Q_A$ | Client B $Q_B$ |
|---|---|---|---|
| I | 0.2668 | 0.45 | 0.05 |
| II | 0.3644 | 0.40 | 0.10 |
| III | 0.1214 | 0.12 | 0.15 |
| IV | 0.0867 | 0.02 | 0.20 |
| V | 0.0837 | 0.01 | 0.25 |
| VI | 0.0771 | 0.00 | 0.25 |

**EMD(P, Q_A)** — Tertiary Oncology (skewed toward fair skin):

Step 1: Compute CDFs.

$$\text{CDF}_P = [0.2668, \ 0.6312, \ 0.7526, \ 0.8393, \ 0.9230, \ 1.0000]$$
$$\text{CDF}_{Q_A} = [0.4500, \ 0.8500, \ 0.9700, \ 0.9900, \ 1.0000, \ 1.0000]$$

Step 2: Compute absolute CDF differences at each breakpoint $k \in \{1,\ldots,5\}$:

$$|\text{CDF}_P(1) - \text{CDF}_{Q_A}(1)| = |0.2668 - 0.4500| = 0.1832$$
$$|\text{CDF}_P(2) - \text{CDF}_{Q_A}(2)| = |0.6312 - 0.8500| = 0.2188$$
$$|\text{CDF}_P(3) - \text{CDF}_{Q_A}(3)| = |0.7526 - 0.9700| = 0.2174$$
$$|\text{CDF}_P(4) - \text{CDF}_{Q_A}(4)| = |0.8393 - 0.9900| = 0.1507$$
$$|\text{CDF}_P(5) - \text{CDF}_{Q_A}(5)| = |0.9230 - 1.0000| = 0.0770$$

Step 3: Sum:
$$\text{EMD}(P, Q_A) = 0.1832 + 0.2188 + 0.2174 + 0.1507 + 0.0770 = 0.8471$$

**EMD(P, Q_B)** — Telemedicine (skewed toward dark skin):

$$\text{CDF}_{Q_B} = [0.0500, \ 0.1500, \ 0.3000, \ 0.5000, \ 0.7500, \ 1.0000]$$

$$\text{EMD}(P, Q_B) = |0.2668 - 0.0500| + |0.6312 - 0.1500| + |0.7526 - 0.3000| + |0.8393 - 0.5000| + |0.9230 - 0.7500|$$
$$= 0.2168 + 0.4812 + 0.4526 + 0.3393 + 0.1730 = 1.6629$$

**Interpretation:** Client B (telemedicine, dark-skin skew) has EMD ≈ 1.66 from the global distribution, compared to Client A's 0.85. Client B's distribution is further from the global mean — it is more **distributionally novel**. Under FedMelanoma, Client B receives a higher diversity score $D_i$, meaning its updates are partially shielded from staleness decay. This is the mathematical mechanism by which phenotype erasure is prevented.

#### 3.2.4 EMD for Unordered Categories (Modality, Confirmation Type)

For imaging modality and confirmation type, no natural ordinal structure exists. We use the **total variation distance**, which is equivalent to EMD with unit costs between all distinct categories:

$$\text{EMD}_{\text{unordered}}(P, Q) = \frac{1}{2} \sum_c |P(c) - Q(c)|$$

For example, comparing global confirmation distribution against a histopathology-heavy client:

$$P_{\text{global}} = [0.5009, \ 0.1799, \ 0.1720, \ 0.1472], \quad Q_{\text{histo-heavy}} = [0.05, \ 0.05, \ 0.05, \ 0.85]$$

$$\text{EMD} = \tfrac{1}{2}(|0.5009 - 0.05| + |0.1799 - 0.05| + |0.1720 - 0.05| + |0.1472 - 0.85|) = \tfrac{1}{2}(0.4509 + 0.1299 + 0.1220 + 0.7028) = 0.7028$$

This client carries extremely high distributional novelty in terms of confirmation methodology — correctly reflected as a high EMD.

#### 3.2.5 Combined Diversity Score

The final diversity score $D_i$ for client $i$ integrates EMD across all three metadata axes using configurable weights:

$$D_i^{\text{raw}} = w_\varphi \cdot \text{EMD}(\hat{P}_\varphi^{\text{global}}, P_\varphi^{(i)}) + w_\tau \cdot \text{EMD}(\hat{P}_\tau^{\text{global}}, P_\tau^{(i)}) + w_\lambda \cdot \text{EMD}(\hat{P}_\lambda^{\text{global}}, P_\lambda^{(i)})$$

with weights $w_\varphi = 0.40$, $w_\tau = 0.35$, $w_\lambda = 0.25$ (default), and $w_\varphi + w_\tau + w_\lambda = 1$.

This is then **normalized** to $[0,1]$ by dividing by the maximum raw EMD across all clients:

$$D_i = \frac{D_i^{\text{raw}}}{\displaystyle\max_{j \in \mathcal{H}} D_j^{\text{raw}}}$$

The normalization ensures $D_i \in [0,1]$, making the diversity bonus $(1 + \gamma D_i)$ interpretable: at $D_i = 0$ (client identical to global distribution) the bonus is 1.0 (no effect). At $D_i = 1$ (maximally novel client) the bonus is $1 + \gamma$, e.g., $1.30$ for $\gamma = 0.3$.

---

### 3.3 Model Architecture: Step-by-Step Mathematical Derivation

#### 3.3.1 EfficientNet-B3: Why and How

**Why EfficientNet-B3?** EfficientNet [5] introduced **compound scaling** — the observation that simultaneously scaling the width (number of channels), depth (number of layers), and resolution (input image size) in a fixed ratio yields substantially better accuracy-efficiency tradeoffs than scaling any single dimension alone. The scaling coefficients are determined by a constrained optimization over the width-depth-resolution triple $(\phi_w, \phi_d, \phi_r)$ such that:

$$\text{FLOPS} \propto (2^\phi_d) \cdot (2^\phi_w)^2 \cdot (2^\phi_r)^2$$

EfficientNet-B3 represents a specific point on this Pareto frontier: it uses an input resolution of 300×300 (we resize to 224×224), 26 MBConv blocks, and 12 million parameters — significantly smaller than ResNet-50 (25M) while achieving better accuracy on ImageNet. For dermoscopic imaging, smaller model size is critical because (a) clients train locally on potentially limited GPU hardware, and (b) communication is over compressed weight deltas.

**Feature Extraction:** Internally, EfficientNet-B3 operates through a series of **Mobile Inverted Bottleneck Convolution (MBConv)** blocks, each performing:

1. Pointwise (1×1) convolution to expand channel count
2. Depthwise (k×k) spatial convolution
3. Squeeze-and-Excitation (SE) channel attention
4. Pointwise (1×1) projection back to the bottleneck dimension

The final block is followed by a global average pooling layer that collapses spatial dimensions $(H \times W)$ to a single vector. For EfficientNet-B3, this produces a **1536-dimensional image embedding** $\mathbf{e}_{\text{img}} \in \mathbb{R}^{1536}$:

$$\mathbf{e}_{\text{img}} = \text{GlobalAvgPool}(\text{EfficientNet-B3}_{\text{backbone}}(\mathbf{x})) \in \mathbb{R}^{1536}$$

The pretrained ImageNet weights provide a powerful initialization that captures general visual features (textures, edges, structures). Fine-tuning in the FL setting adapts these features to dermoscopic imaging while preserving transferable low-level representations, which is especially important for clients with small local datasets (as low as 300 samples in our simulated telemedicine nodes).

#### 3.3.2 Metadata MLP: Layer-by-Layer Transformation

The Metadata MLP encodes the 13-dimensional one-hot metadata vector into a semantically rich 32-dimensional embedding:

**Input:** $f(\mathbf{m}_i) \in \mathbb{R}^{13}$ — a sparse binary vector with exactly three 1s (one per axis).

**Layer 1 — Linear projection:**
$$\mathbf{h}_1 = \text{ReLU}(\text{BN}(\mathbf{W}_1 f(\mathbf{m}_i) + \mathbf{b}_1)), \quad \mathbf{W}_1 \in \mathbb{R}^{64 \times 13}, \ \mathbf{b}_1 \in \mathbb{R}^{64}$$

The linear layer $\mathbf{W}_1$ projects the sparse 13-d one-hot space into a dense 64-dimensional continuous representation. This is equivalent to a learned **embedding lookup**: each of the 13 binary input dimensions corresponds to a specific metadata category, and $\mathbf{W}_1$'s columns are learned per-category embedding vectors. Batch normalization $\text{BN}(\cdot)$ normalizes the pre-activation distribution, stabilizing training across heterogeneous client datasets with widely differing metadata distributions. ReLU introduces non-linearity, allowing the MLP to capture interaction effects between metadata dimensions.

**Layer 2 — Dimensionality compression:**
$$\mathbf{e}_{\text{meta}} = \text{ReLU}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2), \quad \mathbf{W}_2 \in \mathbb{R}^{32 \times 64}, \ \mathbf{b}_2 \in \mathbb{R}^{32}$$

The output $\mathbf{e}_{\text{meta}} \in \mathbb{R}^{32}$ is the metadata embedding. **Why 32 dimensions?** The metadata input has 13 dimensions with 3 non-zero entries. A 32-d embedding provides sufficient capacity to encode non-linear interactions between the three axes (e.g., dark skin + clinical overview + histopathology co-occurring together) without dominating the fusion — the metadata embedding is only 32/(1536+32) ≈ 2% of the total fusion vector. This deliberate asymmetry ensures the image signal drives the primary classification while metadata provides a calibrating prior.

#### 3.3.3 Fusion and Logit Computation

The fusion embedding is the concatenation of the image and metadata embeddings:

$$\mathbf{z} = [\mathbf{e}_{\text{img}} \ ; \ \mathbf{e}_{\text{meta}}] \in \mathbb{R}^{1568}$$

This is a **simple concatenation** (no learned gating or attention). Concatenation is preferred over addition or product fusion because it preserves all information from both branches independently, allowing the subsequent linear classifier to learn arbitrary linear combinations of image and metadata features.

**Dropout:** Before the final classifier, a dropout layer with rate $p = 0.3$ is applied:

$$\tilde{\mathbf{z}} = \text{Dropout}(\mathbf{z}, p=0.3)$$

During training, dropout randomly zeros 30% of the 1568 neurons in each forward pass, forcing the classifier to not over-rely on any single feature. This is especially important in the federated setting where small-client overfitting is a significant risk.

**Logit Computation — The Final Linear Layer:**

$$\mathbf{l} = \mathbf{W}_c \tilde{\mathbf{z}} + \mathbf{b}_c, \quad \mathbf{W}_c \in \mathbb{R}^{2 \times 1568}, \ \mathbf{b}_c \in \mathbb{R}^2$$

The output $\mathbf{l} = [l_0, l_1]^T$ are **logits** — raw, unconstrained real-valued scores. 

**What are logits?** The term "logit" comes from the logistic function's inverse. A logit is a real number $l \in (-\infty, +\infty)$ that represents the model's "pre-probability" belief. A large positive $l_1$ indicates strong confidence in malignancy; a large negative $l_1$ indicates strong confidence in benignity. Logits have no direct probabilistic interpretation until transformed.

**Softmax transformation to probabilities:**

$$\hat{p}_c = P(y = c \mid \mathbf{x}, \mathbf{m}) = \frac{\exp(l_c)}{\exp(l_0) + \exp(l_1)}, \quad c \in \{0, 1\}$$

The softmax enforces $\hat{p}_0 + \hat{p}_1 = 1$ and $\hat{p}_c \in (0,1)$, converting logits into a proper probability distribution. The difference $l_1 - l_0$ determines the classification: if $l_1 > l_0$, the sample is classified as malignant.

**Why not work directly with probabilities?** Logits are numerically more stable (no division by small numbers), are the natural output of linear layers, and are the direct input to the cross-entropy and focal loss functions. The loss functions apply log and softmax internally in a numerically stable combined form.

---

### 3.4 Class-Weighted Focal Loss: Full Derivation

#### 3.4.1 Why Standard Cross-Entropy Fails

The standard binary cross-entropy loss for a sample with true class $y$ and predicted probability $\hat{p}$:

$$\mathcal{L}_{\text{CE}} = -[y \log(\hat{p}) + (1-y) \log(1-\hat{p})]$$

has two critical failure modes in our setting:

**Failure 1 — Class imbalance domination:** With a benign:malignant ratio of ~17:1, a model that predicts "benign" for every sample achieves 94.45% accuracy. The gradient signal from benign samples numerically dominates training, pushing the model toward this degenerate solution. The loss from 9,199 benign samples overwhelms the loss from 541 malignant samples.

**Failure 2 — Easy examples dominate gradient:** Even within the malignant class, the majority of samples are "easy" — they are clearly malignant and the model quickly assigns them high probability. Their cross-entropy loss approaches 0, but they contribute the same gradient as hard examples. The model converges without meaningfully learning the hard, ambiguous cases that define clinical difficulty.

#### 3.4.2 Focal Loss Derivation (Lin et al., 2017)

Let $p_t$ denote the model's predicted probability for the **true class**:

$$p_t = \begin{cases} \hat{p}_1 & \text{if } y = 1 \ \text{(malignant)} \\ \hat{p}_0 = 1 - \hat{p}_1 & \text{if } y = 0 \ \text{(benign)} \end{cases}$$

**Step 1 — Standard cross-entropy in terms of $p_t$:**

$$\mathcal{L}_{\text{CE}}(p_t) = -\log(p_t)$$

When $p_t \to 1$ (easy, well-classified example), $\mathcal{L}_{\text{CE}} \to 0$ as expected. When $p_t = 0.5$ (maximally ambiguous), $\mathcal{L}_{\text{CE}} = \log(2) \approx 0.693$.

**Step 2 — Focal modulation:**

$$\mathcal{L}_{\text{focal}}(p_t) = -(1 - p_t)^{\gamma_f} \log(p_t)$$

The factor $(1 - p_t)^{\gamma_f}$ is the **focusing weight**. When the model is confident and correct ($p_t = 0.9$): $(1-0.9)^2 = 0.01$ — the loss is reduced by **100×**. When the model is uncertain ($p_t = 0.5$): $(1-0.5)^2 = 0.25$ — loss is reduced by only 4×. Hard examples ($p_t = 0.1$): $(1-0.1)^2 = 0.81$ — minimal reduction.

This mechanism automatically re-focuses the optimizer on hard, misclassified cases, which in our medical context are the ambiguous borderline lesions that carry the most diagnostic uncertainty.

**Step 3 — Class weighting (per-client, computed locally):**

For client $i$, the per-class inverse frequency weight is:

$$\alpha_c^{(i)} = \frac{n_{\text{total}}^{(i)}}{2 \cdot n_c^{(i)}}$$

where $n_c^{(i)}$ is the count of class $c$ samples at client $i$, and $n_{\text{total}}^{(i)} = n_0^{(i)} + n_1^{(i)}$.

These are normalized: $\hat{\alpha}_c^{(i)} = \alpha_c^{(i)} / (\alpha_0^{(i)} + \alpha_1^{(i)})$.

**Step 4 — Complete class-weighted focal loss:**

$$\boxed{\mathcal{L}_{\text{CW-Focal}}(p_t) = -\hat{\alpha}_c^{(i)} \cdot (1 - p_t)^{\gamma_f} \cdot \log(p_t)}$$

Over a local batch of $N$ samples:

$$\mathcal{L}_{\text{total}}^{(i)} = \frac{1}{N} \sum_{j=1}^{N} \hat{\alpha}_{y_j}^{(i)} \cdot (1 - p_t^{(j)})^{\gamma_f} \cdot \left(-\log(p_t^{(j)})\right)$$

**Privacy note:** $\hat{\alpha}_c^{(i)}$ is computed from local class counts that are **never transmitted to the server**. The server never learns the per-client malignancy rate, preserving class distribution privacy.

**Numerical effect:** Consider a client with 2% malignancy rate ($n_0 = 980$, $n_1 = 20$):
$$\alpha_0 = \frac{1000}{2 \times 980} = 0.510, \quad \alpha_1 = \frac{1000}{2 \times 20} = 25.0$$
$$\hat{\alpha}_0 = \frac{0.510}{0.510 + 25.0} = 0.020, \quad \hat{\alpha}_1 = \frac{25.0}{25.5} = 0.980$$

The malignant class receives $\hat{\alpha}_1 / \hat{\alpha}_0 = 49\times$ the weight of the benign class, ensuring the 20 malignant samples generate comparable gradient signal to the 980 benign samples.

#### 3.4.3 Training Pipeline: Forward and Backward Pass

**Forward Pass** (per batch):

1. **Image path:** Batch $\{\mathbf{x}_j\}_{j=1}^B$ → EfficientNet-B3 → $\{\mathbf{e}_{\text{img}}^{(j)}\}$, each in $\mathbb{R}^{1536}$
2. **Metadata path:** Batch $\{f(\mathbf{m}_j)\}_{j=1}^B$ → Metadata MLP → $\{\mathbf{e}_{\text{meta}}^{(j)}\}$, each in $\mathbb{R}^{32}$
3. **Fusion:** $\mathbf{z}^{(j)} = [\mathbf{e}_{\text{img}}^{(j)} ; \mathbf{e}_{\text{meta}}^{(j)}] \in \mathbb{R}^{1568}$
4. **Dropout + Classification:** $\mathbf{l}^{(j)} = \mathbf{W}_c \cdot \text{Dropout}(\mathbf{z}^{(j)}) + \mathbf{b}_c \in \mathbb{R}^2$
5. **Probabilities:** $\hat{p}_c^{(j)} = \text{softmax}(\mathbf{l}^{(j)})_c$
6. **Loss:** $\mathcal{L} = \frac{1}{B}\sum_{j=1}^{B} \mathcal{L}_{\text{CW-Focal}}(p_t^{(j)})$

**Backward Pass:**

7. **Gradient computation:** PyTorch's autograd computes $\partial \mathcal{L} / \partial \theta$ for all parameters $\theta$ of the model through the chain rule, propagating from the loss through the classifier, dropout, concatenation, EfficientNet backbone, and Metadata MLP.

8. **Gradient clipping:** $\|\nabla_\theta\| = \min(\|\nabla_\theta\|, g_{\max})$, with $g_{\max} = 1.0$. This prevents gradient explosion from rare, high-loss malignant samples.

9. **AdamW update:** For each parameter $\theta$:
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla_\theta\mathcal{L}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla_\theta\mathcal{L})^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon} m_t - \eta \cdot \lambda_{\text{wd}} \cdot \theta_t$$

where $\eta = 10^{-4}$ (learning rate), $\lambda_{\text{wd}} = 10^{-4}$ (weight decay). AdamW's decoupled weight decay is critical: it regularizes weights directly rather than through gradient modification, which is especially important when focal loss gradients are highly variable in magnitude.

10. **Repeat for $E = 5$ local epochs** before computing $\Delta\mathbf{w}_i = \mathbf{w}_i^{t+1} - \mathbf{w}^t$ and transmitting to the server.

---

### 3.5 Staleness-Weighted Aggregation with Diversity Bonus

#### 3.5.1 Standard FedAvg and Its Limitations

Standard FedAvg [2] at round $t+1$:

$$\mathbf{w}^{t+1} = \sum_{i=1}^{K} \frac{n_i}{n} \cdot \mathbf{w}_i^{t+1} = \mathbf{w}^t + \sum_{i=1}^{K} \frac{n_i}{n} \cdot \Delta\mathbf{w}_i$$

This is **synchronous**: the server waits for all selected clients before aggregating. It is also **staleness-blind**: all updates receive the same size-proportional weight regardless of when they were generated.

#### 3.5.2 FedMelanoma Aggregation

Let $\mathcal{A}^t \subseteq \mathcal{S}^t$ be the set of **arrived** clients — those who responded within the time window $W_t$. The FedMelanoma update rule is:

$$\boxed{\mathbf{w}^{t+1} = \mathbf{w}^t + \sum_{i \in \mathcal{A}^t} c_i \cdot \underbrace{\left(\mathbf{w}_i^{t+1} - \mathbf{w}^t\right)}_{\Delta\mathbf{w}_i}}$$

The **normalized contribution weight** $c_i$ is:

$$c_i = \frac{\tilde{c}_i}{\sum_{j \in \mathcal{A}^t} \tilde{c}_j}, \quad \tilde{c}_i = \underbrace{\frac{n_i}{n}}_{\text{size}} \cdot \underbrace{\delta^{s_i}}_{\text{staleness}} \cdot \underbrace{(1 + \gamma D_i)}_{\text{diversity bonus}}$$

where $\delta \in (0,1)$ is the staleness decay factor, $s_i = t - t_i$ is the number of rounds since client $i$'s update was generated, and $D_i \in [0,1]$ is the normalized diversity score.

**The normalization $\sum_{i \in \mathcal{A}^t} c_i = 1$ is critical.** Without normalization, if few clients arrive, the update step size would be small, causing premature slowdown. With normalization, the server always makes a full gradient step sized appropriately for the arrived clients.

---

### 3.6 The Staleness (Ageing) Factor: Analysis & Conclusion

#### 3.6.1 What Is the Staleness Factor?

In asynchronous federated learning, different clients complete their training at different times. A client that responds immediately in round $t$ has **staleness $s_i = 0$**. A client that is buffered and used in round $t+1$ has **staleness $s_i = 1$**. The staleness factor $\delta^{s_i}$ is an exponential decay function that reduces the contribution of stale updates.

Think of it as a **freshness guarantee**: a gradient computed on model version $\mathbf{w}^{t-2}$ (two rounds ago) points toward a loss landscape that has since shifted — it is stale information. If we naively treat it with the same weight as a fresh gradient, we introduce a systematic error in the aggregated update direction.

Mathematically, the staleness error can be bounded. A stale update $\Delta\mathbf{w}_i$ generated at round $t_i$ satisfies:

$$\mathbb{E}\left[\left\|\Delta\mathbf{w}_i^{(t_i)} - \Delta\mathbf{w}_i^{(t)}\right\|^2\right] \leq (s_i)^2 \cdot \eta^2 \cdot L^2 \cdot \sigma^2$$

where $L$ is the loss smoothness constant and $\sigma^2$ is gradient variance. The error grows quadratically in staleness. By weighting stale updates by $\delta^{s_i}$, we reduce their contribution proportionally to this error.

#### 3.6.2 The Trade-Off: Stability vs. Fairness

The staleness factor introduces a fundamental tension:

| Aspect | Small $\delta$ (e.g., 0.5) | Large $\delta$ (e.g., 0.95) |
|---|---|---|
| **Staleness penalty** | Aggressive: $0.5^3 = 12.5\%$ weight at $s=3$ | Mild: $0.95^3 = 85.7\%$ weight at $s=3$ |
| **Convergence stability** | Better: stale gradients barely influence model | Worse: stale updates can cause oscillation |
| **Fairness/Coverage** | Poor: slow clinics are effectively excluded | Good: slow clinics still participate meaningfully |
| **Minority preservation** | Poor: dark-skin, rare phenotype clients discarded | Better: their updates retain influence |

In healthcare FL, this trade-off has direct clinical implications. **A small $\delta$ value is a fairness violation**: it systematically excludes rural clinics and telemedicine nodes — the very clients most likely to carry underrepresented phenotypes. A globally optimized model trained without Type V and VI patients will perform poorly on those patients.

#### 3.6.3 The Straggler-Value Paradox

This is the central insight of FedMelanoma's aggregation design. Let's state it formally:

> **Paradox:** The probability of a client being a straggler (high $s_i$) is positively correlated with the probability of that client carrying underrepresented, high-value distributional information (high $D_i$). Therefore, pure staleness-penalized aggregation is systematically biased against the most distributionally valuable clients.

**Evidence from the dataset:** Clients simulated as telemedicine nodes (high-straggler) are assigned high affinity for Fitzpatrick IV-VI patients — the 0.00% malignancy group that represents a critical blind spot. Without diversity compensation, these clients' updates would be consistently down-weighted, producing a model that has never meaningfully learned from dark-skin patients.

#### 3.6.4 Conclusion: Should We Use the Ageing Factor?

**Yes — but only in combination with the diversity bonus.**

The staleness factor alone ($\delta^{s_i}$) is necessary for **convergence stability**: it prevents stale gradients from derailing training. But applied without the diversity bonus, it produces a **systematically biased global model** that ignores minority phenotypes.

The combined formula $\tilde{c}_i = (n_i/n) \cdot \delta^{s_i} \cdot (1 + \gamma D_i)$ is the correct solution:

- For a **fresh, on-time reliable client** ($s_i = 0$, $D_i = 0.2$): $\tilde{c}_i \propto (n_i/n) \cdot 1.0 \cdot 1.06$
- For a **stale, diverse straggler** ($s_i = 3$, $D_i = 0.9$, $\delta = 0.85$): $\tilde{c}_i \propto (n_i/n) \cdot 0.614 \cdot 1.27$

The diversity bonus $(1.27)$ partially compensates for the staleness penalty $(0.614)$, with the combined effect $(0.779)$ being a moderate down-weight rather than near-zero exclusion. The straggler is not treated as equally fresh, but it is not discarded.

The default parameter $\delta = 0.85$ is empirically calibrated to this dataset: at maximum staleness $S_{\max} = 3$, a pure stale update receives $0.85^3 = 0.614$, and the diversity bonus at $D_i = 1.0$, $\gamma = 0.3$ restores it to $0.614 \times 1.30 = 0.798$ of a fresh same-size client.

---

### 3.7 Adaptive Client Selection: Full System

#### 3.7.1 Step 1: What Is a Client?

In the FedMelanoma architecture, a **client is a hospital node** — a distinct computing unit that holds a private local partition of the ISIC dataset and is capable of training the shared model locally. Each client is characterized by:

- $\mathcal{D}_i$: its local dataset (private, never transmitted)
- $n_i$: local dataset size (transmitted once at registration)
- $\mathbf{m}_i^{\text{dist}}$: its metadata distribution profile (a compact statistical summary, not raw data)
- $R_i$: its reliability score (server-maintained)
- $D_i$: its diversity score (server-maintained, computed from EMD)
- $M_i$: its metadata coverage score (server-maintained)
- $g_i \in \mathcal{G}$: its distributional cluster membership

#### 3.7.2 Step 2: The Selection Process

Prior to each round $t$, the Selection Engine evaluates all registered clients $h_i \in \mathcal{H}$ using a composite score:

$$\text{score}_i = \alpha R_i + \beta D_i + \gamma M_i$$

where $\alpha + \beta + \gamma = 1$, defaults $\alpha = 0.4$, $\beta = 0.3$, $\gamma = 0.3$.

**Component 1 — Reliability $R_i$:**

$$R_i^{(t+1)} = \beta_R \cdot R_i^{(t)} + (1 - \beta_R) \cdot \mathbb{1}[\text{client } i \text{ completed round } t]$$

where $\beta_R = 0.9$. This is an exponential moving average over the client's completion history. A client with consistent failures accumulates a low $R_i$, deprioritizing it in future rounds. A new client starts with $R_i = 1.0$ (optimistic initialization).

**Component 2 — Diversity $D_i$:** The normalized EMD-based score computed as described in Section 3.2. High $D_i$ means the client's data distribution diverges from the current global model's implicit distribution — it carries novel, underrepresented information.

**Component 3 — Metadata Coverage $M_i$:**

$$M_i = \frac{|\{c \in \mathcal{C}_{\text{rare}} : n_{i,c} > \theta_{\min}\}|}{|\mathcal{C}_{\text{rare}}|}$$

where $\mathcal{C}_{\text{rare}}$ is the set of rare metadata category combinations (e.g., Fitzpatrick V+VI, clinical overview modality) with global frequency below a rarity threshold. A client that carries rare Fitzpatrick V patients gets a high $M_i$ even if its overall $D_i$ is moderate.

#### 3.7.3 Step 3: The Aging Factor in Client Selection

The aging/staleness factor extends to the **selection mechanism** as well. A client that has not been selected in many rounds accumulates an **aging penalty boost** — its effective score is increased to prevent permanent exclusion:

$$\text{score}_i^{\text{eff}} = \text{score}_i + \eta_{\text{age}} \cdot (t - t_i^{\text{last\_selected}})$$

where $t_i^{\text{last\_selected}}$ is the last round in which client $i$ was selected, and $\eta_{\text{age}} > 0$ is the aging boost coefficient. This implements a soft version of the fairness constraint in Step 4.

#### 3.7.4 Step 4: The Critical Fairness Constraint — All Clients Must Be Selected

**Why this is required:** In standard FL with pure score-based selection, high-performing reliable clients dominate every round, while low-reliability or low-score clients may never be selected for long stretches. This creates a **selection blindspot**: the global model develops along the gradient of only the best-connected, most well-resourced hospitals.

In a clinical context, this means the global model never receives training signal from dark-skin patients, rural clinics, or less reliable telemedicine nodes — despite these being the populations with the most diagnostic need.

**The fairness window constraint:** Every hospital $h_i \in \mathcal{H}$ must be selected at least once within every $W_{\text{fair}}$ consecutive rounds (default $W_{\text{fair}} = 10$ rounds):

$$\forall i \in \mathcal{H}: \exists t' \in [t - W_{\text{fair}}, t] \text{ s.t. } i \in \mathcal{S}^{t'}$$

**Enforcement mechanisms:**

- **Aging boost** (soft): The $\eta_{\text{age}} \cdot (t - t_i^{\text{last}})$ term in the score naturally pushes unselected clients' scores higher over time, making their selection increasingly likely.

- **Mandatory inclusion** (hard): If a client has not been selected for exactly $W_{\text{fair}} - 1$ rounds, it is forcibly included in the next round's selection set, even if its composite score would otherwise exclude it. This guarantees the window constraint.

- **Cluster coverage constraint** (structural): By requiring at least one client from each distributional cluster $\mathcal{G}$ per round, the system enforces distributional diversity at the group level, even when individual low-score clients are not selected.

**Why this matters mathematically:** Without this constraint, the global model's convergence path becomes biased toward the submanifold of the loss landscape defined by the high-score clients' data distributions. The diversity constraint ensures the optimization trajectory traverses the full distribution of the global objective, not just a high-resource subset.

#### 3.7.5 Step 5: Only Selected Clients Train

This is architecturally important and worth stating explicitly:

1. The server broadcasts the current global model $\mathbf{w}^t$ **only to the selected clients** $\mathcal{S}^t \subseteq \mathcal{H}$.
2. Only clients in $\mathcal{S}^t$ perform local training (run $E$ epochs on their local data).
3. Only clients in $\mathcal{S}^t$ compute weight deltas $\Delta\mathbf{w}_i$ and transmit them.
4. Clients not in $\mathcal{S}^t$ are idle that round — they hold their current model copy but do not update it.

This is not merely an efficiency consideration. It is a **privacy requirement**: unselected clients should not receive model updates that could be used for gradient inversion attacks or model reconstruction. Model parameter transmission is itself a potential privacy vector, and limiting it to selected clients reduces the attack surface.

#### 3.7.6 Step 6: Straggler Handling — Deep Explanation

**What is a straggler?** A straggler is a selected client that fails to return its model update within the agreed time window $W_t$. Straggler behavior arises from:

- **Compute constraints:** Rural clinics may use CPU-only machines or shared hospital servers that are preempted by clinical workflows during high-census hours.
- **Network constraints:** Telemedicine nodes in developing regions operate on unstable or low-bandwidth connections where transmitting ~10-15 MB of compressed model deltas may take 5-10 minutes.
- **Data volume:** Some clients have larger local datasets and require more time for $E$ epochs.

**The time window $W_t$:**

$$W_t = \tilde{T}_{\text{med}} + k \cdot \text{MAD}(T_1, \ldots, T_n)$$

where $\tilde{T}_{\text{med}}$ is the median historical completion time across selected clients, $\text{MAD} = \text{median}_i(|T_i - \tilde{T}_{\text{med}}|)$ is the median absolute deviation (a robust spread estimator that ignores extreme outlier straggler times), and $k = 2.5$ is the coverage multiplier. This formula is **robust to outliers**: a single extreme straggler inflating the mean does not inflate the window, because MAD is not affected by outliers beyond the median.

**Straggler state machine:**

```
Client selected for round t
        │
        ▼
   Local training begins
        │
        ├── Update arrives within W_t
        │         → Status: ON-TIME
        │         → Included in round t aggregation with s_i = 0
        │
        ├── Update arrives after W_t but before round t+2
        │         → Status: STRAGGLER
        │         → Buffered with staleness s_i = 1
        │         → Included in round t+1 aggregation with weight c_i × δ^1 × (1 + γD_i)
        │         → Reliability R_i updated negatively
        │
        └── No update by round t+2
                  → Status: FAILED
                  → Buffered update discarded
                  → Reliability R_i penalized significantly
```

**The Straggler-Value Paradox in Practice:**

Consider two clients in our simulated K=10 setup:

- **Client C5** (tertiary oncology, reliable, $s_i = 0$, $D_i = 0.15$, $n_i = 8000$): $\tilde{c}_5 = (8000/n) \times 1.0 \times 1.045$
- **Client C10** (telemedicine, straggler, $s_i = 2$, $D_i = 0.95$, $n_i = 300$): $\tilde{c}_{10} = (300/n) \times 0.723 \times 1.285$

Without diversity bonus, Client C10's weight would be $(300/n) \times 0.723 = 0.217 \times (300/n)$. With the bonus, it becomes $0.279 \times (300/n)$ — a **28.6% recovery** of information that would otherwise be lost to the staleness penalty.

---

### 3.8 Theoretical Convergence Analysis

#### 3.8.1 Standard Assumptions

**A1 (L-smoothness):** $\|\nabla F_i(\mathbf{u}) - \nabla F_i(\mathbf{v})\| \leq L\|\mathbf{u} - \mathbf{v}\|$ for all $\mathbf{u}, \mathbf{v}$, all $i$.

**A2 (Bounded Variance):** $\mathbb{E}_\xi\|\nabla F_i(\mathbf{w}; \xi) - \nabla F_i(\mathbf{w})\|^2 \leq \sigma^2$.

**A3 (Bounded Non-IID Dissimilarity):** $\mathbb{E}\|\nabla F_i(\mathbf{w}) - \nabla F(\mathbf{w})\|^2 \leq \zeta^2$.

**A4 (Bounded Staleness):** $s_i \leq S_{\max}$ for all arrived clients (enforced by the quorum/discard policy).

#### 3.8.2 FedMelanoma Convergence Theorem

Under Assumptions A1–A4, with learning rate $\eta \leq \frac{1}{4EL}$ and diversity coefficient $\gamma \leq \frac{1}{2}$, the FedMelanoma aggregation satisfies:

$$\frac{1}{T}\sum_{t=0}^{T-1}\mathbb{E}\|\nabla F(\mathbf{w}^t)\|^2 \leq \underbrace{\mathcal{O}\!\left(\frac{F(\mathbf{w}^0) - F^*}{\eta \tau_{\text{eff}} T}\right)}_{\text{Term 1: convergence rate}} + \underbrace{\mathcal{O}(\eta L \sigma^2)}_{\text{Term 2: stochastic noise}} + \underbrace{\mathcal{O}(\eta^2 L^2 E^2 \zeta^2)}_{\text{Term 3: Non-IID drift}} + \underbrace{\mathcal{O}(\gamma^2 \eta^2 L^2 E^2 \sigma^2)}_{\text{Term 4: diversity overhead}}$$

where $\tau_{\text{eff}} = \min_t |\mathcal{A}^t| \geq \lfloor \rho C \rfloor = \lfloor 0.7 \times 6 \rfloor = 4$ is the minimum arrived-client count guaranteed by the quorum policy.

**Term-by-term interpretation:**

- **Term 1** ($\mathcal{O}(1/T)$): Convergence toward a stationary point of $F$ at rate $1/T$. Accelerated by $\tau_{\text{eff}}$: more arriving clients per round means faster convergence. The quorum policy lower-bounds $\tau_{\text{eff}} \geq 4$, ensuring progress even under 40% attrition.

- **Term 2** ($\mathcal{O}(\eta \sigma^2)$): Irreducible stochastic gradient noise, controlled by the learning rate. With cosine annealing, $\eta \to 0$ over $T$ rounds, asymptotically eliminating this term.

- **Term 3** ($\mathcal{O}(\eta^2 E^2 \zeta^2)$): Non-IID client drift, growing with local epochs $E$ and Non-IID degree $\zeta^2$. FedMelanoma addresses this via the diversity-enforced selection (ensuring diverse gradients per round), while FedProx addresses it via the proximal regularization term. The two approaches are complementary: FedProx reduces $\zeta^2$ at the optimization level, while FedMelanoma's selection reduces it at the aggregation level.

- **Term 4** ($\mathcal{O}(\gamma^2 \eta^2 \sigma^2)$): The **price of the diversity bonus**. For $\gamma = 0.3$, this contributes $0.09\eta^2 L^2 E^2 \sigma^2$ — a 9% additive overhead on Term 2's stochastic variance. This is the formal cost of preserving minority-distribution information. The tradeoff is acceptable: we pay a 9% variance overhead to prevent systematic phenotype erasure, which would otherwise produce a model with near-zero sensitivity for skin types V and VI.

---


---

### 3.9 Cost-Matrix Optimal Transport Formulation

> *This section supersedes and replaces the simplified total variation treatment in Section 3.2.4. The TV approximation incorrectly treats all category mismatches as equally costly, which misrepresents the semantic structure of both imaging modality and confirmation methodology. The general Optimal Transport framework, presented here, is the correct formulation.*

#### 3.9.1 Why Total Variation Distance Was Insufficient

Recall that in Section 3.2.4, unordered categories were handled via:

$$\text{EMD}_{\text{unordered}}(P, Q) = \frac{1}{2} \sum_c |P(c) - Q(c)|$$

This is the **total variation distance**, which implicitly uses the **unit cost matrix** — every pair of distinct categories costs 1 to move mass between. This is semantically incorrect for both of our unordered axes:

- **Modality:** Moving mass from "dermoscopic" to "clinical close-up" is a smaller representational shift than moving from "dermoscopic" to "clinical overview." The former share similar scales and framing; the latter differ in diagnostic intent, resolution, and clinical context. The unit cost treats these moves identically.

- **Confirmation Type:** Moving diagnostic confidence from "histopathology" to "expert consensus" involves a small epistemic step (both are high-confidence), while moving from "histopathology" to "single contributor" involves a large one (from near-certain to highly uncertain). Unit cost treats them equivalently.

The correct formulation requires a **semantically grounded cost matrix** $\mathbf{C}$ and the machinery of **Optimal Transport** (OT) to solve the minimum-cost redistribution problem.

#### 3.9.2 Optimal Transport: General Formulation

Let $P = (p_1, \ldots, p_n)$ and $Q = (q_1, \ldots, q_m)$ be two discrete probability distributions over category sets of sizes $n$ and $m$ respectively.

A **transport plan** $\boldsymbol{\pi} \in \mathbb{R}^{n \times m}$ describes how mass is moved from the source distribution $P$ to the target distribution $Q$. The entry $\pi_{ij} \geq 0$ represents the amount of mass transported from source category $i$ to target category $j$.

The transport plan must satisfy the **marginal constraints**:

$$\sum_{j=1}^{m} \pi_{ij} = p_i \quad \forall i \in \{1,\ldots,n\} \qquad \text{(all source mass is shipped)}$$

$$\sum_{i=1}^{n} \pi_{ij} = q_j \quad \forall j \in \{1,\ldots,m\} \qquad \text{(all target mass is received)}$$

Define the feasible set of transport plans:

$$\Pi(P, Q) = \left\{ \boldsymbol{\pi} \in \mathbb{R}_{\geq 0}^{n \times m} : \boldsymbol{\pi} \mathbf{1}_m = P, \ \boldsymbol{\pi}^T \mathbf{1}_n = Q \right\}$$

Given a **cost matrix** $\mathbf{C} \in \mathbb{R}_{\geq 0}^{n \times m}$, where $C_{ij}$ is the semantic cost of moving one unit of mass from category $i$ to category $j$, the **Earth Mover's Distance** is the minimum-cost transport plan:

$$\boxed{\text{EMD}(P, Q; \mathbf{C}) = \min_{\boldsymbol{\pi} \in \Pi(P,Q)} \sum_{i=1}^{n} \sum_{j=1}^{m} \pi_{ij} \cdot C_{ij} = \min_{\boldsymbol{\pi} \in \Pi(P,Q)} \langle \boldsymbol{\pi}, \mathbf{C} \rangle_F}$$

where $\langle \cdot, \cdot \rangle_F$ denotes the Frobenius inner product (element-wise dot product). This is a **linear program** in $\pi_{ij}$, solvable in $\mathcal{O}(n^3 \log n)$ time. For the small category counts in FedMelanoma ($n, m \leq 6$), this is computationally trivial. For larger problems, the **Sinkhorn algorithm** [13] replaces the LP with an efficient entropy-regularized approximation:

$$\text{EMD}_\varepsilon(P, Q; \mathbf{C}) = \min_{\boldsymbol{\pi} \in \Pi(P,Q)} \langle \boldsymbol{\pi}, \mathbf{C} \rangle_F + \varepsilon H(\boldsymbol{\pi})$$

where $H(\boldsymbol{\pi}) = -\sum_{ij} \pi_{ij} \log \pi_{ij}$ is the entropy of the transport plan and $\varepsilon > 0$ is the regularization strength. As $\varepsilon \to 0$, the Sinkhorn solution converges to the exact OT solution.

#### 3.9.3 Cost Matrix for Imaging Modality

The three imaging modality categories in the dataset are:

$$\mathcal{M} = \{\text{dermoscopic}, \ \text{clinical: close-up}, \ \text{clinical: overview}\}$$

The cost matrix $\mathbf{C}_{\text{mod}} \in \mathbb{R}^{3 \times 3}$ is defined as:

$$\mathbf{C}_{\text{mod}} = \begin{pmatrix} 0 & 0.30 & 0.70 \\ 0.30 & 0 & 0.45 \\ 0.70 & 0.45 & 0 \end{pmatrix}$$

where rows and columns are ordered: (dermoscopic, clinical:close-up, clinical:overview).

**Semantic justification for each entry:**

- $C_{12} = C_{21} = 0.30$ (**dermoscopic ↔ close-up**): Both modalities capture high-resolution images of individual lesions under controlled conditions. Dermoscopy uses cross-polarization or immersion contact to visualize subsurface structures; clinical close-up captures the same anatomical region without optical enhancement. The feature representations learned from each are partially transferable — the gap is real but bridgeable. Moderate cost.

- $C_{13} = C_{31} = 0.70$ (**dermoscopic ↔ overview**): Clinical overview images capture broader anatomical context (a limb, the back, facial region) with the lesion comprising a small fraction of the image. The diagnostic task is fundamentally different: dermoscopy requires reading microstructural patterns (atypical pigment network, regression structures, vascular patterns), while clinical overview requires spatial localization followed by macro-morphological assessment. Models fine-tuned on one perform poorly on the other. High cost.

- $C_{23} = C_{32} = 0.45$ (**close-up ↔ overview**): Clinical close-up shares the surface-reflectance imaging approach with clinical overview (no dermoscope), but differs in framing and scale. An intermediate cost reflects the partial representational compatibility.

- Diagonal $C_{ii} = 0$: Moving mass within the same category has zero cost.

**Why does this matter clinically?** A hospital whose data is 95% dermoscopic is distributionally very different from one whose data is 10% clinical overview — and the 0.70 cost entry correctly reflects that this is a large distributional gap, contributing more to the diversity score $D_i$ than a dermoscopic-to-close-up shift would. This causes the server to give higher aggregation weight to clients holding clinical overview data, preventing this rare modality (1.18% of the dataset, with its distinctive 54.78% malignancy rate) from being washed out by the dermoscopy-heavy majority.

#### 3.9.4 Cost Matrix for Confirmation Type

The four confirmation type categories are:

$$\mathcal{L} = \{\text{histopathology}, \ \text{serial imaging}, \ \text{expert consensus}, \ \text{single contributor}\}$$

These categories have a natural **epistemic confidence ordering** grounded in the dataset's empirical malignancy rates:

| Category | Empirical Malignancy Rate | Confidence Tier |
|---|---|---|
| Histopathology | 37.38% | Gold standard (Tier 1) |
| Serial imaging showing no change | 0.00% | High confidence benign (Tier 2) |
| Single image expert consensus | 0.29% | Moderate confidence (Tier 3) |
| Single contributor clinical assessment | 0.00% | Low confidence (Tier 4) |

The confidence-based distance between two categories is the absolute difference in their confidence tier divided by the maximum possible gap:

$$C_{ij}^{\text{conf}} = \frac{|\text{tier}(i) - \text{tier}(j)|}{3}$$

yielding the cost matrix $\mathbf{C}_{\text{conf}} \in \mathbb{R}^{4 \times 4}$:

$$\mathbf{C}_{\text{conf}} = \begin{pmatrix} 0 & 1/3 & 2/3 & 1 \\ 1/3 & 0 & 1/3 & 2/3 \\ 2/3 & 1/3 & 0 & 1/3 \\ 1 & 2/3 & 1/3 & 0 \end{pmatrix}$$

**Semantic interpretation:** The top-left entry $C_{14} = 1.0$ represents the maximum epistemic distance in the system — moving mass from histopathology-confirmed labels (near-certain ground truth) to single-contributor assessments (uncertain, 85% inter-rater agreement). A client whose dataset is entirely histopathology-confirmed is distributionally maximally distant from one whose labels come entirely from single-contributor assessment, and this full-cost entry ensures the diversity score correctly reflects that gap. Conversely, serial imaging and single contributor assessment receive a moderate cost of $2/3$ despite both having 0.00% empirical malignancy, because their epistemic confidence structures differ meaningfully (serial imaging is high-confidence benign; single contributor is uncertain).

#### 3.9.5 Worked Numerical Example

**Scenario:** Compare two hospital clients on the confirmation type axis.

- **Client A** (tertiary oncology): $P_A = [0.70, 0.15, 0.10, 0.05]$ (heavily histopathology)
- **Client B** (primary care): $P_B = [0.05, 0.10, 0.15, 0.70]$ (heavily single contributor)

This is a $4 \times 4$ OT problem. Intuitively, Client B must "ship" most of its 0.70 single-contributor mass to histopathology (cost 1.0) to match Client A's distribution — this should yield a high EMD.

The optimal transport plan $\boldsymbol{\pi}^*$ for this symmetric, nearly-antipodal problem can be computed via the linear program. A near-optimal plan ships mass along the cost-minimizing paths:

$$\pi^*_{14} = 0.05, \quad \pi^*_{11} = 0.05, \quad \pi^*_{21} = 0.10, \quad \pi^*_{31} = 0.10, \quad \pi^*_{41} = 0.45, \quad \pi^*_{42} = 0.05, \quad \pi^*_{43} = 0.15, \quad \pi^*_{44} = 0.05$$

(Exact solution requires LP solver; this illustrates the structure.) The resulting EMD is approximately:

$$\text{EMD}(P_A, P_B; \mathbf{C}_{\text{conf}}) \approx 0.45 \times 1.0 + 0.10 \times \tfrac{2}{3} + 0.10 \times \tfrac{1}{3} + \cdots \approx 0.627$$

**Compare with TV approximation:**

$$\text{TV}(P_A, P_B) = \tfrac{1}{2}(|0.70-0.05| + |0.15-0.10| + |0.10-0.15| + |0.05-0.70|) = \tfrac{1}{2}(0.65 + 0.05 + 0.05 + 0.65) = 0.700$$

The TV approximation overestimates the distance by treating all moves as unit cost — it assigns full cost to both the histopathology→single-contributor shift (cost 1.0, correctly expensive) and the serial imaging→expert consensus shift (cost 1/3, should be cheap). The OT formulation correctly penalizes the former more than the latter, producing a more semantically faithful distance of 0.627.

**Clinical interpretation:** Client B's confirmation distribution is genuinely less "gold-standard" than Client A's, but the OT distance is modestly lower than TV predicts because some of its mass (serial imaging, 0.10) is still high-confidence even if not histopathological. The OT metric correctly recognizes this partial compatibility.

#### 3.9.6 Updated Combined Diversity Score

With the cost-matrix OT formulation, the combined diversity score $D_i^{\text{raw}}$ is updated to:

$$D_i^{\text{raw}} = w_\varphi \cdot W_1\!\left(\hat{P}_\varphi^{\text{global}}, P_\varphi^{(i)}\right) + w_\tau \cdot \text{OT}\!\left(\hat{P}_\tau^{\text{global}}, P_\tau^{(i)}; \mathbf{C}_{\text{mod}}\right) + w_\lambda \cdot \text{OT}\!\left(\hat{P}_\lambda^{\text{global}}, P_\lambda^{(i)}; \mathbf{C}_{\text{conf}}\right)$$

where:
- $W_1(\cdot, \cdot)$ is the 1D Wasserstein distance for Fitzpatrick (ordinal, Section 3.2.2)
- $\text{OT}(\cdot, \cdot; \mathbf{C}_{\text{mod}})$ is the OT distance with the modality cost matrix (Section 3.9.3)
- $\text{OT}(\cdot, \cdot; \mathbf{C}_{\text{conf}})$ is the OT distance with the confirmation cost matrix (Section 3.9.4)
- Default weights: $w_\varphi = 0.40$, $w_\tau = 0.35$, $w_\lambda = 0.25$

This replaces the hybrid $W_1$ + TV formulation from Section 3.2.5. The normalization step is unchanged: $D_i = D_i^{\text{raw}} / \max_j D_j^{\text{raw}}$.

---

### 3.10 Logit Calibration & Clinical Reliability

> *"A model that says it is 95% confident should be correct 95% of the time. In medical AI, overconfidence is not a statistical curiosity — it is a patient safety risk."*

#### 3.10.1 The Calibration Problem

A model is **well-calibrated** if its predicted confidence is an accurate estimate of the true probability of being correct. Formally, for any predicted confidence $\hat{p}$:

$$P(y = 1 \mid \hat{p}) = \hat{p}$$

In practice, deep neural networks — particularly those trained with cross-entropy loss on large datasets — are **systematically overconfident** [14]. A dermoscopy model may output $\hat{p}_{\text{malignant}} = 0.92$ for a lesion where the true probability (based on inter-rater agreement and histopathological follow-up) is only 0.68. This gap has direct clinical consequences: a clinician using the model's confidence score to prioritize biopsies or triage patients is working with a distorted risk estimate.

The overconfidence problem is exacerbated in the federated setting by two compounding effects:

1. **Focal loss sharpening:** The class-weighted focal loss suppresses easy examples and amplifies hard ones. While this improves sensitivity, it produces a training signal that systematically pushes predicted probabilities toward extremes, worsening calibration.

2. **Non-IID distributional mismatch:** The global model is aggregated from clients with very different data distributions. At inference time, an input from a distribution not well-represented in any single client's training data may receive highly confident — but entirely spurious — predictions.

#### 3.10.2 Temperature Scaling

Temperature scaling [14] is the simplest and most effective post-hoc calibration technique for neural networks. Given the raw logit vector $\mathbf{l} = [l_0, l_1]^T \in \mathbb{R}^2$ produced by the model, the calibrated probability is:

$$\hat{\mathbf{p}}_{\text{cal}} = \text{softmax}\!\left(\frac{\mathbf{l}}{T}\right), \quad \hat{p}_c = \frac{\exp(l_c / T)}{\exp(l_0/T) + \exp(l_1/T)}$$

where $T > 0$ is a single learnable **temperature parameter**.

**Effect of temperature on confidence:**

| Temperature $T$ | Effect on probabilities | Use case |
|---|---|---|
| $T < 1$ | Sharpens: predictions push toward 0 and 1 | Default network behavior (overconfident) |
| $T = 1$ | Identity: standard softmax, no change | Uncalibrated baseline |
| $T > 1$ | Softens: predictions retreat toward $0.5$ | Calibrated output (typical result after tuning) |
| $T \to \infty$ | Uniform distribution: $\hat{p}_c = 0.5 \,\forall c$ | Maximum uncertainty |

Intuitively, $T > 1$ "spreads out" the logit differences before applying softmax. A raw logit difference of $l_1 - l_0 = 3$ corresponds to $\hat{p}_1 \approx 0.95$. With $T = 1.5$, the effective difference is $3/1.5 = 2$, corresponding to $\hat{p}_1 \approx 0.88$ — a more conservative, better-calibrated estimate.

#### 3.10.3 Calibration Objective

Temperature $T^*$ is found by minimizing the **negative log-likelihood (NLL)** on a held-out calibration set $\mathcal{D}_{\text{cal}}$, treating all other model parameters as fixed:

$$T^* = \underset{T > 0}{\arg\min} \; \frac{1}{|\mathcal{D}_{\text{cal}}|} \sum_{(x_j, y_j) \in \mathcal{D}_{\text{cal}}} -\log \hat{p}_{y_j}^{\text{cal}}(T)$$

This is a one-dimensional convex optimization — it can be solved exactly with bisection or gradient descent in seconds. Critically, **temperature scaling does not change the model's predictions** (argmax is preserved) — it only adjusts the confidence values. This makes it a "free" calibration: no retraining, no change in accuracy metrics, only improved reliability of confidence estimates.

In the federated context, temperature calibration is applied **at the server after aggregation**, on the global model using the server's held-out validation set. Each federated round can produce a slightly different optimal $T^*$, so recalibration is run after each evaluation checkpoint.

#### 3.10.4 Expected Calibration Error (ECE)

The **Expected Calibration Error** quantifies the average miscalibration across all confidence levels. The procedure is:

1. Partition all $n$ validation predictions into $K_{\text{bin}} = 10$ equally spaced confidence bins $\mathcal{B}_k = ((k-1)/K_{\text{bin}}, \, k/K_{\text{bin}}]$ for $k = 1, \ldots, K_{\text{bin}}$.

2. For each bin $\mathcal{B}_k$, compute:
   - $\text{acc}(\mathcal{B}_k) = \frac{1}{|\mathcal{B}_k|} \sum_{j \in \mathcal{B}_k} \mathbb{1}[\hat{y}_j = y_j]$ — mean accuracy within the bin
   - $\text{conf}(\mathcal{B}_k) = \frac{1}{|\mathcal{B}_k|} \sum_{j \in \mathcal{B}_k} \hat{p}_{y_j}^{\text{cal}}$ — mean predicted confidence within the bin

3. Compute ECE as the confidence-weighted average gap:

$$\text{ECE} = \sum_{k=1}^{K_{\text{bin}}} \frac{|\mathcal{B}_k|}{n} \left| \text{acc}(\mathcal{B}_k) - \text{conf}(\mathcal{B}_k) \right|$$

**Interpretation:** A perfectly calibrated model has $\text{ECE} = 0$. A typical uncalibrated deep network has $\text{ECE} \approx 0.05$–$0.15$ (5–15% systematic miscalibration). After temperature scaling, ECE typically drops below $0.03$.

**Clinical implication:** An ECE of 0.10 means that when the model outputs "90% confident this is malignant," it is actually correct only ~80% of the time. At scale — if this model screens 10,000 patients per year — this 10% miscalibration translates to hundreds of incorrectly risk-stratified patients, some of whom may receive delayed diagnosis.

#### 3.10.5 Brier Score

The **Brier Score** provides a proper scoring rule that rewards both accuracy and calibration jointly:

$$\text{BS} = \frac{1}{n} \sum_{j=1}^{n} \left(\hat{p}_{1,j} - y_j\right)^2$$

A Brier Score of 0 is perfect; 1 is the worst possible. The Brier Score decomposes into:

$$\text{BS} = \underbrace{\text{Calibration}}_{\text{reliability term}} + \underbrace{\text{Resolution}}_{\text{sharpness term}} - \underbrace{\text{Uncertainty}}_{\text{climatological baseline}}$$

Unlike ECE, the Brier Score is a **single unified metric** — it penalizes both overconfident wrong predictions and underconfident correct predictions. It is particularly appropriate for highly imbalanced medical classification, where ECE can be dominated by the majority class.

#### 3.10.6 Reliability Diagram

The **reliability diagram** (or calibration plot) is the standard visual tool for assessing calibration. For each confidence bin $\mathcal{B}_k$, a point is plotted at $(\text{conf}(\mathcal{B}_k), \text{acc}(\mathcal{B}_k))$. A perfectly calibrated model produces points on the diagonal $y = x$.

**What deviations indicate:**

- **Points above the diagonal:** The model is **underconfident** at that confidence level — it says "70% certain" but is actually correct 85% of the time. Common in models trained with heavy regularization.

- **Points below the diagonal:** The model is **overconfident** — it says "90% certain" but is only correct 75% of the time. This is the typical failure mode of focal-loss-trained networks and is the target of temperature scaling.

In the FedMelanoma context, monitoring the reliability diagram across rounds reveals whether the aggregation process is progressively improving or degrading calibration. Improvements in calibration — in addition to sensitivity and AUC — should be tracked as a first-class training objective for a medically deployable system.

## 4. End-to-End Implementation Strategy

This section describes the complete implementation strategy at the system level. It is organized as eight sequential phases, each with precise technical specifications. No code is included — this is the architectural and procedural blueprint from which code is derived.

### Phase 1: Data Pipeline

**1.1 Preprocessing Assumptions**

The preprocessed dataset (`preprocessed_data.csv`) is already cleaned and contains no missing values. The key preprocessing decisions and their rationale are:

- **No imputation required.** Unlike the full ISIC archive (where Fitzpatrick type was absent for 90.1% of samples), this preprocessed subset contains complete Fitzpatrick annotations for all 9,740 records. All three metadata axes ($\varphi$, $\tau^{\text{mod}}$, $\lambda$) are fully populated.

- **Binary label encoding.** The `label` column is already binary (0/1). No transformation needed.

- **Image retrieval.** The `isic_id` column serves as the primary key for fetching associated images from the ISIC archive via the standard naming convention `{isic_id}.jpg`.

**1.2 Metadata Encoding**

Metadata encoding follows a strict one-hot scheme producing a 13-dimensional binary vector:

- Fitzpatrick skin type `skin_type`: 6-dimensional one-hot for $\{$I, II, III, IV, V, VI$\}$
- Imaging modality `image_type`: 3-dimensional one-hot for $\{$dermoscopic, clinical:close-up, clinical:overview$\}$
- Confirmation type `confirm_type`: 4-dimensional one-hot for $\{$histopathology, serial imaging, single image expert consensus, single contributor clinical assessment$\}$

The encoding order is fixed across all clients and the server. Consistent encoding is a hard requirement: any mismatch would cause the Metadata MLP's weights to learn different semantic mappings on different clients, breaking parameter compatibility.

**1.3 Image Preprocessing**

All images are resized to 224×224 pixels and normalized using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). During training, augmentation includes: random horizontal/vertical flips ($p=0.5$), random rotation ($\pm 15°$), color jitter (brightness $\pm 0.2$, contrast $\pm 0.2$, saturation $\pm 0.1$), and random erasing ($p=0.1$). DullRazor-inspired morphological filtering removes hair artifacts prevalent in dermoscopic images.

Augmentation parameters are **intentionally varied slightly across clients** to simulate device-specific preprocessing pipelines. This reinforces Non-IID realism and prevents the global model from learning preprocessing artifacts specific to a single client.

**1.4 Train/Validation/Test Split**

A **stratified split** of 70%/15%/15% is applied globally, stratified by both `label` and `image_type`. Stratification by modality is critical given that clinical overview carries a 54.78% malignancy rate — random splitting could create severe test-set contamination and misleadingly inflated evaluation numbers.

---

### Phase 2: Client Simulation / Hospital Setup

**2.1 Dataset Splitting (Non-IID)**

The 9,740-sample dataset is partitioned into $K=10$ client subsets using the softmax-temperature assignment with $\tau_{\text{soft}} = 0.5$ and the $K \times 3$ facility profile matrix $\mathbf{W}$. Each sample is probabilistically assigned to exactly one client. Resulting client sizes range from ~300 (telemedicine) to ~2,500 (tertiary oncology). Within-client malignancy rates range from ~0–2% (primary care, telemedicine) to ~25–35% (tertiary oncology), a 10–35× spread that directly motivates the focal loss formulation.

**2.2 Straggler Probability Model**

Straggler behavior is not modeled as a simple binary flag — it is a continuous stochastic process. For each client $i$ and round $t$, two independent random variables govern straggler behavior:

**Completion probability** $p_{\text{complete}}^{(i)}$: The probability that client $i$ returns an update before the time window $W_t$ expires. This is sampled from a Beta distribution:

$$p_{\text{complete}}^{(i)} \sim \text{Beta}(a_i, b_i)$$

where $(a_i, b_i)$ parameterizes the client's reliability tier:

| Client Tier | Example Clients | $(a_i, b_i)$ | Mean $p_{\text{complete}}$ | Variance |
|---|---|---|---|---|
| Reliable | C1–C5 (oncology, derm) | $(19, 1)$ | $0.95$ | $0.0045$ |
| Moderate | C6–C8 (primary care) | $(7, 3)$ | $0.70$ | $0.021$ |
| High-Straggler | C9–C10 (telemedicine) | $(2, 3)$ | $0.40$ | $0.048$ |

Using a Beta distribution rather than a fixed probability captures the **round-to-round variability** in straggler behavior — a telemedicine node may complete on time in rounds with light hospital load and fail in rounds coinciding with evening patient surges or connectivity issues.

**Compute delay** $T_{\text{delay}}^{(i,t)}$: The wall-clock time for local training and upload, sampled from a log-normal distribution:

$$\log T_{\text{delay}}^{(i,t)} \sim \mathcal{N}(\mu_i, \sigma_i^2)$$

| Client Tier | $\mu_i$ (log-seconds) | $\sigma_i$ | Expected delay | 95th percentile |
|---|---|---|---|---|
| Reliable | $1.39$ ($\approx \log 4$) | $0.3$ | ~5 sec | ~8 sec |
| Moderate | $2.77$ ($\approx \log 16$) | $0.5$ | ~20 sec | ~45 sec |
| High-Straggler | $4.09$ ($\approx \log 60$) | $0.8$ | ~75 sec | ~240 sec |

The log-normal distribution is appropriate because compute delays are strictly positive, right-skewed (occasional extreme outliers from server contention or network drops), and multiplicatively structured (doubling dataset size roughly doubles training time).

**Round-level staleness computation:** At the start of round $t$, the straggler detector computes the adaptive time window $W_t = \tilde{T}_{\text{med}} + k \cdot \text{MAD}$ from the history of completion times. A client $i$ is classified as a straggler for round $t$ if $T_{\text{delay}}^{(i,t)} > W_t$, in which case its staleness for the next round is $s_i = 1$.

**2.3 Client Process Architecture**

Each client is an independent Python subprocess, isolated with its own model copy, dataset partition, and training state. Clients communicate with the server exclusively through the Flower gRPC channel. No inter-client communication exists. This isolation ensures privacy: clients cannot observe each other's data, updates, or scores.

The client process lifecycle is:

1. **Registration:** Client submits dataset size $n_i$, metadata distribution profile (summary statistics only — no raw data), and cluster assignment request to the server. Server assigns cluster $g_i \in \mathcal{G}$, computes initial $D_i$ and $M_i$, and issues an X.509 certificate.

2. **Standby:** Client listens on its gRPC channel for `FitIns` (training instruction) from the server.

3. **Training:** Upon receiving `FitIns` carrying the current global model $\mathbf{w}^t$, the client loads parameters, runs $E$ local epochs, computes $\Delta\mathbf{w}_i$, and returns `FitRes` with the compressed delta, $n_i$, and any client-side metrics.

4. **Idle:** If not selected in round $t$, the client discards the round notification and returns to standby. Its local model copy is **not updated** — it retains $\mathbf{w}^{t-1}$ (the last round it was selected for) until it receives the next `FitIns`.

---

### Phase 3: Training Loop

**3.1 Round-Based Workflow — Detailed Five-Stage Flow**

Each federated round $t \in \{1, \ldots, T\}$ executes the following ordered stages:

**Stage 1 — Client Selection.** The server's Selection Engine runs the composite scoring and constraint-enforced greedy selection algorithm (Phase 6) to produce $\mathcal{S}^t \subseteq \mathcal{H}$, $|\mathcal{S}^t| = C = 6$. The Straggler Detector records $t_{\text{start}}^t = \text{now}()$ and initializes the time window $W_t$.

**Stage 2 — Global Model Broadcast.** The server serializes the current global weights $\mathbf{w}^t$ as a Protocol Buffers message, compresses with zstd, and transmits to all $c \in \mathcal{S}^t$ simultaneously over their established mTLS gRPC channels. Non-selected clients receive no transmission.

**Stage 3 — Parallel Local Training.** Each selected client $i \in \mathcal{S}^t$, upon receiving $\mathbf{w}^t$, initializes its local model to these weights and trains for $E = 5$ epochs over its local dataset $\mathcal{D}_i$. Specifically:

- The optimizer state (AdamW momentum and variance buffers) is **reset at the start of each round**. This prevents stale momentum from prior rounds accumulating drift — a subtle but important correctness detail in FL.
- Local training proceeds with cosine annealing scheduled from $\eta = 10^{-4}$ (annealing within the local $E$ epochs is minimal at this scale; the schedule primarily operates over the global $T$ rounds by resetting $\eta$ to the schedule's current value at each round).
- Class weights $\hat{\alpha}_c$ are recomputed once per round from the local dataset's label distribution before the first epoch.
- After $E$ epochs, the client computes $\Delta\mathbf{w}_i = \mathbf{w}_i^{t+1} - \mathbf{w}^t$ and packages it with $n_i$ into a `FitRes` message.

**Stage 4 — Update Collection.** The Straggler Detector monitors incoming `FitRes` messages against the time window $W_t$. Two termination conditions trigger aggregation:

- **(A) Quorum:** $|\mathcal{A}^t| \geq \rho \cdot C = 0.70 \times 6 = 4.2 \Rightarrow 5$ arrived clients. Early termination upon quorum prevents unnecessary waiting when all fast clients have already responded.
- **(B) Window expiry:** $\text{now}() - t_{\text{start}}^t \geq W_t$. Forces aggregation regardless of how many clients have arrived. Late-arriving updates after window expiry are either buffered (if $s_i \leq S_{\max}$) or discarded (if $s_i > S_{\max}$).

**Stage 5 — Aggregation and State Update.** The server runs the staleness-weighted aggregation (Phase 5) to produce $\mathbf{w}^{t+1}$, updates all client state variables ($R_i$, last-selection round, straggler buffer), and checkpoints the global model if $t \bmod 5 = 0$. The round counter increments and Stage 1 begins for $t+1$.

**3.2 Local Training Details**

Each client trains with AdamW ($\eta = 10^{-4}$, $\lambda_{\text{wd}} = 10^{-4}$), cosine annealing LR schedule, class-weighted focal loss ($\gamma_f = 2$), and batch size 32. After $E$ epochs, only the weight delta $\Delta\mathbf{w}_i$ is transmitted. For EfficientNet-B3 with the metadata fusion head, the full model is ~45 MB; deltas compressed with zstd are ~10–15 MB.

**The rationale for $E = 5$ local epochs** is grounded in the convergence analysis of Section 3.8. With Non-IID degree $\zeta^2 > 0$, Term 3 of the convergence bound grows as $\mathcal{O}(\eta^2 E^2 \zeta^2)$. Increasing $E$ accelerates local convergence but amplifies client drift. $E = 5$ is the empirically stable setting recommended for high-heterogeneity FL in the FedProx analysis [4]; smaller $E$ (e.g., 1) gives FedSGD-like behavior with high communication cost, while larger $E$ (e.g., 20) risks divergence under our Non-IID partition.

---

### Phase 4: Communication

**4.1 gRPC over Mutual TLS — Technical Rationale**

All model delta transmissions use **gRPC** (Google Remote Procedure Call) over HTTP/2 with Protocol Buffers v3 serialization. The choice of gRPC over REST or WebSocket is motivated by three technical requirements specific to FL:

- **Bidirectional streaming:** HTTP/2 multiplexing allows the server to simultaneously receive updates from multiple clients over a single persistent connection, without the per-request connection overhead of HTTP/1.1. This is critical for the concurrent Phase 3 → Phase 4 transition where all 6 selected clients transmit in parallel.

- **Protobuf efficiency:** Protocol Buffers serialize float32 tensors at ~4 bytes/parameter with no per-element overhead. A 45 MB model delta transmits as a ~45 MB binary blob with minimal framing overhead, compared to JSON's ~3× size overhead for numeric arrays.

- **Streaming support:** Large weight deltas can be streamed in chunks rather than loaded entirely into memory before transmission, reducing peak memory usage on resource-constrained telemedicine clients.

**Mutual TLS (mTLS)** provides the following security guarantees in the medical context:

| Security Property | Mechanism | Medical Relevance |
|---|---|---|
| Data confidentiality | AES-256-GCM encryption | Weight deltas must not be interceptable (gradient inversion risk) |
| Server authentication | Server X.509 cert, CA-signed | Clients must not connect to spoofed aggregation servers |
| Client authentication | Per-client X.509 cert | Server must reject unauthorized "fake hospital" injections |
| Certificate revocation | CRL / OCSP | Compromised clients can be revoked without full redeployment |

**Gradient inversion attacks** [Zhu et al., 2019] demonstrate that model gradients can be used to reconstruct training images with high fidelity. mTLS ensures that weight deltas are never visible to network-level adversaries, adding a transport-level privacy layer above the primary FL privacy guarantee (no raw data leaving the client).

**4.2 Compression Trade-offs**

| Compression Level | Algorithm | Compression Ratio | Compression Time | Decompression Time |
|---|---|---|---|---|
| Level 1 | zstd | ~2.5× | Very fast | Very fast |
| Level 3 (default) | zstd | ~3.5× | Fast | Fast |
| Level 9 | zstd | ~5.0× | Moderate | Fast |
| Level 19 (max) | zstd | ~6.5× | Slow | Fast |

**Bandwidth-adaptive compression** is applied based on estimated client bandwidth:

- Clients with estimated bandwidth $> 10$ Mbps: Level 3 (balanced)
- Clients with bandwidth $1$–$10$ Mbps: Level 9 (bandwidth-constrained)
- Clients with bandwidth $< 1$ Mbps (telemedicine): Level 19 (maximize compression, accept encoding latency)

Zstd decompression is asymmetrically fast regardless of compression level — the server always decompresses quickly even when clients use maximum compression, preventing server-side bottlenecks.

**Communication budget per round:** At Level 3 with $C = 6$ clients transmitting 13 MB deltas each, the total server-received data per round is $6 \times 13 = 78$ MB. Over $T = 50$ rounds, total ingress is $\approx 3.9$ GB — a very reasonable communication budget for a 50-round training run.

---

### Phase 5: Aggregation

**5.1 Step-by-Step Aggregation Procedure**

The aggregation engine executes the following steps upon quorum trigger or window expiration:

**Step 1 — Collect arrived updates.** Gather $\{(\Delta\mathbf{w}_i, n_i, s_i)\}_{i \in \mathcal{A}^t}$ where $\mathcal{A}^t$ includes on-time arrivals plus any buffered straggler updates from round $t-1$ (staleness incremented to $s_i = 2$).

**Step 2 — Retrieve diversity scores.** For each $i \in \mathcal{A}^t$, retrieve $D_i$ from the Selection Engine's state. These scores were computed at the start of round $t$ and remain valid for this aggregation. (Diversity scores are not recomputed mid-round.)

**Step 3 — Compute raw unnormalized weights.** For each arrived client:

$$\tilde{c}_i = \frac{n_i}{n_{\mathcal{A}}} \cdot \delta^{s_i} \cdot (1 + \gamma D_i), \quad n_{\mathcal{A}} = \sum_{j \in \mathcal{A}^t} n_j$$

The size weight $n_i / n_\mathcal{A}$ uses the sum of **arrived** clients' dataset sizes, not the total federation size. This ensures the normalization is internally consistent: if large-dataset clients are all stragglers and only small-dataset clients arrive, the small clients still contribute meaningfully rather than being reduced to near-zero by a denominator dominated by absent clients.

**Step 4 — Normalize to sum to 1.**

$$c_i = \frac{\tilde{c}_i}{\sum_{j \in \mathcal{A}^t} \tilde{c}_j}$$

This normalization is the key that makes the update well-scaled regardless of how many clients arrived. Formally, it converts the aggregation from a weighted average of absolute model states (which would collapse if few clients arrive) into a weighted average of deltas scaled to produce a single full gradient step.

**Step 5 — Apply weighted delta aggregation.**

$$\mathbf{w}^{t+1} \leftarrow \mathbf{w}^t + \sum_{i \in \mathcal{A}^t} c_i \cdot \Delta\mathbf{w}_i$$

This is computed layer-by-layer for all parameter tensors in the model. The operation is a simple weighted sum of tensors of identical shape, confirming why **architectural homogeneity** (all clients using identical model architectures) is a hard system requirement.

**Step 6 — Post-hoc temperature recalibration.** If $t \bmod 5 = 0$ (evaluation round), recalibrate $T^*$ on the server's held-out validation set using the procedure from Section 3.10.3. Store $T^*$ alongside the global model checkpoint.

**Step 7 — Checkpoint and state update.** Write $\mathbf{w}^{t+1}$ and $T^*$ to disk. Update all client state variables: $R_i$ EMA, last-selection timestamp, straggler buffer management.

**5.2 Handling the Straggler Buffer**

Late-arriving updates are held in a keyed buffer indexed by client ID. At round $t+1$'s aggregation:

- Buffered updates with $s_i \leq S_{\max} = 3$ are included with their incremented staleness.
- Buffered updates with $s_i > S_{\max}$ are discarded; the originating client's $R_i$ receives an additional penalty of $-0.1$ (applied directly to the EMA, overriding the normal binary update).

The buffer is a **single-slot per client**: if a client submits a second late update while a previous one is already buffered, the newer update replaces the older (higher staleness) one, since it represents a more recent gradient direction.

---

### Phase 6: Selection Engine

**6.1 Score Computation — Full Pipeline**

At the start of each round $t$, the Selection Engine executes a five-step scoring pipeline:

**Step 1 — Recompute global distribution centroid.** The server maintains a running weighted average of all client metadata distributions, updated whenever a client's profile changes (which in the simulation is stable, but in real deployment would update as new patient data arrives):

$$\hat{P}_\varphi^{\text{global}} = \frac{\sum_{i=1}^{K} n_i \cdot P_\varphi^{(i)}}{\sum_{i=1}^{K} n_i}, \quad \text{similarly for } \hat{P}_\tau, \hat{P}_\lambda$$

**Step 2 — Compute per-axis OT distances.** For each client $i$, compute:
- $W_1(\hat{P}_\varphi^{\text{global}}, P_\varphi^{(i)})$ using the 1D Wasserstein formula (Section 3.2.2)
- $\text{OT}(\hat{P}_\tau^{\text{global}}, P_\tau^{(i)}; \mathbf{C}_{\text{mod}})$ using the modality cost matrix LP (Section 3.9.3)
- $\text{OT}(\hat{P}_\lambda^{\text{global}}, P_\lambda^{(i)}; \mathbf{C}_{\text{conf}})$ using the confirmation cost matrix LP (Section 3.9.4)

**Step 3 — Combine and normalize to $D_i$.** Weight the three OT distances, sum, and normalize across all clients to $[0,1]$.

**Step 4 — Compute metadata coverage $M_i$.** For each client, count rare category presence:

$$M_i = \frac{1}{|\mathcal{C}_{\text{rare}}|} \sum_{c \in \mathcal{C}_{\text{rare}}} \mathbb{1}[P_c^{(i)} > \theta_{\min}]$$

where $\mathcal{C}_{\text{rare}} = \{$Fitzpatrick IV, V, VI; clinical:overview modality; histopathology confirmation$\}$ and $\theta_{\min} = 0.01$ (must represent at least 1% of client's data).

**Step 5 — Compute composite score with aging boost.**

$$\text{score}_i^{\text{eff}} = \alpha R_i + \beta D_i + \gamma M_i + \eta_{\text{age}} \cdot \max(0, \ t - t_i^{\text{last}} - 1)$$

The aging boost is zero for the first round after a client was last selected, and increases by $\eta_{\text{age}} = 0.05$ per unselected round thereafter. This produces a predictable, deterministic boost that the operator can reason about: after $W_{\text{fair}} = 10$ unselected rounds, the aging boost is $0.05 \times 9 = 0.45$ — large enough to overcome almost any score disadvantage and guarantee selection.

**6.2 Constraint-Enforced Greedy Selection**

The selection procedure is a **three-phase greedy algorithm** with hard constraint enforcement:

**Phase 1 — Cluster coverage (constraint C2):** For each of the $|\mathcal{G}| = 4$ distributional clusters, select the highest $\text{score}_i^{\text{eff}}$ eligible client from that cluster. "Eligible" means $R_i \geq 0.8 \cdot R_{\min}$ (relaxed reliability for coverage guarantee). This produces 4 selected clients.

**Phase 2 — Score-based fill (constraint C1):** Fill the remaining $C - 4 = 2$ slots by greedily selecting the highest-$\text{score}_i^{\text{eff}}$ remaining clients, subject to the constraint that the mean reliability of all 6 selected clients satisfies $\bar{R} \geq R_{\min} = 0.50$. Each candidate addition is tested against this constraint before being confirmed.

**Phase 3 — Fairness enforcement:** Scan all unselected clients for those satisfying $t - t_i^{\text{last}} \geq W_{\text{fair}} - 1 = 9$. For each such client, forcibly insert it into the selected set, displacing the lowest-$\text{score}_i^{\text{eff}}$ Phase 2 selection. This is the **hard fairness guarantee** — it fires regardless of score.

The three-phase structure ensures the algorithm is **complete**: it always produces exactly $C = 6$ selected clients satisfying all three constraints, even under adversarial scenarios (e.g., all low-score clients happening to be in the same cluster, or multiple clients simultaneously triggering the fairness window).

---

### Phase 7: Deployment

**7.1 Real Hospital Deployment**

In production, each hospital node runs the FedMelanoma client software as a **Docker container** on its local infrastructure. Containerization enforces a consistent runtime environment (Python version, library dependencies, CUDA version) across heterogeneous hospital hardware, eliminating the "works on my machine" problem that is particularly acute in healthcare settings where IT infrastructure is fragmented.

The client container:

- Accesses the hospital's local DICOM archive (or ISIC-compatible CSV + image directory) via a **data access adapter layer** — an abstraction that decouples the FL client from the hospital's specific PACS system, supporting HL7 FHIR, DICOM SR, and flat-file exports without code changes.
- Registers with the central server by transmitting a compact metadata profile (distribution statistics, cluster assignment, dataset size) — never raw patient records.
- Participates in rounds as scheduled by the server's selection engine.
- Runs as a **low-priority process** that can be preempted by clinical workloads without data corruption (training state is checkpointed after each epoch; incomplete epochs are discarded and restarted from the last checkpoint on the next round selection).

**Deployment prerequisites per hospital node:**

| Requirement | Minimum | Recommended |
|---|---|---|
| GPU | None (CPU fallback) | NVIDIA RTX 3060 or better |
| RAM | 8 GB | 16 GB |
| Storage | 50 GB | 200 GB (for image cache) |
| Network | 1 Mbps sustained | 10+ Mbps |
| OS | Ubuntu 20.04+ / Windows Server 2019+ | Ubuntu 22.04 LTS |

**7.2 Telemedicine Nodes**

Telemedicine nodes in low-resource settings operate under three specialized constraints not present in hospital nodes:

**Offline training mode:** The client software supports a **disconnected training workflow** where local epochs are performed during idle periods (nights, weekends, between patient consultations) and updates are buffered locally. Transmission occurs opportunistically when connectivity is available. The server accommodates this via the straggler buffer — a telemedicine node that trains overnight and transmits at dawn is handled as a straggler with $s_i = 1$, not as a failure.

**Bandwidth-adaptive compression:** For nodes with sustained bandwidth below 1 Mbps, zstd Level 19 is applied. At this compression level, the ~45 MB model delta compresses to ~7 MB, requiring approximately 60 seconds for transmission at 1 Mbps — feasible within a reasonable time window.

**Graceful degradation on CPU:** EfficientNet-B3 training on CPU for $E = 5$ epochs over a 300-sample partition takes approximately 25–40 minutes on a modern laptop CPU. This is the worst-case compute scenario for telemedicine nodes and directly informs the $W_t$ upper bound of $\max_t W_t = 600$ seconds (10 minutes) in the config — deliberately set below this worst-case to classify CPU-only telemedicine nodes as expected stragglers, whose updates are buffered rather than awaited.

**7.3 Edge Deployment Constraints**

Several architectural decisions in FedMelanoma were made specifically to support edge deployment:

- **Delta-only transmission:** Transmitting $\Delta\mathbf{w}_i$ rather than full $\mathbf{w}_i^{t+1}$ halves the effective model size in practice (deltas are sparser and compress better than absolute weights), reducing both bandwidth and storage requirements on the client.

- **Single-file Docker image:** The client container bundles all dependencies into a single Docker image (~4 GB including CUDA libraries), enabling deployment on air-gapped hospital networks with no outbound internet access beyond the gRPC channel to the FL server.

- **Resumable training:** All training state (optimizer, epoch counter, current batch) is checkpointed to disk every 100 batches. Power outages or scheduled maintenance restarts pick up from the last checkpoint without discarding the current round's work.

---

### Phase 8: Monitoring

**8.1 Logging Architecture**

FedMelanoma uses a **three-tier logging strategy** that separates operational, mathematical, and privacy-sensitive logs:

**Tier 1 — Operational logs** (written to disk, rotatable): Round number, wall-clock time per stage, number of clients arrived vs. selected, quorum outcome, straggler count, checkpoint events. These logs contain no model weights or patient-derived statistics.

**Tier 2 — Mathematical logs** (written to experiment tracker, e.g., MLflow or Weights & Biases): Per-round loss curves (global and per-client when available), gradient norms, contribution weights $c_i$, diversity scores $D_i$, reliability EMAs $R_i$, time window $W_t$ values, staleness distribution. These enable post-hoc convergence analysis and hyperparameter tuning.

**Tier 3 — Audit logs** (write-once, tamper-evident): Round timestamps, client participation records, global model version hashes (SHA-256 of $\mathbf{w}^t$), certificate serial numbers. These logs satisfy clinical auditability requirements — the hospital's compliance officer can verify which clients contributed to which model version.

**8.2 Global Model Performance Monitoring**

Every 5 rounds, the global model is evaluated on the held-out test set. Metrics reported are formalized in Section 4.1 (Evaluation Framework). Alert thresholds trigger notifications to the system operator:

- AUC-ROC drops by $> 0.03$ relative to the previous evaluation checkpoint
- Sensitivity for the malignant class falls below $0.75$
- ECE (post-temperature-scaling) exceeds $0.05$
- Fitzpatrick fairness gap $\Delta_{\text{fair}}$ increases by $> 0.05$ relative to the previous checkpoint (indicating diverging cross-phenotype bias)

**8.3 Bias and Fairness Detection**

The Fitzpatrick fairness gap $\Delta_{\text{fair}}$ is the primary equity monitor:

$$\Delta_{\text{fair}}^{(t)} = \max_{\varphi \in \Phi} \text{Sensitivity}(\varphi) - \min_{\varphi \in \Phi} \text{Sensitivity}(\varphi)$$

where $\Phi = \{$I, II, III, IV, V, VI$\}$. A well-functioning diversity-compensated system should show a **decreasing or stable** $\Delta_{\text{fair}}^{(t)}$ over rounds, reflecting that the global model is learning to perform equitably across phenotypes.

For Fitzpatrick V and VI (0.00% malignancy in the training data), sensitivity is **undefined** at training time — there are no positive labels to compute true positives against. Monitoring for these groups requires **external validation** on independent datasets containing confirmed dark-skin melanoma cases. This is flagged as a known limitation of the current dataset and addressed in the deployment strategy by ensuring telemedicine nodes (which carry these phenotypes) are included in the federation even when their malignancy contribution is zero.

**8.4 Distribution Drift Detection**

In a live deployment where hospital data evolves over time (new patients, seasonal diagnosis patterns, updated confirmation standards), the metadata distributions $P^{(i)}$ may shift between FL rounds. Distribution drift invalidates the previously computed diversity scores and cluster assignments.

FedMelanoma detects drift using a **sliding-window two-sample test** per client per axis:

- For Fitzpatrick distribution: Kolmogorov-Smirnov (KS) test comparing the current round's local distribution against the baseline registration profile.
- For modality and confirmation: Chi-squared goodness-of-fit test.

When drift is detected at $p < 0.05$ for any axis, the affected client's diversity score and cluster assignment are recomputed from scratch. If cluster assignment changes (e.g., a telemedicine node's demographics shift from dark-skin to fair-skin majority), the cluster coverage constraint in Phase 6 is re-evaluated to ensure the new cluster structure is preserved.

**8.5 Client-Level Monitoring**

Per-client metrics tracked server-side (using only metadata statistics, no raw data):

- Reliability EMA trajectory $R_i^{(t)}$: plot over rounds to identify declining-reliability clients before they become problematic stragglers
- Contribution weight $c_i$ per round: a client whose $c_i$ has been near-zero for 10+ rounds is effectively excluded from training — this warrants human review
- Aging boost activation: rounds per client in which the aging boost was the decisive factor in selection (indicating the client would not have been selected on merit alone)
- Straggler rate over sliding 10-round window: sudden spikes in straggler rate may indicate infrastructure problems at specific hospitals

---

## 5. Evaluation & Validation Framework

> *This section formally defines all metrics used to evaluate FedMelanoma. Each metric is derived from first principles, with clinical interpretation. These metrics collectively assess model quality, calibration, fairness, and federated system performance.*

### 5.1 Primary Classification Metrics

**Dataset partition for evaluation:** The global test set consists of $n_{\text{test}} = \lfloor 0.15 \times 9740 \rfloor = 1461$ samples, stratified by label and image type. This set is held exclusively at the server and never shared with any client.

#### 5.1.1 AUROC (Area Under the Receiver Operating Characteristic Curve)

The **ROC curve** plots the True Positive Rate (Sensitivity) against the False Positive Rate ($1 - \text{Specificity}$) across all possible classification thresholds $\theta \in [0,1]$:

$$\text{TPR}(\theta) = \frac{\text{TP}(\theta)}{\text{TP}(\theta) + \text{FN}(\theta)}, \quad \text{FPR}(\theta) = \frac{\text{FP}(\theta)}{\text{FP}(\theta) + \text{TN}(\theta)}$$

The AUROC is the area under this curve, computed via the trapezoidal rule over all threshold values:

$$\text{AUROC} = \int_0^1 \text{TPR}(\text{FPR}^{-1}(u)) \, du$$

**Probabilistic interpretation:** AUROC equals the probability that a randomly selected malignant sample receives a higher predicted probability than a randomly selected benign sample:

$$\text{AUROC} = P(\hat{p}_1^{(\text{mal})} > \hat{p}_1^{(\text{ben})})$$

This interpretation is threshold-independent and makes AUROC the standard primary metric for imbalanced binary medical classification.

**Why AUROC for melanoma detection:** At a global malignancy rate of 5.55%, accuracy is a degenerate metric (94.45% accuracy is achievable by predicting "benign" always). AUROC evaluates the model's discriminative ability across the full operating range, independent of which threshold a clinical deployment ultimately chooses. Different deployment contexts — population screening (high specificity required) vs. high-risk patient triage (high sensitivity required) — will choose different operating points on the same ROC curve.

**AUROC $= 1.0$:** Perfect discrimination. Every malignant sample ranks above every benign sample.  
**AUROC $= 0.5$:** Random classifier. Equivalent to coin flipping.  
**AUROC $< 0.5$:** Worse than random — indicates systematic mislabeling (practically, model inversion).

#### 5.1.2 AUPRC (Area Under the Precision-Recall Curve)

For highly imbalanced datasets, **AUPRC** is a more informative metric than AUROC [Davis & Goadrich, 2006]. The Precision-Recall curve plots:

$$\text{Precision}(\theta) = \frac{\text{TP}(\theta)}{\text{TP}(\theta) + \text{FP}(\theta)}, \quad \text{Recall}(\theta) = \text{TPR}(\theta) = \frac{\text{TP}(\theta)}{\text{TP}(\theta) + \text{FN}(\theta)}$$

$$\text{AUPRC} = \int_0^1 \text{Precision}(\text{Recall}^{-1}(r)) \, dr$$

**Why AUPRC is critical for our setting:** With 5.55% positive prevalence, a model that retrieves all positives (Recall = 1.0) but predicts "malignant" for every sample achieves Precision = 0.0555 — the AUPRC rightly assigns this model a low score. AUROC, however, would assign this same degenerate model AUROC = 1.0 (all positives rank above negatives). In imbalanced regimes, AUROC can be misleadingly optimistic. AUPRC penalizes false positives heavily, reflecting the clinical cost of unnecessary biopsies.

**Baseline AUPRC:** For a random classifier on our dataset, AUPRC ≈ prevalence = 0.0555. Any useful model must substantially exceed this baseline.

#### 5.1.3 Sensitivity and Specificity

$$\text{Sensitivity} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}$$

**Clinical interpretation in melanoma screening:**

- **Sensitivity (Recall for malignant class):** The fraction of true melanomas that the model correctly flags. A low sensitivity means missed diagnoses — patients with melanoma are told "benign" and go untreated. This is the highest-stakes error type in oncological screening.

- **Specificity:** The fraction of benign lesions correctly identified as benign. A low specificity means unnecessary biopsies — patients without melanoma undergo invasive procedures. This is an economic and patient distress issue, but not life-threatening.

In the FedMelanoma system, **sensitivity is the primary clinical objective**, consistent with the focal loss design that amplifies gradient signal from malignant cases. The target operating threshold is chosen to achieve Sensitivity $\geq 0.85$ at Specificity $\geq 0.70$ — a commonly cited clinical benchmark for AI-assisted dermoscopy triage.

#### 5.1.4 F1 Score (Malignant Class)

$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$$

The F1 Score is the harmonic mean of Precision and Recall. For our imbalanced binary problem, F1 is always computed **for the malignant class specifically** — computing F1 for the benign class would be meaningless given the 17:1 imbalance.

The harmonic mean penalizes extreme imbalances between Precision and Recall more severely than the arithmetic mean. A model with Precision = 0.98, Recall = 0.02 (predicts malignant for only the 2 most obvious cases) achieves arithmetic mean = 0.50 but F1 = 0.039 — correctly reflecting its clinical uselessness despite high Precision.

#### 5.1.5 Balanced Accuracy

$$\text{Balanced Accuracy} = \frac{\text{Sensitivity} + \text{Specificity}}{2}$$

Balanced Accuracy is the arithmetic mean of the per-class recall values — equivalent to AUROC at a single threshold. Unlike standard accuracy, it is invariant to class imbalance: a degenerate all-benign predictor achieves Balanced Accuracy = $(0 + 1)/2 = 0.5$, correctly penalized compared to the misleading 94.45% standard accuracy.

In the federated monitoring context, Balanced Accuracy is a useful single-number summary that can be computed without a threshold and reported per client (using the client's local evaluation set) for cross-client comparison.

---

### 5.2 Calibration Metrics

#### 5.2.1 Expected Calibration Error (ECE)

Formally repeated from Section 3.10.4 for completeness in the evaluation framework:

$$\text{ECE} = \sum_{k=1}^{K_{\text{bin}}} \frac{|\mathcal{B}_k|}{n} \left| \text{acc}(\mathcal{B}_k) - \text{conf}(\mathcal{B}_k) \right|$$

ECE is computed on the global test set **after temperature scaling** with the calibrated $T^*$. Pre- and post-calibration ECE are both reported to quantify the calibration improvement achieved by temperature scaling. Target: post-calibration ECE $< 0.03$.

#### 5.2.2 Brier Score

$$\text{BS} = \frac{1}{n} \sum_{j=1}^{n} \left(\hat{p}_{1,j}^{\text{cal}} - y_j\right)^2$$

Brier Score is tracked alongside ECE as a unified proper scoring rule. Unlike ECE (which bins predictions and ignores within-bin variation), Brier Score evaluates calibration at the individual prediction level. After temperature scaling, a well-calibrated model on our test set (prevalence 0.0555) should achieve BS $< 0.06$.

**Skill score interpretation:** The Brier Skill Score compares the model to the naive climatological baseline $\hat{p}_{1,j} = \bar{y} = 0.0555$ for all samples:

$$\text{BSS} = 1 - \frac{\text{BS}}{\text{BS}_{\text{clim}}}, \quad \text{BS}_{\text{clim}} = \bar{y}(1 - \bar{y}) = 0.0555 \times 0.9445 = 0.0524$$

A BSS $> 0$ indicates the model outperforms the trivial predictor; BSS $= 1$ is perfect.

---

### 5.3 Federated System Metrics

#### 5.3.1 Global vs. Local Performance Gap

The **generalization gap** measures how much performance degrades when moving from a client's local data distribution to the global test distribution:

$$\Delta_{\text{gap}}^{(i)} = \text{AUROC}_{\text{global}}(\mathbf{w}^T) - \text{AUROC}_{\text{local}}^{(i)}(\mathbf{w}_i^{T,\text{local}})$$

where $\mathbf{w}^T$ is the final global model and $\mathbf{w}_i^{T,\text{local}}$ is the locally fine-tuned model of client $i$ after the last round. A large $\Delta_{\text{gap}}^{(i)} > 0$ means the global model outperforms the local model on global data — the expected direction. A negative gap ($\Delta_{\text{gap}}^{(i)} < 0$) at a particular client indicates that client has overfit locally and the federated averaging actually **hurt** that client on global evaluation — a sign of excessive client drift and a candidate for proximal regularization at that node.

#### 5.3.2 Fitzpatrick Fairness Gap

$$\Delta_{\text{fair}} = \max_{\varphi \in \Phi} \text{Sensitivity}(\varphi) - \min_{\varphi \in \Phi} \text{Sensitivity}(\varphi)$$

Evaluated on the test set, stratified by Fitzpatrick type. For the current dataset with 0.00% malignancy in Types V and VI, sensitivity is evaluated only for Types I–IV, where sufficient positive labels exist (Type I: 74 malignant; Type II: 406 malignant; Type III: 54 malignant; Type IV: 7 malignant). Type IV sensitivity will have high variance due to only 7 positive labels — confidence intervals must be reported.

The **target trajectory**: $\Delta_{\text{fair}}$ should be non-increasing over training rounds, validating the diversity compensation mechanism. An increasing $\Delta_{\text{fair}}$ is a system-level alarm indicating that the global model is diverging toward majority-phenotype performance.

#### 5.3.3 Worst-Group Accuracy

Inspired by distributionally robust optimization (DRO), the **worst-group accuracy** identifies the distributional subgroup on which the global model performs worst:

$$\text{WGA} = \min_{g \in \mathcal{G}} \text{AUROC}(g), \quad g \in \mathcal{G} = \{\text{fair-derm, fair-clinical, dark-derm, mixed-tele}\}$$

WGA is maximized by FedMelanoma's cluster-coverage selection constraint: by ensuring at least one client from each distributional cluster $\mathcal{G}$ participates per round, the global model is forced to receive gradient signal from all four distributional quadrants, preventing the WGA from collapsing on any single group.

#### 5.3.4 Communication Efficiency

$$\eta_{\text{comm}} = \frac{\text{AUROC at round } T}{\text{total bytes transmitted}}$$

This metric captures the accuracy-per-bit efficiency of the federated protocol. It rewards aggressive compression and early stopping — a model that reaches AUROC = 0.89 in 30 rounds with 3× compression is strictly more communication-efficient than one reaching the same AUROC in 50 rounds with 1× compression.

---

## 6. Comparison with Existing FL Methods

> *FedMelanoma is designed as a targeted solution to problems that existing FL algorithms address inadequately or not at all. This section provides a formal comparison along the four axes most relevant to medical FL: Non-IID tolerance, straggler handling, minority preservation, and calibration.*

### 6.1 FedAvg (McMahan et al., 2017) [2]

**What it does:** FedAvg is the foundational FL algorithm. In each round, a random subset of clients perform $E$ local SGD steps starting from the global model, and the server aggregates updates via size-weighted averaging:

$$\mathbf{w}^{t+1} = \sum_{i \in \mathcal{S}^t} \frac{n_i}{n} \mathbf{w}_i^{t+1}$$

**Limitation 1 — Non-IID sensitivity:** FedAvg was originally analyzed under the IID assumption. Li et al. [4] showed that under Non-IID data (which characterizes all realistic medical FL deployments), FedAvg's convergence is not guaranteed and in practice leads to **client drift**: local models diverge toward their local optima, and the aggregated global model oscillates. In our setting with a 35× malignancy rate spread across clients, FedAvg without modification exhibits significant convergence instability.

**Limitation 2 — No straggler handling:** FedAvg is synchronous by design. In the original protocol, the server waits for all selected clients before aggregating. Applying FedAvg naively to our K=10 client system with 40–60% straggler rates on C9–C10 would mean these clients either systematically delay every round or are simply dropped, permanently excluding their dark-skin phenotype data from the global model.

**Limitation 3 — No diversity awareness:** Client selection in standard FedAvg is uniform random. There is no mechanism to prefer clients carrying underrepresented phenotypes, ensure cluster coverage, or prevent the global model from converging toward the majority distribution.

**FedMelanoma improvement:** FedMelanoma subsumes FedAvg as the special case $\delta = 1.0$, $\gamma = 0$, $\alpha = 1$, $\beta = \gamma = 0$ (no staleness decay, no diversity bonus, reliability-only selection). Setting $\delta < 1$, $\gamma > 0$, and the full composite score with cluster constraint strictly generalizes FedAvg while inheriting its theoretical convergence guarantees (Section 3.8.2).

### 6.2 FedProx (Li et al., 2020) [4]

**What it does:** FedProx addresses Non-IID divergence by adding a **proximal regularization term** to each client's local objective:

$$\min_{\mathbf{w}} F_i(\mathbf{w}) + \frac{\mu}{2} \|\mathbf{w} - \mathbf{w}^t\|^2$$

The proximal term penalizes the local model for drifting too far from the current global model $\mathbf{w}^t$, directly bounding the gradient dissimilarity $\zeta^2$ from Section 3.1.4. FedProx provably converges under bounded Non-IID heterogeneity with an appropriate $\mu > 0$.

**Limitation 1 — Optimization-side only, no aggregation-side correction:** FedProx constrains local training but does nothing about the aggregation process. It does not distinguish between fresh and stale updates, does not account for distributional diversity during aggregation, and does not prevent minority phenotype erasure. In our setting, a telemedicine node running FedProx still has its update down-weighted by staleness with no diversity compensation.

**Limitation 2 — $\mu$ sensitivity:** The proximal coefficient $\mu$ requires careful tuning. Too small: proximal term is negligible, equivalent to FedAvg. Too large: local training is over-regularized, preventing meaningful local adaptation and reducing the benefit of federated learning over centralized training with a shared dataset.

**Limitation 3 — No straggler handling:** FedProx is synchronous, sharing FedAvg's blocking aggregation behavior.

**FedMelanoma + FedProx:** The proximal term is **compatible with and complementary to** FedMelanoma's aggregation strategy. Adding the proximal regularization to clients' local objectives would reduce client drift (Term 3 of the convergence bound) while FedMelanoma's diversity-weighted aggregation handles staleness and minority preservation. This combination represents a natural extension for high-heterogeneity deployments.

### 6.3 FedLesScan (Elzohairy, 2022) [3]

**What it does:** FedLesScan is the most direct prior work to FedMelanoma. It addresses stragglers in a serverless FL setting by selecting clients based on a **reliability score** $R_i$ alone — the EMA of completion history. Clients with low reliability are progressively de-prioritized, and the aggregation uses size-weighted averaging over arrived clients.

**Limitation 1 — Reliability without diversity:** FedLesScan's selection function $\text{score}_i = R_i$ is strictly dominated by FedMelanoma's composite score $\text{score}_i = \alpha R_i + \beta D_i + \gamma M_i$. By ignoring diversity and coverage, FedLesScan suffers the straggler-value paradox in full: high-straggler clients (low $R_i$) with rare phenotype data (high $D_i$) are progressively excluded.

**Limitation 2 — No staleness decay or diversity bonus in aggregation:** FedLesScan uses standard size-weighted aggregation once arrived clients are determined. It does not discount stale updates or amplify underrepresented client contributions.

**Limitation 3 — No cluster coverage constraint:** There is no mechanism to ensure every distributional group is represented per round.

**FedMelanoma improvement over FedLesScan:** FedMelanoma extends FedLesScan by (a) the three-component composite selection score with aging boost and fairness window, (b) the staleness-weighted aggregation with diversity bonus, and (c) the cost-matrix OT-based diversity computation. The convergence overhead of these additions is $\mathcal{O}(\gamma^2 \eta^2 \sigma^2)$ (Term 4 of Section 3.8.2) — a 9% variance increase for $\gamma = 0.3$, justified by the phenotype erasure prevented.

### 6.4 Comparison Summary Table

| Criterion | FedAvg | FedProx | FedLesScan | **FedMelanoma** |
|---|---|---|---|---|
| Non-IID convergence guarantee | Partial | ✓ | Partial | ✓ |
| Straggler tolerance | ✗ | ✗ | ✓ | ✓ |
| Staleness decay in aggregation | ✗ | ✗ | ✗ | **✓** |
| Diversity-compensated aggregation | ✗ | ✗ | ✗ | **✓** |
| Cost-matrix OT diversity scoring | ✗ | ✗ | ✗ | **✓** |
| Minority phenotype preservation | ✗ | ✗ | ✗ | **✓** |
| Cluster coverage constraint | ✗ | ✗ | ✗ | **✓** |
| Fairness window guarantee | ✗ | ✗ | ✗ | **✓** |
| Post-hoc calibration (temp. scaling) | ✗ | ✗ | ✗ | **✓** |
| Synchronous / Asynchronous | Sync | Sync | Async | **Async** |

---
## 7. System Configuration Schema

**File:** `fedmelanoma/configs/fedmelanoma_config.yaml`

```yaml
# ============================================================
# FedMelanoma v3.0 Configuration
# Aligned with preprocessed_data.csv | OT cost matrices included
# ============================================================

system:
  name: "FedMelanoma_v3"
  version: "3.0.0"
  seed: 42
  device: "cuda"
  log_level: "INFO"
  checkpoint_dir: "models/checkpoints/"
  checkpoint_interval: 5
  audit_log_dir: "logs/audit/"           # Tamper-evident audit trail

federated:
  total_rounds: 50
  num_clients: 10
  clients_per_round: 6
  quorum_fraction: 0.70
  time_window_k: 2.5                     # W_t = T̃_med + k·MAD
  min_time_window_seconds: 60
  max_time_window_seconds: 600
  fairness_window: 10                    # W_fair: mandatory selection window

math:
  noniid:
    temperature_tau: 0.5
    facility_profile_dims: 3            # K×3: φ, τ_mod, λ only
    min_samples_per_client: 100

  aggregation:
    delta: 0.85                         # δ: staleness decay
    gamma_diversity: 0.3                # γ: diversity bonus
    max_staleness_rounds: 3             # S_max

  selection:
    alpha_reliability: 0.4
    beta_diversity: 0.3
    gamma_coverage: 0.3
    R_min: 0.50
    reliability_ema_beta: 0.9
    aging_boost_eta: 0.05
    num_distributional_groups: 4

  diversity:
    method: "optimal_transport"         # "optimal_transport" | "wasserstein_1d" | "tv"
    emd_weight_fitzpatrick: 0.40        # w_φ
    emd_weight_modality: 0.35           # w_τ
    emd_weight_confirmation: 0.25       # w_λ
    sinkhorn_epsilon: 0.01              # Entropy regularization for Sinkhorn solver
    
    # Cost matrix: imaging modality (3×3)
    # Order: [dermoscopic, clinical:close-up, clinical:overview]
    cost_matrix_modality:
      - [0.00, 0.30, 0.70]
      - [0.30, 0.00, 0.45]
      - [0.70, 0.45, 0.00]
    
    # Cost matrix: confirmation type (4×4)
    # Order: [histopathology, serial_imaging, expert_consensus, single_contributor]
    cost_matrix_confirmation:
      - [0.000, 0.333, 0.667, 1.000]
      - [0.333, 0.000, 0.333, 0.667]
      - [0.667, 0.333, 0.000, 0.333]
      - [1.000, 0.667, 0.333, 0.000]

  calibration:
    enabled: true
    method: "temperature_scaling"       # Post-hoc, server-side
    calibration_set: "validation"       # Use server's held-out validation set
    num_bins_ece: 10                    # K_bin for ECE computation
    recalibrate_every_n_rounds: 5

dataset:
  csv_path: "data/preprocessed_data.csv"
  images_dir: "data/images/"
  total_samples: 9740
  malignancy_rate: 0.0555

  metadata_axes:
    - name: skin_type
      categories: ["I", "II", "III", "IV", "V", "VI"]
      encoding_dim: 6
      rare_threshold: 0.10
    - name: image_type
      categories: ["dermoscopic", "clinical: close-up", "clinical: overview"]
      encoding_dim: 3
      rare_threshold: 0.05
    - name: confirm_type
      categories:
        - "histopathology"
        - "serial imaging showing no change"
        - "single image expert consensus"
        - "single contributor clinical assessment"
      encoding_dim: 4

  total_meta_input_dim: 13              # 6 + 3 + 4

  split:
    train: 0.70
    val: 0.15
    test: 0.15
    stratify_by: ["label", "image_type"]

  confirmation_confidence_weights:
    histopathology: 1.00
    serial_imaging_showing_no_change: 0.90
    single_image_expert_consensus: 0.85
    single_contributor_clinical_assessment: 0.75

model:
  backbone: "efficientnet-b3"
  pretrained: true
  image_size: 224
  img_embedding_dim: 1536

  metadata_mlp:
    input_dim: 13
    hidden_dim: 64
    output_dim: 32
    dropout: 0.2

  fusion:
    dim: 1568
    dropout: 0.3
    num_classes: 2

training:
  optimizer: "AdamW"
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
  lr_schedule: "cosine_annealing"
  local_epochs: 5
  batch_size: 32
  grad_clip_norm: 1.0
  reset_optimizer_each_round: true      # Prevent stale momentum drift

  loss:
    type: "class_weighted_focal"
    gamma_f: 2.0
    reduction: "mean"

straggler:
  probability_model: "beta"            # Beta distribution per client tier
  high:
    clients: ["C9", "C10"]
    beta_params: [2, 3]                # Mean completion rate = 0.40
    log_normal_delay_mu: 4.09          # ln(seconds)
    log_normal_delay_sigma: 0.8
  moderate:
    clients: ["C6", "C7", "C8"]
    beta_params: [7, 3]                # Mean = 0.70
    log_normal_delay_mu: 2.77
    log_normal_delay_sigma: 0.5
  reliable:
    clients: ["C1", "C2", "C3", "C4", "C5"]
    beta_params: [19, 1]               # Mean = 0.95
    log_normal_delay_mu: 1.39
    log_normal_delay_sigma: 0.3

communication:
  compress: true
  compressor: "zstd"
  compression_level_default: 3
  compression_level_low_bandwidth: 9
  compression_level_very_low_bandwidth: 19
  bandwidth_threshold_low_mbps: 10.0
  bandwidth_threshold_very_low_mbps: 1.0
  transmit_delta_only: true
  grpc_max_message_bytes: 536870912
  mtls_enabled: true
  cert_dir: "certs/"

evaluation:
  eval_every_n_rounds: 5
  early_stopping_patience: 10
  target_operating_point:
    sensitivity_min: 0.85
    specificity_min: 0.70

  metrics:
    - auroc
    - auprc
    - sensitivity
    - specificity
    - f1_malignant
    - balanced_accuracy
    - ece
    - brier_score
    - fitzpatrick_fairness_gap
    - worst_group_accuracy
    - communication_efficiency

  alert_thresholds:
    auroc_drop: 0.03
    sensitivity_min: 0.75
    ece_max: 0.05
    fairness_gap_increase: 0.05

  baselines:
    fedavg:
      enabled: true
    fedprox:
      enabled: true
      mu: 0.01
    fedlesscan:
      enabled: true

monitoring:
  logging:
    operational: "logs/operational/"
    mathematical: "logs/metrics/"
    audit: "logs/audit/"
  drift_detection:
    enabled: true
    ks_test_alpha: 0.05
    chi2_test_alpha: 0.05
    window_size_rounds: 10

gui:
  host: "0.0.0.0"
  port: 5000
  gradcam:
    target_layer: "_blocks.25._depthwise_conv"
    alpha: 0.5
  allowed_extensions: ["jpg", "jpeg", "png"]
  max_upload_size_mb: 10
  show_calibrated_confidence: true      # Display T*-scaled probabilities
```

---

## 8. Repository Architecture

```
fedmelanoma/
│
├── server/
│   ├── server.py                    # Flower ServerApp: round controller & entry point
│   ├── aggregation.py               # FedMelanomaStrategy: staleness-weighted FedAvg + diversity bonus
│   ├── selection_engine.py          # OT-based EMD, composite scoring, cluster selection
│   ├── straggler_detector.py        # W_t = T̃_med + k·MAD; quorum trigger; buffer management
│   └── calibrator.py                # Temperature scaling: T* search via NLL minimization
│
├── client/
│   ├── client.py                    # FedMelanomaClient: Flower NumPyClient wrapper
│   ├── local_trainer.py             # Training loop: AdamW, cosine LR, E epochs, optimizer reset
│   ├── dataset.py                   # ISICDataset: image loading, one-hot encoding, augmentation
│   └── focal_loss.py                # ClassWeightedFocalLoss: γ_f=2, local α_c computation
│
├── models/
│   ├── fusion_model.py              # FedMelanomaModel: EfficientNet-B3 + MetadataMLP (13→64→32)
│   └── checkpoints/                 # Global model + T* per evaluation round
│       ├── global_model_r005.pt
│       └── calibration_T_r005.json
│
├── data/
│   ├── preprocessed_data.csv        # 9,740 records: isic_id|label|skin_type|image_type|confirm_type
│   ├── images/                      # ISIC dermoscopic image files
│   └── partition/                   # Non-IID client splits (generated at runtime)
│
├── data_prep/
│   ├── partitioner.py               # Softmax-temperature Non-IID partitioner (K×3 matrix)
│   ├── encoder.py                   # One-hot (13-d) encoding; consistency validation across clients
│   └── eda.py                       # Dataset statistics, OT cost matrices, EMD visualizations
│
├── math/
│   ├── optimal_transport.py         # LP-based OT solver + Sinkhorn approximation
│   ├── emd_scorer.py                # 3-axis EMD pipeline using OT + W_1
│   └── cost_matrices.py             # C_mod (3×3), C_conf (4×4) definitions and validation
│
├── evaluation/
│   ├── metrics.py                   # AUROC, AUPRC, Sensitivity, ECE, Brier, Fairness Gap, WGA
│   ├── calibration_eval.py          # Reliability diagram, ECE computation, Brier Score
│   ├── federated_metrics.py         # Local vs. global gap, straggler participation, comm. efficiency
│   └── baseline_comparison.py       # FedAvg, FedProx (μ=0.01), FedLesScan runners
│
├── monitoring/
│   ├── logger.py                    # Three-tier logging (operational / math / audit)
│   ├── drift_detector.py            # KS test + chi-squared for distribution drift per client
│   └── alert_manager.py             # Threshold-based alerts for AUROC, ECE, fairness gap
│
├── gui/
│   ├── app.py                       # Flask inference: upload → calibrated predict → heatmap
│   ├── gradcam.py                   # Grad-CAM on EfficientNet-B3 final conv block
│   ├── static/
│   └── templates/
│       ├── index.html
│       └── result.html
│
├── configs/
│   └── fedmelanoma_config.yaml      # Master configuration (Section 7)
│
├── certs/
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
│   ├── test_aggregation.py          # Staleness weight correctness, normalization
│   ├── test_focal_loss.py           # Loss values, class weight computation
│   ├── test_partitioner.py          # Non-IID distribution statistics verification
│   ├── test_ot_emd.py               # OT solver correctness; cost matrix symmetry; edge cases
│   ├── test_calibration.py          # Temperature scaling monotonicity; ECE computation
│   ├── test_selection.py            # Cluster coverage, aging boost, fairness window enforcement
│   └── test_drift_detector.py       # KS test sensitivity; false positive rate under stable distributions
│
├── requirements.txt
└── README.md
```

---

## 9. References

[1] American Cancer Society, "Cancer Facts & Figures 2023," Atlanta, 2023.

[2] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, "Communication-Efficient Learning of Deep Networks from Decentralized Data," in *Proc. AISTATS*, 2017, pp. 1273–1282. *(FedAvg — foundational FL algorithm; size-proportional aggregation)*

[3] M. Elzohairy, "Mitigation of Stragglers in Serverless Federated Learning (FedLesScan)," M.Sc. Thesis, Technical University of Munich, 2022. *(Straggler-aware selection without diversity compensation — direct prior work)*

[4] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Smola, and V. Smith, "Federated Optimization in Heterogeneous Networks (FedProx)," in *Proc. MLSys*, 2020. *(Proximal regularization for Non-IID convergence — baseline comparison)*

[5] M. Tan and Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," in *Proc. ICML*, 2019, pp. 6105–6114. *(Model backbone — compound scaling justification)*

[6] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal Loss for Dense Object Detection," in *Proc. ICCV*, 2017, pp. 2980–2988. *(Focal loss derivation and $\gamma_f = 2$ default)*

[7] C. Xie, S. Koyejo, I. Gupta, "Asynchronous Federated Optimization," in *NeurIPS Optimization Workshop*, 2019. *(FedAsync — staleness-weighted aggregation theoretical framework)*

[8] P. Kairouz et al., "Advances and Open Problems in Federated Learning," *Foundations and Trends in Machine Learning*, vol. 14, no. 1–2, pp. 1–210, 2021. *(Comprehensive FL theory survey)*

[9] S. Caldas et al., "LEAF: A Benchmark for Federated Settings," in *NeurIPS Workshop on Federated Learning*, 2019. *(Non-IID simulation and temperature-based partitioning)*

[10] R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," in *Proc. ICCV*, 2017. *(Clinical interpretability — attention heatmaps)*

[11] C. Villani, *Optimal Transport: Old and New*, Springer, 2008. *(EMD/Wasserstein distance theoretical foundation)*

[12] International Skin Imaging Collaboration (ISIC), "ISIC Archive," [Online]. Available: https://www.isic-archive.com. *(Source dataset)*

[13] M. Cuturi, "Sinkhorn Distances: Lightspeed Computation of Optimal Transport Distances," in *Proc. NeurIPS*, 2013, pp. 2292–2300. *(Entropy-regularized OT; Sinkhorn algorithm for fast EMD approximation)*

[14] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On Calibration of Modern Neural Networks," in *Proc. ICML*, 2017, pp. 1321–1330. *(Temperature scaling; ECE metric; calibration of deep classifiers — foundational calibration reference)*

[15] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, "ImageNet: A Large-Scale Hierarchical Image Database," in *Proc. CVPR*, 2009, pp. 248–255. *(Normalization statistics [0.485, 0.456, 0.406] source; pretrained EfficientNet-B3 initialization)*

[16] L. Zhu, Z. Liu, and S. Han, "Deep Leakage from Gradients," in *Proc. NeurIPS*, 2019. *(Gradient inversion attack motivation for mTLS enforcement)*

[17] J. Davis and M. Goadrich, "The Relationship Between Precision-Recall and ROC Curves," in *Proc. ICML*, 2006, pp. 233–240. *(AUPRC theoretical justification for imbalanced classification)*
