# FedMelanoma Novelty vs standard FedStrag — what each one actually does

Written 2026-11-20 from the code and run logs (pre-experiment audit and corrections), not from the blueprints.
Novelty = `RESNET 50_ fed novelty/RESNET50_NOVELTY MODEL`, run `resnet50-seed42` (finished, not modified).
FedStrag = this folder, run `resnet50-fedstrag-final-seed42`.

## 1. Shared, byte-identical (tests/test_00, configs/reference_manifest.json)
ResNet-50 (ImageNet V2) + metadata MLP fusion, class-weighted focal loss, AdamW (lr 1e-4, wd 1e-4), 5 local epochs,
batch 20, the same 10 hospitals with the same sorted mapping, the same Train/Test CSVs and pixel cache, the same
initial model (hash `330768fe…`), seed 42, the same pooled 1,952-image evaluator, and the same straggler/failure
environment code (`simulate_response`: tier Beta completion → Bernoulli failure → tier LogNormal delay; failure
probabilities ≈ 5 % / 30 % / 60 %; the client trains first, then the update may be lost). Calling the client
directly (FedStrag) returns byte-identical weights to Novelty's Flower proxy route (audit check, CPU).

## 2. What Novelty's waiting mechanisms do IN THE CODE
* **Adaptive window W_t acts as a latency classifier, not as waiting.** Each selected client gets one simulated
  latency; it is labelled *on time* if latency ≤ W_t (the window computed from earlier rounds; round 1 uses
  W_min = 2.0 s) and *late* otherwise. Nothing waits: training runs to completion for every selected client, and the
  label only decides whether the update is aggregated now or buffered. W_t = max(median + 2.5·MAD, 2 s) over the
  last 10 latencies (run: mean 14.3 s, range 4.5–43.9 s).
* **Quorum is logged but does not trigger aggregation.** ⌈0.7 × selected⌉ (5 of 6) is computed and written to the
  logs (met in 27 of 50 rounds); every round aggregates its on-time + drained updates whether or not quorum was met.
  The blueprints (`docs/FedMelanoma_Blueprint_v4.md`, `docs/FedMelanoma_Master_Blueprint.md`, `v2`) describe a
  quorum *trigger*; that is the design intent, not the executed behaviour.
* **Late updates** are buffered (one per client; a newer one overwrites it) and applied in the next round with
  staleness s = 1 however late they really were; a buffered client is not selectable until drained; S_max = 3 expiry
  exists but never fired in the run (72 late: 70 drained, 2 still buffered at the end).

## 3. Side by side

| | Novelty | FedStrag |
|---|---|---|
| Who trains | 6 of 10, Boltzmann over R_i·(1 + γ·D_i/(1+λN_eff)) | all 10, every broadcast |
| Client state | reliability EMA R_i, exposure N_eff, static EMD diversity D_i | none |
| Time model | no clock; latency vs window label per round | event-driven virtual clock; poll every 4 s |
| Round / iteration ends | always after one synchronous round | first 4 s poll with ≥ 1 arrived update |
| Late update | buffered, applied next round with s = 1 | arrives at its real time; applied then with its true staleness |
| Several pending updates per client | no (one buffered, overwritten) | yes (FIFO backlog, oldest first) |
| Discard by age | yes, S_max = 3 (did not occur) | never |
| Weight | n_i · 0.9^s · (1 + D_i), normalised; w + Σ c_i Δ_i | W·D with W = 1/(s+1), normalised; replaces global |
| Failure | update lost; R_i drops (quality 0) → selected less | update lost; no penalty |
| Time used by a failed job | none (no clock) | one tier LogNormal job duration |
| Per-round evaluation set | the 6 selected clients (incl. failed) | clients aggregated in that iteration |
| Primary checkpoint | round 50 (`results/resnet50-seed42/final_round50/`) | iteration 50 (`results/<run>/final/`) |

## 4. Accounting (Novelty: run logs; FedStrag: deterministic replay of the full run's schedule)

| | Novelty | FedStrag |
|---|---|---|
| Local trainings | 299 | 243 |
| Failed attempts (updates lost) | 65 | 27 |
| Applied updates | 232 (162 on time + 70 drained) | 208 |
| Pending at the end, excluded | 2 buffered late updates | 8 in flight + 2 failing jobs; 257 queued broadcasts never started |
| Updates applied from high-straggler hospitals 2, 5, 8 | 27 (11, 4, 12; counted from `aggregation_c_i` in the math log) | 1 (0, 0, 1) |
| Virtual time | none (sum of windows ≈ 713 s) | 204 s |

Neither count is equalised: each method spends and loses work in its own way.

## 5. Consequences to state when reporting
1. FedStrag's fixed 4 s poll is far shorter than Novelty's typical window; with 50 iterations FedStrag covers 204
   virtual seconds, so the slowest tier (median 60 s) almost never contributes. This follows from the paper's
   protocol with the official 4 s poll, not from a tuning choice.
2. Random numbers: Novelty used one generator shared by selection and the environment, so its exact draws cannot be
   replayed; FedStrag uses per-client streams (7921, plus child stream for failed-job durations). Same distributions,
   different numbers.
3. Neither method has a validation split; "best" checkpoints are selected on test partitions and are secondary only.
4. Hardware: Novelty trained on an RTX 2050 (PyTorch 2.2.1), FedStrag on an H100 MIG slice (PyTorch 2.14). Both start
   from the identical initial model (hash `330768fe…`; FedStrag loads it from `initial/novelty_initial_model.pkl`
   because PyTorch 2.14 builds `06b67e7e…` from seed 42), but GPU training is not bit-identical across machines.

## 6. Results (added 2026-11-20 after the full run)
FedStrag run `resnet50-fedstrag-final-seed42`: 50 iterations in one attempt on the H100, 1.7 h, 02:51 UTC
20 Nov 2026. Accounting exactly as the precomputed schedule (243 jobs, 27 failed, 208 applied, 8 in flight + 2
failing at the end, 0 unaccounted). Primary checkpoints, full pooled test set (1,952 images, 109 malignant),
threshold 0.5:

| Metric | FedStrag (iteration 50) | Novelty (round 50) |
|---|---|---|
| Accuracy | 0.9483 | **0.9570** |
| Balanced accuracy | **0.8518** | 0.8348 |
| Precision | 0.5260 | **0.5984** |
| Recall | **0.7431** (81 TP, 28 FN) | 0.6972 (76 TP, 33 FN) |
| Specificity | 0.9604 (73 FP) | **0.9723** (51 FP) |
| F1 | 0.6160 | **0.6441** |
| ROC-AUC | **0.9512** | 0.9474 |
| PR-AUC | **0.6915** | 0.6899 |
| ROC-AUC 95 % interval (patient-clustered bootstrap, 637 patients) | 0.918–0.977 | 0.913–0.972 |

Reading: the two are close. FedStrag ranks slightly better (ROC-AUC, PR-AUC) and finds 5 more melanomas; Novelty
raises 22 fewer false alarms and has the higher precision, F1, accuracy and specificity. The ROC-AUC intervals
overlap almost entirely, so no ranking can be claimed from one seed. Secondary, test-selected checkpoints (not for
reporting): FedStrag ROC-AUC 0.9705, Novelty 0.9768. For context, FedLesScan in the same environment scored ROC-AUC
0.9756 at round 50 (interval 0.961–0.985), but started from the server's default initial model, not Novelty's.

Sources: `results/resnet50-fedstrag-final-seed42/final/metrics.json`, `.../leakage_analysis.json`,
`RESNET 50_ fed novelty/RESNET50_NOVELTY MODEL/results/resnet50-seed42/final_round50/metrics.json` and its
`leakage_analysis.json`.
