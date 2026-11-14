# FedMelanoma ResNet-50 Novelty: audit, ablation design and execution plan

Branch `resnet50-novelty`, commit `dd013b4`. Audit date 2026-11-14.
No blueprint PDF is in the repository, so "claimed" below means the README, `configs/experiment_config.yaml`
and the design docstrings inside the modules. Source code is the authority throughout.

## 1. Actual execution flow

```
tools/run_supervised.py <run_id>              restarts on crash, passes --resume
└─ runtime/run_experiment.py main()           (L100) seeds python/numpy/torch with cfg seed
   ├─ server/server.py server_fn()            (L181) registry -> HospitalConfig, fresh model weights
   │   └─ FedMelanomaStrategy(...)            (L251) rng = np.random.default_rng(seed)   (L259)
   │       ├─ compute_normalized_diversity_priors -> emd.py, D_i fixed for the whole run (strategy L247)
   │       ├─ Selector()                      default constants (strategy L270)
   │       └─ StragglerManager(rng=self._rng) SAME rng as selection (strategy L271)
   ├─ runtime/in_process.py build_federation  (L87) 10 clients, ONE shared model object, cid "00".."09"
   ├─ Server.set_max_workers(1)               (run_experiment L135) clients train one after another on 1 GPU
   └─ for t in rounds:                        (L161)
        server.fit_round
          configure_fit      strategy L397   Boltzmann selection of 6 from non-buffered clients
          client.fit         client.py L264  load global, NEW AdamW, 5 epochs focal loss, return weights
          aggregate_fit      strategy L496   simulate response -> manage_window -> c_i -> w update
                                             -> update R_i, N_eff -> save global_round_XXXX.pth
        server.evaluate_round
          configure_evaluate strategy L776   ONLY the 6 clients selected this round
          client.evaluate    client.py L328  loss = 1 - ROC-AUC on that client's test split
          aggregate_evaluate strategy L793   example-weighted mean; best_global_model.pth by roc_auc
        save resume_state.pkl (atomic, runtime/state.py)
```

Config actually loaded: `configs/experiment_config.yaml` via `runtime/config.py`. Keys used: experiment.seed,
run_id; data.*; federation.{num_server_rounds, clients_per_round, local_epochs}; local_training.*;
runtime.*. `model.*` and `reference_constants.*` are documentation only: nothing reads them.

## 2. Component audit (A–K)

| | Component | What the code does | Where | Invoked? |
|---|---|---|---|---|
| A | Model | torchvision ResNet-50 (ImageNet V2 weights, fc=Identity) 2048-d + MetadataMLP(13→64 BN→32) 32-d, concat, dropout 0.3, Linear(2080→2). ~25.6 M params, all trained | models/model.py L36–87 | yes |
| B | FL architecture | Flower `Server` with an in-process ClientProxy; 10 logical clients share one model object; strictly sequential | runtime/in_process.py, run_experiment.py L134–135 | yes |
| C | Aggregation | c̃_i = (n_i/n_A)·δ^{s_i}·(1+γD_i), normalised; w ← w + Σ c_i (w_i − w_current) | server/aggregation.py L226; strategy L592–624 | yes |
| C' | FedAvg weights | n_i/n | aggregation.py L182 | **no** (defined, never called) |
| D | Client selection | score = R·(1+γ·D/(1+λN_eff)); Boltzmann τ=0.2, 6 draws without replacement; buffered clients excluded | selector.py L178, L239; strategy L450–460 | yes |
| D' | Greedy top-K | | selector.py L317 | **no** |
| D'' | R_i / N_eff updates | EMA ρ=0.85 of δ^s; N_eff += δ^s; only for selected clients | selector.py L340, L392, L436; strategy L636–644 | yes |
| — | Straggler simulation | per selected client: Beta(tier) completion → Bernoulli failure; LogNormal(tier) latency; late if > W_t | straggler.py L209 | yes, **after** the client already trained |
| — | Adaptive window | W_t = max(median + 2.5·MAD, 2 s) over last 10 simulated latencies | straggler.py L500 | yes (only reclassifies simulated latencies) |
| — | Straggler buffer | late update buffered with s=1, drained next round with δ^1, expires at age 3 | straggler.py L362, L527 | yes |
| — | Quorum trigger | computed and printed | straggler.py L362 (end) | **logged only, never gates aggregation** (tests/test_02 L344) |
| — | StragglerAggregationWindow | a second copy of the window/buffer logic | aggregation.py L333 | **no** |
| E | Personalization | none: every client loads the global model and returns full weights | client.py L294 | **absent** |
| F | Data / partition | 10 hospital folders, each with fixed metadata_train/test CSVs (image-level split). Tiers assigned round-robin by sorted hospital name (reliable / moderate / high_straggler) | hospital_registry.py L145; dataset.py | yes |
| G | Local training | AdamW lr 1e-4, wd 1e-4, rebuilt every round (state reset), 5 epochs, batch 20, fp32, no augmentation, workers 0 | client.py L296–312; local_trainer.py | yes |
| H | Evaluation | per round: 1−AUC on test splits of the ≤6 selected clients (set changes every round). Final: evaluate.py on the pooled test sets of all 10 hospitals | strategy L776; evaluation/evaluate_checkpoint.py | yes |
| I | Loss | class-weighted focal loss γ_f=2, inverse-frequency weights computed locally per client | focal_loss.py L105, L279; client.py L170 | yes |
| J | Communication | full fp32 state_dict (incl. BN buffers) both ways, every round; no compression or quantisation | client.py L238 | yes; **no communication optimisation exists** |
| K | Checkpoint / logs | global_round_XXXX.pth every round (written twice), best_global_model.pth, atomic resume_state.pkl, jsonl op/math/audit/straggler logs, resource CSV | monitoring/checkpointing.py; runtime/state.py | yes |

## 3. Discrepancy report

| # | Claim (README / docstrings / config) | What the code implements | Location | Status |
|---|---|---|---|---|
| 1 | "straggler simulation and staleness-aware aggregation across 10 hospital clients" | All 10 clients train synchronously in one process. Failures, latencies and lateness are random draws made **after** each selected client has already trained. Staleness never comes from real asynchrony | strategy L569–583 | Different (simulated) |
| 2 | Aggregation triggers on (A) quorum or (B) window expiry | Quorum is computed and logged; aggregation always proceeds with whatever is on time | straggler.py manage_window; test_02 L344 | Partially implemented (cosmetic) |
| 3 | best_global_model chosen by "validation ROC-AUC" | No validation split exists. Selection uses the **test** split of the clients selected that round, then is reported on the pooled test set | checkpointing.py L15, L48; strategy L892 | Different (test-set model selection) |
| 4 | Per-round "global" evaluation | Evaluates only the ≤6 selected clients, so the evaluation population changes each round | strategy L776–786 | Different |
| 5 | Diversity, reliability, staleness "unified δ" | True: δ=0.9 in R_i, N_eff and c_i | selector.py, aggregation.py | Matches |
| 6 | Boxed update w^{t+1} = w^t + Σ c_i (w_i^{t+1} − w^t) | For a drained straggler, w_i was trained from an older global, but the delta is taken against the current global | strategy L603–624 | Ambiguous / differs for stale updates |
| 7 | `reference_constants` in the YAML | Not read by any code; modules use their own constants | config.yaml comment; strategy L270–271 | Documentation only (as the file says) |
| 8 | FedAvg baseline, greedy selection, StragglerAggregationWindow "for ablation" | Present but never invoked | aggregation.py L182, L333; selector.py L317; server.py L122–155 | Present, unused |
| 9 | Personalization / communication optimisation / adaptive LR | Not implemented anywhere | — | Missing (so not ablated) |
| 10 | Hospital tiers are a "documented runtime assumption" | Round-robin by alphabetical hospital name, so tier is unrelated to the data | hospital_registry.py L145 | Matches docstring; worth stating in the paper |
| 11 | Selection and straggler simulation independent | One shared numpy Generator, so changing selection changes every later failure / latency draw | strategy L271 | Coupled |
| 12 | Test set measures generalisation | `results/resnet50-seed42/leakage_analysis.json`: 1,822 / 1,952 test rows are from patients seen in training, 1,332 are lesions seen in training. AUC 0.990 on seen patients vs 0.885 on the 130 unseen-patient rows | hospital_partitions splits | **Data leakage (image-level split)** |

## 4. Ablation design (one factor at a time)

Training budget for every arm: 30 rounds, 6 clients per round, 5 local epochs, batch 20, AdamW 1e-4,
identical partitions. Seeds 42, 43, 44; the same seeds in every arm (paired design).

| ID | Changes exactly one thing | How (no reference file edited) | Question |
|---|---|---|---|
| E00 | — full method | — | Reference point |
| E01 | Selection → uniform random 6 of non-buffered | Selector subclass, `boltzmann_selection` = uniform draw; R_i/N_eff still updated | Does score-based selection help at all? |
| E02 | Selection score without diversity (γ_sel = 0 → score = R_i) | `Selector(gamma=0.0)` | Is the diversity term in selection what matters? |
| E03 | Aggregation without diversity bonus (γ_agg = 0) | set `server.strategy.AGGREGATION_GAMMA = 0` at runtime | Does (1+γD_i) in c_i help? |
| E04 | Aggregation → FedAvg weights n_i/n (no staleness discount, no diversity) | route to unused `fedavg_baseline_weights` | Does the c_i formula as a whole help? |
| E05 | No straggler buffer: late updates discarded (sync-FL behaviour); R_i/N_eff updates unchanged | StragglerManager subclass | Is buffering stale updates worth it? |
| E06 | Loss γ_f = 0 (class-weighted CE instead of focal) | config `local_training.focal_gamma: 0` | Does the focal term help? |
| R01 | Reference: FedAvg (uniform selection + n_i/n + drop late) | combination | Total gain over the textbook baseline (not an ablation) |

Not ablated because the code does not contain them: personalization, communication optimisation,
adaptive learning rate. The adaptive window is part of the simulated environment (it only
reclassifies simulated latencies), so it is held fixed rather than ablated.

Metrics (from actual outputs): final-round pooled-test ROC-AUC (primary), PR-AUC, F1 / sensitivity /
specificity / balanced accuracy at 0.5, per-hospital AUC, Fitzpatrick sensitivity gap, per-round
curves, clients aggregated per round, bytes transmitted (2 × params × 4 B × clients), wall time,
GPU peak memory. Accuracy is reported but not used for conclusions (5.6 % positives).

Statistics: per arm mean ± SD over 3 seeds; paired differences vs E00 per seed, mean and 95 % CI
(t, n=3); paired t-test and per-seed sign shown, and patient-clustered bootstrap CI on the pooled test
set for each run. With n=3 seeds, small differences will be descriptive, not significant; the report will say so.

## 5. Harness changes that need approval

| # | Change | Why | Effect on method |
|---|---|---|---|
| H1 | Common random numbers for the simulated environment: failure / latency for (seed, round, client) drawn from its own generator `default_rng([seed, round, client])` | Today one RNG is shared (item 11), so arms see different failures and the comparison is confounded | Baseline no longer bit-identical to the old seed-42 run; method logic unchanged |
| H2 | Clients whose CRN outcome is "failed" skip local training (their weights are discarded anyway) | ≈ 22 % less GPU time (1.3 of 6 clients fail per round in the seed-42 run) | Identical aggregation; only the torch shuffling stream differs |
| H3 | Mixed precision (torch.autocast fp16 + GradScaler) for all arms | T4 tensor cores, roughly 1.5–2.5× faster (to be measured) | Numerics change; measured on 1–2 rounds against fp32 first; applied to every arm or none |
| H4 | Primary metric = round-30 model; per-round pooled evaluation on all 10 test sets (read-only) | Fixes items 3 and 4 for the ablation conclusions | best-checkpoint numbers still saved, labelled "test-selected, optimistic" |
| H5 | Keep only last + best global checkpoint and resume state per run | 30 × ~100 MB × 2 writes per run otherwise | none |
| H6 | DataLoader workers 4 on Kaggle (Linux) | config sets 0 for Windows | none |

## 6. Resource plan

Measured so far: the 50-round seed-42 run took 18,562 s of training (≈ 371 s/round) on the laptop
GPU, peak 2.6 GB GPU memory. Kaggle session now: 2 × Tesla T4 (15 GB each), 4 CPU, 31 GB RAM.

Two runs fit side by side (one per T4, `CUDA_VISIBLE_DEVICES`), since each needs < 3 GB GPU and
~3 GB RAM. Running two processes on one T4 is not planned unless measured to help.

Per-run estimate for 30 rounds (to be replaced by a measured 1-round benchmark): fp32 ≈ 2.5–4 h;
with H2 ≈ 2–3 h; with H2 + H3 ≈ 1–2 h.

| Batch | Runs | Session-hours on T4×2 (2 runs in parallel) |
|---|---|---|
| Benchmark (1 round fp32 vs AMP) | — | ≈ 0.3 |
| Batch 1: E00, E01, E03, E04, E05, E06 × 3 seeds | 18 | ≈ 9–27 depending on H2/H3 |
| Batch 2: E02, R01 × 3 seeds | 6 | ≈ 3–9 |

Kaggle's weekly GPU quota is shown on the account page (third-party sources say ~30 h/week, sessions up
to 12 h; the number of concurrent GPU sessions is not documented). Quota is counted per session, so
T4×2 with two runs gives two runs per quota hour.

## 7. Execution model

A. Independent ablation runs in parallel: yes, one run per T4, and more sessions only if the account
   allows concurrent GPU sessions (checked, not assumed).
B. Parallel clients inside one FL run: no. Clients share one model object and train in selection
   order, which the reproducibility of the run depends on.
C. Distributed training across machines: not used.

Two ways to run, both with checkpoint resume every round:
1. Kaggle API batch kernels (`kaggle kernels push`), one kernel per GPU pair, background, outputs
   persisted as kernel output and pulled with `kaggle kernels output`. Needs a Kaggle API token in
   `~/.kaggle/kaggle.json` on the laptop. No credentials are present now.
2. The current SSH-connected interactive session: works today, but needs the browser tab open, ends
   at the session limit, and results must be copied off after every round (rsync to the laptop).

## 8. Implementation plan (new files only; the 22 reference files stay byte-identical)

```
ablation/
  experiments.yaml          registry: id, overrides, seeds
  config_generator.py       per-run config + sha256 config hash + split hash
  variants.py               Selector / StragglerManager subclasses, aggregation hooks (E01–E05, H1, H2)
  run_one.py                wraps runtime/run_experiment with a variant, AMP flag, pooled per-round eval,
                            results/raw/<EXP>/seed_<s>/{config.yaml, metrics.csv, metrics.json, training.log}
  scheduler.py              queue -> GPU slots, skips completed, resumes from resume_state.pkl
  kaggle_launcher.py        builds kernel folders, pushes, records submission
  status_monitor.py         QUEUED/SUBMITTED/RUNNING/COMPLETED/FAILED/INTERRUPTED/RETRY_PENDING,
                            OOM / traceback / missing-round / NaN checks
  result_collector.py       pulls outputs, validates completeness, never overwrites (run ids are unique)
  analysis/                 aggregate_metrics, statistical_analysis, ablation_plots, generate_report
manifests/experiments.json, manifests/experiment_status.json
```

Order: build + unit-test the variants on CPU (schedule dry-run per arm shows each arm differs only in its
factor) → 2-round GPU smoke test per arm → benchmark → Batch 1 → analysis → Batch 2 → report.
