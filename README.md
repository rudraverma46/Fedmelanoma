# Novelty model: hyperparameter tuning on the Dirichlet-GMM partitioning

Code and scripts for tuning the Novelty federated ResNet-50 model on the Dirichlet-GMM hospital partitioning
(10 hospitals, no `diagnosis_confirm_type`), run on the DGX in November 2026. Results are not stored in this branch.

## Layout
- `code/`: the Novelty model with its ablation harness, copied from branch `resnet50-no-confirm-type` at commit `c474a6b`
  (folder `RESNET50_NOVELTY MODEL`). Only five files differ from that commit; see `provenance/code_vs_source.diffstat` and `.patch`:
  - `ablation/hpo_trials.json` (new): the hyperparameters of every configuration.
  - `ablation/common.py`: reads the trial's learning rate, weight decay, batch size, local epochs and clients per round into the run config.
  - `ablation/run_one.py`: applies the selector, aggregation and straggler settings and the diversity weights after the variant is built, and logs them (`hpo_override` audit event).
  - `ablation/experiments.yaml`: registers the trial names (H00-H11, F01-F04, C00-C01) with their focal γ.
  - `models/model.py`: dropout can be set through `FEDMEL_DROPOUT`.
- `scripts/`: setup, launch, monitoring and report scripts (paths point to `/workspace/novelty_hpo` on the DGX).
- `provenance/`: source commit and the exact diff of `code/` against it.

## Tuned parameters (19)
- Local training: learning rate, weight decay, batch size, local epochs, dropout, focal γ.
- Federation: clients per round.
- Client selection: τ (temperature), γ (diversity amplification), λ (exposure attenuation), ρ (reliability decay).
- Staleness discount δ (set in the selector, straggler manager and aggregation together).
- Aggregation: diversity amplification γ.
- Straggler handling: s_max, quorum fraction, MAD k, w_min, history length.
- Diversity: Fitzpatrick weight (modality weight = 0.75 − Fitzpatrick weight).

Kept fixed: hospitals and data, architecture, optimizer (AdamW), simulated straggler environment.

## Procedure and scripts
1. **Validation split**: `setup_hpo.py` holds out 15% of training patients (stratified, seed 42) as a validation set, so the test set is not used for selection.
2. **Stage 1 search**: `setup_hpo_v2.py` writes 12 configurations (H00 = default, H01-H11 random draws, seed 2026). `stage1.sh` (seed 42) and `stage1_seeds.sh` with `run_trial.sh` (seeds 43, 44) train them for 12 rounds. `select_trials_v3.py` ranks them by mean validation ROC-AUC over the last 3 rounds and 3 seeds.
3. **Stage 2 test**: `stage2_v3.sh` retrains the top two and the default on the full training data, 30 rounds, seeds 42-44, evaluated on the test set. `final_report.py` computes the metrics, patient-level bootstrap CIs, DeLong and McNemar tests and skin-type fairness gaps.
4. **One-factor follow-up**: `setup_followup.py` + `followup.sh` apply four shared settings one at a time to the default (F01-F04).
5. **Combined check**: `setup_combo.py` + `combo.sh` run the default with aggregation γ 0.5 and MAD k 2.0 together (C01) next to a fresh copy of the default (C00); `final_report_combo.py` compares them.

Helpers: `verify_hooks*.py` check from the audit logs that every requested setting was applied; `pruner*.py` keep only the latest checkpoints during training; `status.py` prints progress.
`setup_hpo.py`, `select_trials_v2_unused.py`, `stage2_v2_unused.sh` and `pruner.py` belong to an earlier 6-parameter version that was replaced; they are kept for the record.

## Running it
Environment variables used by the scripts: `FEDMEL_DATA`, `FEDMEL_CACHE`, `FEDMEL_RAW_DIR`, `FEDMEL_ROUNDS`, `FEDMEL_WORKERS`, `FEDMEL_EXECUTION_ID`.
Each run is started with `python3 -m ablation.worker --gpu 0 --runs <config>:<seed>` from `code/`.
