# Six federated ResNet-50 models on the Dirichlet-GMM partitioning

This branch holds the code, run scripts and results for six federated learning methods trained on the
Dirichlet-GMM hospital partitioning (10 hospitals) on the DGX, November 2026.

| Folder | Method | Source branch | Source commit |
|---|---|---|---|
| `models/fedavg` | FedAvg (textbook) | `resnet50-fedavg-textbook` | `624c87a` |
| `models/fedprox` | FedProx (textbook) | `resnet50-fedprox-textbook` | `5414729` |
| `models/novelty` | Novelty method | `resnet50-no-confirm-type` | `c474a6b` |
| `models/fedlesscan` | FedLesScan | `resnet_50_fedlesscan` | `e8c625c` |
| `models/fedasync` | FedAsync | `resnet50-fedasync` | `61ee7b9` |
| `models/fedstrag` | FedStrag | `resnet_50_fedstrag` | `90c52e2` |

Each `models/<name>/` folder is the code exactly as it ran: the source commit plus the local changes listed in
`provenance/<name>.patch` (and `provenance/<name>.untracked` for new files). The original branches are unchanged.

## What was changed from the source branches
- **No `diagnosis_confirm_type`** in the metadata (9 instead of 13 features), because it leaks the label. This is the same patch as on `resnet50-no-confirm-type` (`scripts/noconf.patch`), plus `drop_last` for a one-row last batch (`scripts/droplast.patch`).
- **Cross-platform `runtime/resources.py`** (the Windows-only version crashed on Linux), taken from FedLesScan.
- **`evaluate_checkpoint.py`** sets the hospital names of the new partitioning.
- **FedStrag**: a new initial model, since the stored one had 13 input features.
- **FedAsync**: client failures are simulated the same way as in the other straggler setups (FedStrag's failure environment, same random stream). Lost updates are not applied or retried; 300 dispatches, 45 lost, 255 applied. Tests: `tests/test_05_failures.py`, `tools/dry_run_schedule.py`.

## Setup
- Data: Dirichlet-GMM partitioning, 9,549 training and 2,393 test images (417 malignant), 10 hospitals. The data itself is not in this repository.
- 50 rounds (FedAsync: 300 dispatches; FedStrag: 50 iterations), seed 42, all six trained in parallel on one H100 MIG slice.
- Evaluation on the pooled test set at threshold 0.5, for the final and the best checkpoint.

## How to run
`scripts/setup_six.py` prepares the six folders, `scripts/run_all.sh` trains and evaluates all six (`scripts/run_one.sh` for one),
`scripts/prune_checkpoints.py` keeps disk use bounded. After training:
`scripts/compute_results.py` (metrics and fairness), `scripts/extra_analysis.py` (patient-level bootstrap, DeLong, McNemar, Holm),
`scripts/threshold_best.py` (operating points), `scripts/ablation_results.py` (Novelty ablations). Paths inside the scripts point to the DGX layout (`/workspace/...`).

## Results (final checkpoint, pooled test set, threshold 0.5)

| Model | ROC-AUC | Recall | Fairness gap (6 skin types) |
|---|---|---|---|
| FedAvg | 0.970 | 0.823 | 0.273 |
| FedProx | 0.959 | 0.839 | 0.227 |
| Novelty | 0.969 | 0.825 | 0.250 |
| FedLesScan | 0.969 | 0.839 | 0.227 |
| FedAsync | 0.965 | 0.799 | 0.250 |
| FedStrag | 0.952 | 0.803 | 0.250 |

- After Holm correction over the 15 pairs, FedStrag has a significantly lower AUC than FedAvg, Novelty and FedLesScan. FedProx has higher recall than FedStrag (McNemar). Other differences are not significant.
- The fairness gap depends mostly on skin type I: types IV-VI have only 6 malignant test images, and every model gets all of them right.
- Ablations of the Novelty method (seed 42, 30 rounds): no component shows a measurable AUC benefit after Holm correction.
- Single seed; see `results/extra_summary.md` for confidence intervals.

## Where things are
- `results/`: summary tables for all six models (`six_models_*`), statistical tests (`extra_*`), operating points (`threshold_best/`), Novelty ablations (`ablations/`).
- `models/<name>/run_outputs/{final,best}/`: per-model metrics, classification report, ROC/PR curves, confusion matrix, per-hospital and per-skin-type metrics.
- `provenance/`: source commits (`heads.txt`) and the exact diff of each model folder against its source commit.

Not included: checkpoints, image caches, training logs, per-image predictions and the partitioned data.
