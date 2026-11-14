# Ablation runbook: start, monitor, resume

Code: branch `resnet50-novelty-ablation` of github.com/rudraverma46/Fedmelanoma, folder `RESNET50_NOVELTY MODEL`.
Data: Kaggle dataset `visheshkumarbansal/fedmel01` (mounts at `/kaggle/input/datasets/visheshkumarbansal/fedmel01/hospital_partitions`).
Plan and audit: `ablation/AUDIT_AND_PLAN.md`. Experiments: `ablation/experiments.yaml`.

## Where state lives (nothing important lives only in a Kaggle session)

| What | Where | Survives a Kaggle disconnect? |
|---|---|---|
| Code | GitHub branch, pinned commit per job | yes |
| Job list, chain progress | `manifests/jobs.json` (laptop) | yes |
| Per-run state while a job runs | `results/raw/<exp>/seed_<s>/status.json` + `checkpoints/<run_id>/resume_state.pkl` in the job's `/kaggle/working` | yes, saved as the job's output; the next job of the chain copies it in |
| Finished runs | `results/raw/<exp>/seed_<s>/` (DONE.json, metrics, evals, logs, final + best checkpoint) | yes |
| Collected results | `results/raw` on the laptop after `collect` | yes |

A run resumes from its last completed round (atomic `resume_state.pkl`, RNG states included). A run with
`DONE.json` is never run again. A changed config gives a new run id, so it can never resume or overwrite an old run.

## Kaggle background jobs (main path)

One-time: Kaggle -> Settings -> API -> Create New Token; save `kaggle.json` to `C:\Users\vishe\.kaggle\`.
Run from `RESNET50_NOVELTY MODEL` on the laptop (code must be committed and pushed; the launcher checks):

```
python -m ablation.kaggle_launcher plan    --batch batch1
python -m ablation.kaggle_launcher submit  --batch batch1          # job a1: T4 x2, two queues of 9 runs
python -m ablation.kaggle_launcher status                         # live STATUS from `kaggle kernels logs`
python -m ablation.kaggle_launcher advance                        # job ended? submit a2 continuing from a1's output
python -m ablation.kaggle_launcher collect                        # download results (no checkpoints)
python -m ablation.analysis.generate_report                       # results/reports/ablation_report.md
```

**Switch to the blueprint arms (approved 2026-11-15).** Job a1 runs Batch 1. When it has ended, and before
`advance`, point the chain at blueprint v4 Section 7 (A0-A8, 30 rounds):

```
python -m ablation.kaggle_launcher requeue --chain batch1-a --batch blueprint   # prints the two new queues
python -m ablation.kaggle_launcher advance                                     # job a2 (fedmel-blueprint-a2)
```

`requeue` drops runs a1 finished, puts runs a1 left part-way first, and splits the rest over the two GPUs by
rounds left. Batch 1 arms that are not in the blueprint (E01, E03-E06) keep whatever seeds a1 finished and are
reported as supplementary. The first job after a requeue first runs A6, A8, A3, A4 for 2 rounds (smoke test of
the new code paths) and stops if any fails. `kaggle kernels logs` is empty while a job runs; the job's
Logs tab on kaggle.com shows the live STATUS lines.

Each job stops itself after the round that would cross 11 h (Kaggle's limit is 12 h), so its output is
always saved. `advance` is safe to run any time: it does nothing while a job is running, continues a chain
whose job ended, stops a chain after 2 jobs in a row end without a clean finish (then read
`kaggle kernels logs <slug>`), and marks a chain finished when all its runs are COMPLETED.

## Interactive session over SSH (smoke tests)

Setup cell: `Desktop/capstone/kaggle_ssh_setup.py`; host `kaggle` in `~/.ssh/config`.
On the session: Flower deps in `/tmp/fmdeps` (PYTHONPATH), image cache in `/tmp/fedmel_cache`, repo clone in
`/kaggle/working/repo`. These are lost when the session ends; `/kaggle/working` is lost too unless saved.

## Run states

QUEUED -> RUNNING -> COMPLETED; RUNNING -> RETRY_PENDING (crash, resumes from checkpoint, up to 3 tries
without progress) -> FAILED (reason kept in status.json and training.log); RUNNING -> INTERRUPTED (job
deadline; next job resumes it).
