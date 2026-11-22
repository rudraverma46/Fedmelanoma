# Handoff: chain b (seed 44) on Rudra's Kaggle account

Written 2026-11-15. The FedMelanoma ResNet-50 ablation follows blueprint v4 Section 7 (arms A0-A8,
30 rounds). Vishesh's account runs seeds 42 and 43 (chain a: jobs a1, a2, a3). This file is the share for a
second account: **seed 44 of all 8 blueprint arms**, chain **b**. The two shares never overlap.

Background: `ablation/AUDIT_AND_PLAN.md` (design, Section 4b = blueprint arms), `ablation/RUNBOOK.md`
(how jobs, chains and resume work), `ablation/experiments.yaml` (the arms).

## Your 8 runs (about 20 GPU-hours, run two at a time on T4 x2, so about one 11 h job)

| GPU 0 | GPU 1 |
|---|---|
| E00_full:44 (A0, full model) | A1_no_diversity:44 |
| A2_no_attenuation:44 | A3_greedy_selection:44 |
| A4_binary_reliability:44 | A5_static_reliability:44 |
| A6_tv_diversity:44 | A8_synchronous:44 |

## One-time setup

1. Vishesh shares the private dataset `visheshkumarbansal/fedmel01` with Rudra's Kaggle account
   (kaggle.com, the dataset, Settings, Sharing, add the user as Viewer). Jobs cannot read it otherwise.
2. Rudra's Kaggle account must be phone-verified (needed for GPU and internet in notebooks).
3. Rudra: Kaggle, Settings, API, Create New Token. Save it in `~/.kaggle/` on his own laptop
   (`kaggle.json`, or `access_token`). Never paste the token into a chat.
4. Set your Kaggle username so jobs are created on your account (needed with an `access_token` file):
   Windows PowerShell `$env:KAGGLE_USERNAME="<your kaggle username>"`, bash `export KAGGLE_USERNAME=<username>`.
5. `pip install kaggle pyyaml pandas` (Python 3.10+).

## Run it

```
git clone -b resnet50-novelty-ablation https://github.com/rudraverma46/Fedmelanoma
cd "Fedmelanoma/RESNET50_NOVELTY MODEL"
python -m ablation.kaggle_launcher plan   --batch blueprint --seeds 44 --chain b   # must list exactly the 8 runs above
python -m ablation.kaggle_launcher submit --batch blueprint --seeds 44 --chain b   # job fedmel-blueprint-b1
```

The job first runs a 2-round smoke test of A6, A8, A3, A4 (about 30 min) and stops if any fails; then it
trains until about 11 h and saves its output.

Watch it on kaggle.com: Code, `fedmel-blueprint-b1`, Logs tab (STATUS lines every 5 min).
`kaggle kernels logs` stays empty until the job ends, so the launcher's `status` shows nothing while it runs.

When it has ended (Kaggle shows Complete):

```
python -m ablation.kaggle_launcher advance    # starts b2 only if runs are left; safe to run any time
```

Repeat `advance` after each job until it prints `blueprint-b: all runs completed`. Before starting a job,
check that your weekly GPU quota has at least 12 h left; a job killed by the quota may lose its progress.

## Hand the results back

```
python -m ablation.kaggle_launcher collect    # downloads finished runs into results/raw (no checkpoints)
```

Zip `results/raw/*/seed_44` and send it to Vishesh. He copies the folders into his own `results/raw` and
builds the report (`python -m ablation.analysis.generate_report`), which merges both accounts' runs.

## Rules (agreed with Vishesh; ask him before changing any)

- Do not edit `ablation/experiments.yaml`, `ablation/variants.py`, `ablation/common.py`, `configs/` or any
  file outside `ablation/`. Changing them changes run ids, so your runs would not match his.
- Do not commit or push to the branch; the launcher refuses uncommitted changes and pins the pushed commit.
- Run only seed 44 and only `--chain b`. Never run jobs for seeds 42 / 43 (they are on Vishesh's account).
- Kaggle GPUs only (T4 x2), 30 rounds, no AMP or other speed changes (not approved).
- Use only your own Kaggle account and token, on your own laptop.
