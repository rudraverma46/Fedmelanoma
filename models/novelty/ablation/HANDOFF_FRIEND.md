# Handoff: an independent share of the ablation on your own Kaggle account

Written 2026-11-15. The FedMelanoma ResNet-50 ablation follows blueprint v4 Section 7 (arms A0-A8,
30 rounds). Vishesh's own account runs part of it; this file is for a collaborator running a further,
disjoint slice **on their own Kaggle account and laptop**, in parallel. Each slice is an explicit list of
runs (experiment:seed), fixed by Vishesh so no two people's accounts ever train the same run.

Background: `ablation/AUDIT_AND_PLAN.md` (design), `ablation/RUNBOOK.md` (how jobs/chains/resume work),
`ablation/experiments.yaml` (the arms).

## Your runs and chain letter

Vishesh gives you both of these (filled into the prompt he pastes to you):
- **Chain letter:** `<CHAIN>` (a single lowercase letter, unique to you — never `a`, which is his own account)
- **Run list:** `<RUNS>` (a list of `experiment:seed`, e.g. `A2_no_attenuation:43`)

Run only this exact list. Do not add, remove or guess runs — if you want more, ask Vishesh for another list
rather than extending this one; he tracks which runs are assigned to which account centrally.

## One-time setup

1. Dataset `visheshkumarbansal/fedmel01` is public, so no sharing step is needed.
2. Your Kaggle account must be phone-verified (needed for GPU and internet in notebooks).
3. Create your own API token: Kaggle -> Settings -> API -> Create New Token. Save it under `~/.kaggle/`
   on your own laptop. Never paste the token into a chat or send it to anyone.
4. `pip install kaggle pyyaml pandas` (Python 3.10+).

## Run it

```
git clone -b resnet50-novelty-ablation https://github.com/rudraverma46/Fedmelanoma
cd "Fedmelanoma/RESNET50_NOVELTY MODEL"
python -m ablation.kaggle_launcher plan   --chain <CHAIN> --runs <RUNS>   # confirm it lists exactly your runs
python -m ablation.kaggle_launcher submit --chain <CHAIN> --runs <RUNS>   # pushes the job
```

The job first runs a 2-round smoke test of A3, A4, A6, A8 (whichever of those are in your list), then
trains until about 11 h and saves its output.

Watch it on kaggle.com: Code, the new kernel, Logs tab (STATUS lines every 5 min). `kaggle kernels logs`
stays empty until the job ends, so the launcher's `status` shows nothing while it runs.

When it has ended (Kaggle shows Complete):

```
python -m ablation.kaggle_launcher advance    # continues only if your list has unfinished runs
```

Repeat after each job until it prints `<batch>-<CHAIN>: all runs completed`. Before starting a job, check
your weekly GPU quota has at least 12 h left; a job killed by the quota may lose its progress.

## Hand the results back

```
python -m ablation.kaggle_launcher collect    # downloads finished runs into results/raw (no checkpoints)
```

Zip `results/raw/*/seed_*` (only the runs from your list) and send it to Vishesh. He copies the folders
into his own `results/raw`; each run's folder name (experiment/seed) is unique, so nothing collides with
anyone else's results, and the report merges everyone's runs automatically.

## Rules (agreed with Vishesh; ask him before changing any)

- Do not edit `ablation/experiments.yaml`, `ablation/variants.py`, `ablation/common.py`, `configs/` or any
  file outside `ablation/`. Changing them changes run ids, so your runs would not match his.
- Do not commit or push to the branch; the launcher refuses uncommitted changes and pins the pushed commit.
- Run only the exact list Vishesh gave you, under your own chain letter. Never run a chain letter or a run
  someone else was given.
- Kaggle GPUs only (T4 x2), 30 rounds, no AMP or other speed changes (not approved).
- Use only your own Kaggle account and token, on your own laptop.
