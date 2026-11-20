# resnet_50_fedstrag_final — standard FedStrag baseline for FedMelanoma

Standard FedStrag (Kumar & Srirama, *Digital Communications and Networks* 11(4):1214–1224, 2025,
DOI 10.1016/j.dcan.2024.12.004; official code `cloud-and-smart-labs/FedStrag` @ `9d645c2`) on the ResNet-50
FedMelanoma pipeline, in the same straggler **and failure** environment as the Novelty run. Built 2026-11-20 from
the brief "Finalize and Implement Standard FedStrag Baseline"; corrected 2026-11-20 after the pre-experiment audit
(failed-job time, primary checkpoint, documentation). Run id: `resnet50-fedstrag-final-seed42`.

How this baseline differs from Novelty, including what Novelty's quorum and window actually do in code:
[docs/NOVELTY_VS_FEDSTRAG.md](docs/NOVELTY_VS_FEDSTRAG.md).

## What is identical to the Novelty run (tests/test_00, byte-for-byte)
ResNet-50 + metadata MLP fusion model, class-weighted focal loss, client training code (AdamW, 5 epochs, batch 20,
lr 1e-4, wd 1e-4, γ 2), the 10 hospital partitions and their fixed sorted mapping, the pixel cache (byte-identical
to the Novelty cache), `evaluate.py`, checkpointing, seed 42 and the initial model (hash `330768fe…`). The straggler
environment is the Novelty `server/straggler.py` file itself (copied to `sim/straggler.py`), of which only
`simulate_response` is called. `configs/reference_manifest.json` records the SHA-256 of all of these, so the
identity can be checked on a machine without the Novelty folder.

## The algorithm (server/protocol.py, server/fedstrag.py)
**Global iteration g = 1…50** = one broadcast + one server decision.
1. **Broadcast** the global model to **all 10** hospitals with tag `t = g − 1`; `t_g = t`.
2. **Clients** process broadcasts in **FIFO** order, one at a time (official client: single-threaded MQTT
   handler). Each dispatched job is **trained**, then the Novelty environment draws its outcome:
   * *delivered* → the update reaches the server after the tier LogNormal delay (virtual clock); the client is
     busy until then;
   * *failed* (Beta/Bernoulli of the tier) → the update is lost; no penalty, retry or replacement. The training
     already happened, so the job **occupies the client for one tier LogNormal job duration** (the existing delay
     model, drawn from the per-client child stream `[42, 7921, c, 1]` so stream 7921 itself is unchanged); then
     the loss is recorded (`update_lost`) and the client takes its next queued broadcast.
3. **Poll every 4 s** (official `wait_time = 4`) until ≥ 1 update has arrived (`minimum_clients = 1`).
4. **Aggregate** all arrived updates: `W = 1/(t_g − t_l + 1)` (Eq. 6), `θ = Σ θ·W·D / Σ W·D` (Eq. 7, D = training
   samples); the result **replaces** the global model (model version + 1).

**Empty iteration**: if no update arrived and none can arrive (no client busy, neither with a delivering nor with a
failing job), the iteration ends at that poll with the model unchanged. The server keeps polling while a failing job
runs, since it cannot know the job will fail.
**Termination**: after iteration 50 nothing more is broadcast or aggregated. Updates in flight, failing jobs still
running and broadcasts still queued at clients are **excluded and reported** (`results/<run>/run_summary.json`).

**Primary reported checkpoint = the iteration-50 model** (`checkpoints/<run>/final_global_model.pth`, evaluated into
`results/<run>/final/`). The per-iteration "best" checkpoint is chosen by ROC-AUC on local **test** partitions; it is
evaluated only as `results/<run>/secondary_best_test_selected/` and is never the reported result.

Not present (tests/test_00): EMD diversity, skin-type/modality/confirmation weighting, Boltzmann selection,
reliability EMA, exposure, adaptive window W_t, quorum, S_max discard, one-update-per-client buffer.

## Expected accounting of the full run (deterministic)
The schedule depends only on the seeded environment, not on training results, so a CPU replay gives the exact
counts of the full run: 243 local trainings, 27 failed (updates lost), 216 successful, 208 applied, 8 in flight
and 2 failing jobs running at the end (excluded), 257 queued broadcasts never started, 0 empty iterations,
204 virtual seconds. Applied updates per hospital: 46, 8, **0**, 43, 8, **0**, 47, 9, **1**, 46 — with the 4 s poll
the high-straggler hospitals (2, 5, 8; median delay ~60 s) contribute almost nothing. 93 % of applied updates are stale.

## Deviations and choices
| Item | Paper | Official code | Here |
|---|---|---|---|
| Weight | W·D (Eq. 7) | W only | **W·D** (paper) |
| Freshest update weight | 1 (Eq. 8) | 1/2 (staleness after `iter += 1`) | **1** (paper) |
| Wait | fixed T_d, aggregate what arrived | poll 4 s until ≥ 1 update | **poll 4 s until ≥ 1** (code) |
| Nothing can arrive | aggregate nothing | waits forever | **model unchanged, iteration ends** |
| Busy client | not specified | FIFO, every broadcast trained | **FIFO** (code) |
| Stragglers | fixed client sleep 5–20 s / random 25 % | client3/client5 sleep 20 s | **Novelty tiers** (brief) |
| Failures | "reconnects and contributes" | none simulated | **Novelty Beta/Bernoulli; trained, then lost** (brief) |
| Time of a failed job | not specified | not simulated | **one tier LogNormal job duration** (correction) |
| Local data / epochs | online slices of 60, 16 epochs | 60, 32 epochs | **full partition, 5 epochs** (Novelty) |
| Iterations | 200 | 200 | **50** (brief) |
| Default aggregator | FedStrag | `main()` calls `fedAvg()`; `async_w_fedAvg()` commented out | FedStrag |

## Running
```
python -m pytest tests/test_00_reference_integrity.py tests/test_02_fedstrag_math.py tests/test_04_environment.py
python tools/run_verification.py                 # all tests + GPU smoke run with stop/resume
bash tools/remote_run.sh check|verify|full       # remote Linux GPU server (full run needs approval)
```
Outputs: `logs/{iterations,clients,updates,rounds}/<run>.jsonl`, `logs/resources/<run>.csv`,
`results/<run>/run_config.json`, `results/<run>/run_summary.json`, `results/<run>/final/` (PRIMARY: metrics.json,
confusion matrix, ROC/PR curves, predictions), `results/<run>/secondary_best_test_selected/`, `checkpoints/<run>/`.
