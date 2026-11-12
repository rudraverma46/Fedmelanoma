# FedMelanoma: ResNet-50 Novelty model

Federated melanoma classification with a ResNet-50 image backbone fused with patient metadata,
trained with the Novelty method: diversity-aware Boltzmann client selection, straggler
simulation and staleness-aware aggregation across 10 hospital clients.

Selection, aggregation, straggler simulation, loss, data pipeline, local training and evaluation
are fixed, unmodified files (`REFERENCE_FILES.json`, checked by the tests). `models/model.py`
contains the ResNet-50 fusion model.

## Layout

| Path | Content |
|---|---|
| `client/`, `server/`, `monitoring/`, `emd.py`, `hospital_registry.py`, `evaluate.py` | Reference files, unmodified |
| `models/model.py` | ResNet-50 fusion model |
| `runtime/` | In-process Flower transport, round loop, resume, resource monitor, image cache |
| `evaluation/` | Runs `evaluate.py` on a checkpoint; builds the report |
| `tools/` | Setup, schedule replay, verification and supervised run |
| `tests/` | The 18 verification tests plus reference integrity |
| `configs/experiment_config.yaml` | Configuration |
| `checkpoints/`, `logs/`, `results/`, `cache/` | Generated |

## Reproducing

Python 3.11 with `requirements-lock.txt`, an NVIDIA GPU, and the `hospital_partitions` data at the
path in the config. Run from this folder:

```
python tools/vendor_reference_files.py        # re-copy reference files from a clone (optional)
python tools/build_image_cache.py             # one-time, CPU, about 15 minutes
python tools/dry_run_schedule.py              # replay the participation schedule, no training
python tools/run_verification.py              # all tests, including a real two-round GPU run
python tools/run_supervised.py resnet50-seed42    # the 50-round experiment; resumes after a crash
python evaluation/evaluate_checkpoint.py checkpoints/resnet50-seed42/best_global_model.pth results/resnet50-seed42/final_best
python evaluation/evaluate_checkpoint.py checkpoints/resnet50-seed42/global_round_0050.pth results/resnet50-seed42/final_round50
python evaluation/build_report.py resnet50-seed42
```

Run these one at a time: on an 8 GB machine, overlapping them exhausts memory. To stop a run
cleanly at the end of a round, create the file `logs/STOP`.
