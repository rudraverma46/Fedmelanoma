# Audit: resnet50_fedlesscan_novelty_env

Written 2026-11-20 from the run's own files.

| Check | Result |
|---|---|
| CPU tests (00, 01, 02, 04, 05) | 81 passed, 0 failed (run when the documents were built) |
| Real run vs precomputed schedule | 50 rounds compared, 0 mismatches |
| Counters | selections 300, on time 236, late 10, failed 54, aggregated 243 (planned: 243) |
| Empty rounds / late updates in flight at the end | 0 / 0 |
| Initial global model hash | 06b67e7eaa7aaa6986b87453c5c929ef518ec8b3f3b0bc974efd71e36829fa1a |
| Hash chain broken / hash failures | 0 / 0 |
| Exceptions | 0 |
| Safe stops / resumes | 1 (after round 27) / 1 |
| Final test evaluation | best and round-50 checkpoints, pooled 1,952-image test set, on the local RTX 2050 |

Not checked / known gaps: `tests/test_03_real_round.py` was not run (it needs the smoke-run checkpoints of the server; the schedule comparison covers its purpose). GPU memory and utilisation columns of the resource log are empty (not recorded on the server). Results are from a single seed.
