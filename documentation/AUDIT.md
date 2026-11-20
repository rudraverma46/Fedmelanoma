# Audit: resnet_50_fedstrag_final

Written 2026-11-20 from the run's own files.

| Check | Result |
|---|---|
| CPU tests (00, 02, 04) | 64 passed, 0 failed (run when the documents were built) |
| Local verification (RTX 2050): tests 0-16 / smoke tests 17-22 | 74 / 6 passed |
| Server verification (H100) before the full run: tests 0-16 / smoke tests 17-22 | 75 / 6 passed |
| Pipeline identical to Novelty | SHA-256 of 15 pipeline files, straggler.py, 40 cache files and 20 split CSVs (configs/reference_manifest.json) |
| Initial global model | 330768fe201714e3f26b9973f70fd32cbec63e2bce8b8569194ef65641876f4a (= Novelty's logged initial hash) |
| Schedule (deterministic) | identical on PC and server; run reproduced it: jobs 243, failed 27, applied 208, in flight 8, failing 2 |
| Unaccounted updates / empty iterations | 0 / 0 |
| Novelty mechanisms absent | files absent, runner imports none, identifier scan clean, only simulate_response called (tests/test_00, test_04) |
| Poll interval | 4 s (config, constant, runner check, tests) |
| Restarts | none (one attempt, exit 0) |
| Results transfer | server archive SHA-256 27f12b11... verified on this PC |

Known gaps: one seed; Novelty and FedStrag trained on different GPUs and PyTorch versions (same initial weights); the GPU columns of the resource log are empty on the MIG slice; no validation split.
