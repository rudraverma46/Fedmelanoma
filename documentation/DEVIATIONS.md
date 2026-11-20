# Deviations: resnet_50_fedstrag_final

## From the FedStrag paper / official code
| Item | Paper | Official code (9d645c2) | Here | Why |
|---|---|---|---|---|
| Aggregation weight | W x D | W only | W x D | user decision: paper equation |
| Freshest update weight | 1 | 1/2 | 1 | paper Eq. 8 |
| Waiting | fixed T_d | poll 4 s until >= 1 update | poll 4 s until >= 1 | official value; paper gives no number |
| No update can arrive | aggregate nothing | waits forever | model unchanged, iteration ends | time-bounded rule (did not occur) |
| Busy client | not specified | FIFO | FIFO | official client |
| Stragglers and failures | sleep 5-20 s / reconnect | sleep 20 s, no failures | Novelty tiers, Beta/Bernoulli failure after training | same environment as Novelty |
| Time of a failed job | not specified | not simulated | one tier LogNormal duration (child stream) | computation already consumed; audit correction |
| Local data / epochs | 60-sample slices, 16 epochs | 60, 32 epochs | full partition, 5 epochs | identical to Novelty |
| Iterations | 200 | 200 | 50 | identical to Novelty's 50 rounds |
| Default aggregator | FedStrag | fedAvg() active, async_w_fedAvg() commented | FedStrag | the method under test |

## From a clean, identical-hardware comparison with Novelty
- Initial weights loaded from file (Novelty's 330768fe...) because PyTorch 2.14 builds 06b67e7e... from seed 42.
- Trained on an H100 MIG (PyTorch 2.14) vs Novelty on an RTX 2050 (PyTorch 2.2.1): same code, data, schedule and start, not bit-identical arithmetic.
- Environment random numbers: per-client streams (7921); Novelty used one shared generator. Same distributions.
- Per-iteration evaluation set: hospitals aggregated in that iteration (Novelty: the 6 selected, including failed ones). Affects only the secondary test-selected checkpoint.
- runtime/resources.py is the cross-platform version (Linux server); monitoring only.
- evaluation/leakage_analysis.py copied from the FedLesScan novelty-env folder; only the result folder names differ.
