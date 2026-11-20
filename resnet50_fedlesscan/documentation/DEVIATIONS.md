# resnet50_fedlesscan_novelty_env: deviations and decisions

Same ResNet-50, data, hospital partitions, 5 local epochs, 50 rounds, 6 of 10 clients per round, three straggler tiers
(Beta completion, LogNormal latency; seed 42, stream 7919) as the Novelty run. Only the federated algorithm is FedLesScan.

## Delayed updates: paper vs official code
- Paper (arXiv 2211.05739, Sec. V-D): weight n_k/n * t_k/t; updates with t - t_k >= tau (tau = 2) are discarded.
- Official code (commit 039f964): candidates are results with round_id >= t - tolerance (tolerance 2), i.e. accepted when
  t - t_k <= 2; score (t_k + 1)/(t + 1) with zero-based rounds.
- THIS EXPERIMENT FOLLOWS THE OFFICIAL CODE: staleness 0, 1, 2 accepted; 3 or more discarded; score (t_k+1)/(t+1).
  It does not follow the paper's strict boundary condition.

## Choices made for this environment (user, 2026-11-19)
- No fixed crashers. Failure = the client trains first, then the update is discarded (as in the Novelty run); counted as a
  missed round with cooldown.
- Fixed round deadline 30 s on the simulated latency (replaces the Novelty adaptive window). Later = late update, arriving
  in round t + ceil(latency/30) - 1; it is aggregated the next time aggregation runs, if its staleness is <= 2.
- Training time for clustering = simulated latency in minutes.
- Aggregation: official formula, NOT renormalised (user's choice); late updates (score < 1) therefore scale the model.
- A round with nothing to aggregate keeps the global model (official code raises an error).
- Evaluation: local test sets of the clients selected in the round (as in the Novelty run).

## Not used from the Novelty method (final cleanup, 2026-11-19)
Removed from this folder: server/selector.py, server/aggregation.py, server/straggler.py, emd.py and
monitoring/straggler_logging.py. strategy.py no longer builds D_i, R_i, N_eff, the selector or the straggler manager, and
logs none of them. The only link to the Novelty code is the ENVIRONMENT: server/environment.py reproduces the novelty
simulate_response draws (Beta completion, Bernoulli failure, LogNormal latency; seed 42, stream 7919 per client);
tests/test_00 checks it draw for draw against the novelty file. Not used: Boltzmann selection, diversity-aware selection or
aggregation, adaptive median+MAD window, one-round buffer, novelty aggregation weights.
Comment-only mentions remain in files identical to the reference (client/client.py, hospital_registry.py,
monitoring/experiment_logging.py); hospital_registry.py still computes the metadata histograms, which nothing here reads.
