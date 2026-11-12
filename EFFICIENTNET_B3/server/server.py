"""
server/server.py
=================
Phase 3B — Flower ServerApp Wrapper.
Phase 4  — Wired to the real FedMelanomaStrategy (server/strategy.py).

Phase 3B originally wired FedMelanomaClient (client/client.py) into a
runnable federated loop using a thin subclass of Flower's built-in FedAvg
strategy, deliberately deferring FedMelanoma's own selection/aggregation
math to a later phase. Phase 4 is that later phase: server_fn() below now
builds the real, unmodified FedMelanomaStrategy instead of plain FedAvg.

FedMelanomaFedAvg / build_strategy() are kept, unmodified, as the
FedAvg-baseline ablation reference that aggregation.py's own docstring
describes (Section 3.10.1) -- they are no longer what `app` runs by
default, but remain available for ablation comparisons.

Scope (Phase 4)
----------------------
IN scope:
  - server_fn() / ServerApp: discovers hospitals dynamically (via the
    repo-root hospital_registry adapter -- no hospital name is hardcoded
    here), builds one HospitalConfig per hospital, seeds round-1
    parameters from a fresh FedMelanomaModel(), and constructs
    FedMelanomaStrategy with them.
  - weighted_average(): a small, generic, Flower-idiomatic helper that
    aggregates client-reported evaluation metrics (accuracy, precision,
    recall, f1, roc_auc) weighted by each client's local example count
    -- standard FedAvg-style metric bookkeeping, reused (not duplicated)
    by server/strategy.py.

OUT of scope (explicitly NOT implemented here):
  - Any FedMelanoma-specific aggregation weighting, client selection,
    straggler/staleness handling, or diversity computation -- all of
    that lives in server/aggregation.py, server/selector.py,
    server/straggler.py, and emd.py respectively, called unmodified by
    server/strategy.py's FedMelanomaStrategy (imported here, not
    reimplemented).
  - Any modification to client/local_trainer.py, dataset.py, model.py,
    or focal_loss.py.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from flwr.common import Context, Metrics, NDArrays, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from models.model import FedMelanomaModel

# ---------------------------------------------------------------------------
# Defaults. Overridable via the Flower run config (context.run_config).
# ---------------------------------------------------------------------------
DEFAULT_NUM_ROUNDS: int = 1
DEFAULT_MIN_CLIENTS: int = 2
DEFAULT_LOCAL_EPOCHS: int = 1
DEFAULT_CLIENTS_PER_ROUND: int = 6   # C, Blueprint Section 3.9.2 canonical default
DEFAULT_PARTITIONS_ROOT: str = "hospital_partitions"


# ===========================================================================
# Generic metric aggregation (standard FedAvg bookkeeping)
# ===========================================================================

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate client-reported metrics via a simple example-count-weighted
    average -- this is standard, generic FedAvg metric bookkeeping (as
    shown in Flower's own quickstart examples), not the FedMelanoma
    aggregation formula in server/aggregation.py.

    Each `metrics` entry is (num_examples, metrics_dict) as returned by
    FedMelanomaClient.fit() / .evaluate() (Phase 3A). Keys not present on
    a given client are treated as 0.0 for that client's contribution.

    Phase 5 fix: FedMelanomaClient now also returns a few NON-numeric
    diagnostic fields (parameter hashes, a JSON-encoded subgroup-metrics
    blob -- see client/client.py) alongside the numeric ones. A key is
    only included in the weighted average if every client that reported
    it reported a real number; non-numeric keys are skipped here rather
    than crashing (the original implementation unconditionally did
    `num_examples * value`, which raises TypeError on a str value once
    divided by total_examples). Strategy code that needs those non-
    numeric fields (server/strategy.py) reads them directly from each
    client's raw FitRes/EvaluateRes.metrics instead of through this
    average.
    """
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    keys = set()
    for _, client_metrics in metrics:
        keys.update(client_metrics.keys())

    aggregated: Dict[str, float] = {}
    for key in keys:
        values = [client_metrics.get(key) for _, client_metrics in metrics]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values if v is not None):
            continue  # non-numeric (e.g. a parameter hash or JSON blob) -- not averageable
        aggregated[key] = (
            sum(num_examples * client_metrics.get(key, 0.0) for num_examples, client_metrics in metrics)
            / total_examples
        )
    return aggregated


# ===========================================================================
# Legacy / ablation-only: plain FedAvg strategy
# ===========================================================================
#
# Not used by `app` below (Phase 4 wires FedMelanomaStrategy instead -- see
# server_fn()). Kept unmodified as the FedAvg-baseline ablation reference
# that aggregation.py's own docstring describes (Section 3.10.1:
# "Included purely for ablation comparison against the Section 3.10.2
# formula") -- e.g. for a separate ablation server_fn that swaps this in.

class FedMelanomaFedAvg(FedAvg):
    """
    Thin FedAvg subclass. Deliberately adds NO overridden methods and NO
    custom aggregation math -- it exists only so an ablation entry point
    has one clearly-named, FedMelanoma-scoped class to configure without
    touching Flower's own, already-correct FedAvg implementation.
    """
    pass


def build_strategy(
    num_clients: int,
    initial_parameters: Optional[NDArrays] = None,
) -> FedMelanomaFedAvg:
    """
    Construct a minimal FedAvg strategy sized to `num_clients`.

    `initial_parameters`, if provided, seeds round 1 with a known
    starting point (e.g. a freshly constructed FedMelanomaModel's
    weights) instead of relying on Flower's default behavior of asking
    an arbitrary client for its initial parameters.
    """
    kwargs = dict(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    if initial_parameters is not None:
        kwargs["initial_parameters"] = ndarrays_to_parameters(initial_parameters)
    return FedMelanomaFedAvg(**kwargs)


# ===========================================================================
# ServerApp entry point (Flower's modern flwr>=1.8 API) -- Phase 4
# ===========================================================================

def _build_initial_parameters() -> Tuple[NDArrays, List[str]]:
    """
    Instantiate a fresh FedMelanomaModel() (unmodified) purely to obtain a
    known round-1 parameter set, instead of relying on Flower's default
    behavior of asking an arbitrary client for its initial weights.

    Also returns the model's state_dict key order (param_keys), computed
    from this SAME model instance so it is guaranteed consistent with the
    ndarrays -- threaded through to FedMelanomaStrategy so
    monitoring/checkpointing.py can save a properly named state_dict
    without constructing (and re-downloading pretrained weights for) a
    second model just for its key names.
    """
    model = FedMelanomaModel()
    ndarrays = [val.detach().cpu().numpy().copy() for val in model.state_dict().values()]
    param_keys = list(model.state_dict().keys())
    return ndarrays, param_keys


def server_fn(context: Context) -> ServerAppComponents:
    """
    Flower's ServerApp entry point. Builds the real FedMelanomaStrategy
    (server/strategy.py, unmodified) from dynamically discovered hospital
    partitions and hands control to Flower's own round-orchestration loop.

    Steps:
      1. Discover hospitals under hospital_partitions/ via the repo-root
         hospital_registry adapter -- no hospital name is hardcoded here.
      2. Convert each discovered hospital's registry record into a
         server.strategy.HospitalConfig (a pure field-renaming/construction
         step; no distance/score math happens in this file).
      3. Seed round-1 parameters from a fresh FedMelanomaModel().
      4. Construct FedMelanomaStrategy with the above.

    Overridable via run_config: "num-server-rounds", "clients-per-round",
    "local-epochs", "partitions-root", "seed", "run-id", "checkpoint-dir",
    "results-dir", "logs-dir" (Phase 5 additions -- all optional).

    Only invoked when a run actually starts -- importing this module and
    building `app = ServerApp(server_fn=server_fn)` below do not require
    hospital_partitions/ to exist.
    """
    num_rounds = int(context.run_config.get("num-server-rounds", DEFAULT_NUM_ROUNDS))
    local_epochs = int(context.run_config.get("local-epochs", DEFAULT_LOCAL_EPOCHS))
    partitions_root = str(context.run_config.get("partitions-root", DEFAULT_PARTITIONS_ROOT))
    clients_per_round = int(
        context.run_config.get("clients-per-round", DEFAULT_CLIENTS_PER_ROUND)
    )
    # Blueprint Section 10 "system.seed: 42" -- default reproducibility seed.
    seed = int(context.run_config.get("seed", 42))
    checkpoint_dir = str(context.run_config.get("checkpoint-dir", "checkpoints"))
    results_dir = str(context.run_config.get("results-dir", "results"))
    run_id_override = context.run_config.get("run-id")

    # Local imports: avoids a module-level import cycle with
    # server.strategy (which itself imports server.server.weighted_average),
    # and keeps this module importable without hospital_registry's
    # pandas/numpy dependency chain being required at `import server.server` time.
    from hospital_registry import build_hospital_registry
    from server.strategy import FedMelanomaStrategy, HospitalConfig
    from monitoring.experiment_logging import ExperimentLogger, new_run_id

    hospital_names, hospital_records, global_fitz, global_mod, global_conf = (
        build_hospital_registry(partitions_root)
    )
    if not hospital_records:
        raise RuntimeError(
            f"No hospital partitions discovered under '{partitions_root}'. "
            "server_fn requires hospital_partitions/<hospital>/metadata_train.csv "
            "to exist for at least one hospital before a federated round can "
            "run. (ServerApp construction itself does not require this.)"
        )

    hospital_configs = [HospitalConfig(**record) for record in hospital_records]
    initial_parameters, param_keys = _build_initial_parameters()
    run_id = str(run_id_override) if run_id_override else new_run_id("fedmelanoma")
    logger = ExperimentLogger(run_id)
    logger.log_audit(
        {
            "event": "server_fn_config",
            "num_rounds": num_rounds,
            "clients_per_round": clients_per_round,
            "local_epochs": local_epochs,
            "partitions_root": partitions_root,
            "hospital_names": hospital_names,
            "seed": seed,
        }
    )

    strategy = FedMelanomaStrategy(
        hospital_configs=hospital_configs,
        global_fitzpatrick=global_fitz,
        global_modality=global_mod,
        global_confirmation=global_conf,
        n_select=min(clients_per_round, len(hospital_configs)),
        initial_parameters=initial_parameters,
        local_epochs=local_epochs,
        rng=np.random.default_rng(seed),
        experiment_logger=logger,
        run_id=run_id,
        checkpoint_dir=checkpoint_dir,
        results_dir=results_dir,
        param_keys=param_keys,
        seed=seed,
    )
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


# Building the ServerApp only registers server_fn as a callback -- it does
# NOT invoke it, so this line succeeds whether or not hospital_partitions/
# exists.
app = ServerApp(server_fn=server_fn)
