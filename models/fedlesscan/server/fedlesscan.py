"""FedLesScan: clustering-based client selection and staleness-aware aggregation.

A port of the official implementation, Serverless-Federated-Learning/FedLesScan, commit
039f9640754818855e1b4715733043344db764fe (Elzohairy et al., "FedLesScan: Mitigating Stragglers in
Serverless Federated Learning", IEEE BigData 2022, arXiv 2211.05739). Every constant and decision
rule below is the official code's; each function names the file and lines it comes from. Only
the storage changed: the official code keeps client history and client results in MongoDB, this
file keeps them in memory (and runtime/state.py pickles them for resume).

Round numbers are 0-based inside this module, as in the official controller
(fedless/controller/strategies/fl_strategy.py:58, `for round in range(max_rounds)`). The Flower
strategy passes server_round - 1.

resnet50_fedlesscan_novelty_env (2026-11-19): the straggler environment is the novelty one instead of the
fixed crashers. Differences from resnet50_fedlesscan, all chosen by the user:
  * No fixed stragglers (n_stragglers = 0). A client's job k has (failed, latency) from the novelty
    Beta/Bernoulli/LogNormal trace (server/environment.py). Failure = the client trains first, then the update
    is discarded (as in the novelty run); it counts as a missed round with cooldown.
  * Fixed round deadline of 30 s on the SIMULATED latency (no adaptive window, no buffer). Latency <= 30 s:
    on time. Otherwise the update is late and arrives in round t + ceil(latency/30) - 1; it is stored then
    with round_id = its training round and aggregated with the official score (t_k+1)/(t+1) if t - t_k <=
    tolerance (2), otherwise it is stored but ignored, as in the official code.
  * Training time recorded for clustering = the simulated latency in minutes.
  * Aggregation is the official formula, no renormalisation (user's choice, 2026-11-19). A late update
    (score < 1) therefore scales the whole aggregated model below 1.
  * A round in which nothing can be aggregated keeps the global model unchanged (official code raises).
Common settings (as in resnet50_fedlesscan, 2026-11-10): 6 of 10 clients per round; allowed_stragglers = 6, so a round
never aborts (serverless_strategy.py:288-293 would raise); official constants for the EMA, DBSCAN and cooldown.

Paper versus official code (delayed updates): the paper (arXiv 2211.05739, Sec. V-D) weights an old update by t_k/t and
discards updates with t - t_k >= tau (tau = 2). The official code keeps results with round_id >= t - tolerance (tolerance 2),
i.e. accepts t - t_k <= 2, and scores them (t_k + 1)/(t + 1) with zero-based rounds. THIS IMPLEMENTATION FOLLOWS THE OFFICIAL
CODE, not the paper's strict boundary.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from functools import reduce
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

OFFICIAL_COMMIT = "039f9640754818855e1b4715733043344db764fe"

# Official constants (file:line in the official repository).
TRAINING_TIME_EMA_ALPHA = 0.5     # Intelligent_selection.py:87  get_client_ema(alpha=0.5)
MISSED_ROUND_EMA_ALPHA = 0.8      # Intelligent_selection.py:107 get_client_ema(..., alpha=0.8)
MISSED_ROUND_PENALTY_SCALE = 1.5  # Intelligent_selection.py:107 * 1.5 * max_training_time
DBSCAN_MIN_SAMPLES = 2            # Intelligent_selection.py:373 DBSCAN(eps=eps, min_samples=2)
EPS_START, EPS_STOP, EPS_STEP = 0.01, 2, 0.1  # Intelligent_selection.py:372 and :309 (eps_step=0.1)
STALENESS_TOLERANCE = 2           # experiment/*/*-demo.yaml  aggregator.hyperparams.tolerance: 2


@dataclass
class ClientHistory:
    """fedless/common/models/models.py:259-268 ClientPersistentHistory (fields used by FedLesScan)."""

    client_id: int
    training_times: List[float] = field(default_factory=list)  # minutes
    missed_rounds: List[int] = field(default_factory=list)
    client_backoff: float = 0
    latest_updated: int = -1
    train_cardinality: int = -1


@dataclass
class ClientResult:
    """One stored client update (the official ClientResultDao record)."""

    client_id: int
    round_id: int
    cardinality: int
    parameters: List[np.ndarray]


# ---------------------------------------------------------------------------
# Behaviour features (Intelligent_selection.py:87-107)
# ---------------------------------------------------------------------------

def get_client_ema(times_list: Sequence[float], alpha: float = TRAINING_TIME_EMA_ALPHA) -> float:
    """Intelligent_selection.py:87-98."""
    if len(times_list) == 0:
        return 0
    ema = times_list[0]
    i = 1
    while i < len(times_list):
        ema = ema * (1 - alpha) + alpha * times_list[i]
        i += 1
    return ema


def get_missed_rounds_ema(times_list: Sequence[int], round: int, max_training_time: float) -> float:
    """Intelligent_selection.py:100-107: EMA(alpha 0.8) of (missed_round + 1) / round, x 1.5 x max time."""
    times_one_based = np.divide(np.array(times_list) + 1, round)
    return get_client_ema(times_one_based, alpha=MISSED_ROUND_EMA_ALPHA) * MISSED_ROUND_PENALTY_SCALE * max_training_time


def max_training_time_of(clients: Sequence[ClientHistory]) -> float:
    """Intelligent_selection.py:116-118 and :296-299."""
    return max([max(c.training_times, default=0) for c in clients], default=0)


# ---------------------------------------------------------------------------
# Partitioning, clustering and sampling (Intelligent_selection.py:109-161, 233-430)
# ---------------------------------------------------------------------------

def filter_rookies(clients: Sequence[ClientHistory], round: int):
    """Intelligent_selection.py:233-250: rookies, participants (rest), stragglers (in cooldown)."""
    rookies, rest_clients, stragglers = [], [], []
    for client in clients:
        if len(client.training_times) == 0 and len(client.missed_rounds) == 0:
            rookies.append(client)
        elif len(client.missed_rounds) > 0 and client.client_backoff + client.missed_rounds[-1] >= round:
            stragglers.append(client)
        else:
            rest_clients.append(client)
    return rookies, rest_clients, stragglers


def perform_clustering(training_data, eps_step: float = EPS_STEP):
    """Intelligent_selection.py:362-399: DBSCAN eps grid search, best Calinski-Harabasz score."""
    best_labels = None
    best_score = 0
    best_eps = 0
    X = StandardScaler().fit_transform(training_data)
    for eps in np.arange(EPS_START, EPS_STOP, eps_step):
        db = DBSCAN(eps=eps, min_samples=DBSCAN_MIN_SAMPLES).fit(X)
        labels = db.labels_
        if best_labels is None:
            best_labels = labels
            best_eps = eps
        n_lables = len(set(labels))
        if n_lables <= len(X) - 1 and n_lables > 1:
            clustering_score = metrics.calinski_harabasz_score(X, labels)
            if clustering_score > best_score:
                best_score = clustering_score
                best_labels = labels
                best_eps = eps
    return best_labels, best_score, best_eps


def sort_clusters(clients: Sequence[ClientHistory], labels, round: int):
    """Intelligent_selection.py:109-161: clusters ordered by mean totalEMA (Eq. 2), fastest first.

    DBSCAN outliers (label -1) form one cluster, as the paper states.
    """
    cluster_number_map: Dict[int, Tuple[float, List[ClientHistory]]] = {}
    max_training_time = max_training_time_of(clients)
    for idx in range(len(labels)):
        client_cluster_idx = int(labels[idx])
        total_ema = get_client_ema(clients[idx].training_times)
        total_ema += get_missed_rounds_ema(clients[idx].missed_rounds, round, max_training_time)
        if client_cluster_idx in cluster_number_map:
            old = cluster_number_map[client_cluster_idx]
            old[1].append(clients[idx])
            cluster_number_map[client_cluster_idx] = (old[0] + total_ema, old[1])
        else:
            cluster_number_map[client_cluster_idx] = (total_ema, [clients[idx]])
    for cluster_idx, (cluster_total_ema, cluster_clients) in cluster_number_map.items():
        cluster_number_map[cluster_idx] = (cluster_total_ema / len(cluster_clients), cluster_clients)
    return dict(sorted(cluster_number_map.items(), key=lambda x: x[1][0]))


def sample_starting_from(sorted_clusters, start_cluster_idx: int, n_clients_from_clustering: int) -> list:
    """Intelligent_selection.py:401-427."""
    cluster_list = list(sorted_clusters.items())
    returned_samples = []
    while n_clients_from_clustering > 0:
        cluster_clients = cluster_list[start_cluster_idx][1][1]
        cluster_size = len(cluster_clients)
        if cluster_size >= n_clients_from_clustering:
            cluster_clients_sorted = sorted(
                cluster_clients, key=lambda client: len(client.training_times) + len(client.missed_rounds)
            )
            returned_samples += cluster_clients_sorted[:n_clients_from_clustering]
            n_clients_from_clustering = 0
        else:
            n_clients_from_clustering -= cluster_size
            returned_samples += cluster_clients
        start_cluster_idx = (start_cluster_idx + 1) % len(cluster_list)
    return returned_samples


class FedLesScan:
    """Selection (DBScanClientSelection.db_fit), controller bookkeeping and stall-aware aggregation."""

    def __init__(self, client_ids: Sequence[int], n_clients_in_round: int, max_rounds: int, seed: int,
                 allowed_stragglers: int, tolerance: int = STALENESS_TOLERANCE) -> None:
        self.client_ids = sorted(int(c) for c in client_ids)
        self.n_clients_in_round = int(n_clients_in_round)
        self.max_rounds = int(max_rounds)
        self.allowed_stragglers = int(allowed_stragglers)
        self.tolerance = int(tolerance)
        # Each client's history is created once, in client order, as scripts.py:276-286 does.
        self.histories: Dict[int, ClientHistory] = {c: ClientHistory(client_id=c) for c in self.client_ids}
        self.start_clustering_round = -1  # Intelligent_selection.py:60
        # The official code draws every selection sample from one stream (the global `random`); here it is one
        # seeded generator. (resnet50_fedlesscan also drew its fixed crashers from this stream; this experiment has none.)
        self.selection_rng = random.Random(seed)
        # Stored client results awaiting aggregation (the official ClientResultDao collection).
        self.results: List[ClientResult] = []
        # Late updates in flight: dicts with client_id, round_id, arrival_round, cardinality, parameters, minutes.
        self.in_flight: List[dict] = []
        self.job_counts: Dict[int, int] = {c: 0 for c in self.client_ids}  # dispatches per client (trace index)

    def next_job_index(self, client_id: int) -> int:
        k = self.job_counts[client_id]
        self.job_counts[client_id] = k + 1
        return k

    def launch_late(self, client_id: int, round_id: int, arrival_round: int, cardinality: int,
                    parameters: List[np.ndarray], minutes: float) -> None:
        self.in_flight.append({"client_id": int(client_id), "round_id": int(round_id), "arrival_round": int(arrival_round),
                               "cardinality": int(cardinality), "parameters": parameters, "minutes": float(minutes)})

    def arrivals(self, round: int) -> List[dict]:
        """Late jobs whose simulated finish falls in this round (removed from the in-flight list)."""
        due = [j for j in self.in_flight if j["arrival_round"] <= round]
        self.in_flight = [j for j in self.in_flight if j["arrival_round"] > round]
        return due

    # ---- selection ------------------------------------------------------------------------

    def select(self, round: int) -> Tuple[List[int], dict]:
        """Intelligent_selection.py:252-353 (db_fit). Returns selected ids (in pool order) and a log."""
        all_data = [self.histories[c] for c in self.client_ids]
        rookie_clients, rest_clients, stragglers = filter_rookies(all_data, round)
        info = {
            "round0": round,
            "tiers": {"rookies": [c.client_id for c in rookie_clients],
                      "participants": [c.client_id for c in rest_clients],
                      "stragglers": [c.client_id for c in stragglers]},
            "clustering": None,
        }
        if len(rookie_clients) >= self.n_clients_in_round:
            chosen = self.selection_rng.sample(rookie_clients, self.n_clients_in_round)
            info["from"] = {"rookies": [c.client_id for c in chosen], "clusters": [], "stragglers": []}
            return self._from_pool(chosen), info

        n_clients_from_clustering = min(self.n_clients_in_round - len(rookie_clients), len(rest_clients))
        if n_clients_from_clustering > 0 and self.start_clustering_round == -1:
            self.start_clustering_round = round
        n_from_stragglers = self.n_clients_in_round - (n_clients_from_clustering + len(rookie_clients))
        round_stragglers = self.selection_rng.sample(stragglers, n_from_stragglers)

        max_training_time = max_training_time_of(rest_clients)
        training_data = []
        for client_data in rest_clients:
            ema = get_client_ema(client_data.training_times)
            missed_rounds_ema = get_missed_rounds_ema(client_data.missed_rounds, round, max_training_time)
            client_data.latest_updated = round
            training_data.append([ema, missed_rounds_ema])

        labels, best_score, best_eps = perform_clustering(training_data=training_data, eps_step=EPS_STEP)
        sorted_clusters = sort_clusters(rest_clients, labels, round)

        cluster_idx_list = np.arange(start=0, stop=len(sorted_clusters))
        perc = ((round - self.start_clustering_round) / max(self.max_rounds - self.start_clustering_round, 1)) * 100
        # Official: np.percentile(..., interpolation="lower"); `method` is the same option's current name.
        start_cluster_idx = int(np.percentile(cluster_idx_list, perc, method="lower"))
        cluster_clients = sample_starting_from(sorted_clusters, start_cluster_idx, n_clients_from_clustering)

        chosen = rookie_clients + cluster_clients + round_stragglers
        info["from"] = {"rookies": [c.client_id for c in rookie_clients],
                        "clusters": [c.client_id for c in cluster_clients],
                        "stragglers": [c.client_id for c in round_stragglers]}
        info["clustering"] = {
            "num_clusters": len(sorted_clusters), "score": float(best_score), "eps": float(best_eps),
            "start_clustering_round0": self.start_clustering_round, "progress_percent": float(perc),
            "start_cluster_idx": start_cluster_idx, "max_training_time": float(max_training_time),
            "features": {c.client_id: {"training_ema": float(f[0]), "missed_round_ema": float(f[1])}
                         for c, f in zip(rest_clients, training_data)},
            "clusters": [
                {"order": pos, "label": int(label), "cluster_ema": float(ema_), "clients": [c.client_id for c in members]}
                for pos, (label, (ema_, members)) in enumerate(sorted_clusters.items())
            ],
        }
        return self._from_pool(chosen), info

    def _from_pool(self, chosen: Sequence[ClientHistory]) -> List[int]:
        """Intelligent_selection.py:355-360: the pool is filtered, so pool (client id) order is kept."""
        ids = {c.client_id for c in chosen}
        return [c for c in self.client_ids if c in ids]

    # ---- controller and client bookkeeping ---------------------------------------------------

    def record_client_success(self, client_id: int, round: int, elapsed_minutes: float, cardinality: int) -> None:
        """client.py:213-227: the client appends its time and clears this round from missed rounds."""
        h = self.histories[client_id]
        h.training_times.append(float(elapsed_minutes))
        h.train_cardinality = int(cardinality)
        if round in h.missed_rounds:
            h.missed_rounds.remove(round)

    def finish_round(self, round: int, selected: Sequence[int], succeeded: Sequence[int]) -> dict:
        """serverless_strategy.py:252-293: reset successes, add missed round and backoff for the rest."""
        all_clients = set(selected)
        invocation = {"succs": len(succeeded), "failed": len(all_clients) - len(succeeded), "pending": 0}
        for cid in succeeded:
            self.histories[cid].client_backoff = 0
            all_clients.discard(cid)
        for cid in sorted(all_clients):
            h = self.histories[cid]
            h.missed_rounds.append(round)
            if h.client_backoff <= 0:
                h.client_backoff = 1
            else:
                h.client_backoff *= 2
        if len(succeeded) < (len(selected) - self.allowed_stragglers):
            raise RuntimeError(
                f"Only {len(succeeded)}/{len(selected)} clients finished this round, "
                f"required are {len(selected) - self.allowed_stragglers}."
            )
        return invocation

    # ---- staleness-aware aggregation (stall_aware_aggregation.py) ---------------------------

    def store_result(self, result: ClientResult) -> None:
        self.results.append(result)

    def aggregate(self, current_round: int) -> Tuple[Optional[List[np.ndarray]], List[dict]]:
        """stall_aware_aggregation.py:34-67 and client_daos.py:165-180, then delete (aggregation.py:143-152).

        Candidates: every stored result with round_id >= current_round - tolerance.
        w = sum_k n_k * (t_k + 1)/(t + 1) * w_k / sum_k n_k, with no renormalisation.
        """
        candidates = [r for r in self.results if r.round_id >= current_round - self.tolerance]
        if not candidates:
            self.results = []
            return None, []   # novelty_env: official code raises; here the global model is kept
        num_examples_total = sum(r.cardinality for r in candidates)
        scores = [(r.round_id + 1) / (current_round + 1) for r in candidates]
        weighted_weights = [
            [layer * r.cardinality * score for layer in r.parameters] for r, score in zip(candidates, scores)
        ]
        weights_prime = [reduce(np.add, layer_updates) / num_examples_total for layer_updates in zip(*weighted_weights)]
        record = [
            {"client_id": r.client_id, "round_id": r.round_id, "staleness": current_round - r.round_id,
             "score": s, "cardinality": r.cardinality, "effective_weight": r.cardinality * s / num_examples_total}
            for r, s in zip(candidates, scores)
        ]
        # AggregationStrategy.PER_SESSION deletes every result of the session after aggregating.
        self.results = []
        return weights_prime, record
