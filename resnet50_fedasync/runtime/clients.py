"""The ten hospital clients, built exactly as resnet50_fedavg_textbook/runtime/in_process.py builds them,
without the Flower ClientProxy layer (an asynchronous server does not run Flower's synchronous rounds).

* Same FedMelanomaClient (here the resnet50_fedprox_textbook copy, which adds only the optional proximal
  term; FedAsync's worker objective is f + (rho/2)||x - x_tau||^2, so proximal_mu = rho).
* Same pixel-identical image cache, batch size 20, shuffle flags, 0 workers, pinned memory.
* Same InstrumentedClient.evaluate (adds the focal loss on the local test partition, read-only).
* One shared model object for all clients (difference D6 of the baseline), client k = hospital k.

Calling NumPyClient.fit/evaluate directly instead of through Flower's message path is numerically
identical: resnet50_fedavg_textbook test 19 proved the two routes return the same bytes.
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import torch

from client.client import FedMelanomaClient
from client.focal_loss import ClassWeightedFocalLoss
from client.local_trainer import configure_multiprocessing_env, run_inference
from runtime.image_cache import build_cached_dataset, build_loader


class InstrumentedClient(FedMelanomaClient):
    """resnet50_fedavg_textbook/runtime/in_process.py InstrumentedClient, unchanged."""

    def evaluate(self, parameters, config):
        loss, num_examples, metrics = super().evaluate(parameters, config)
        y_true, _, y_prob = run_inference(self.model, self.test_loader, self.device)
        prob = torch.tensor(y_prob, dtype=torch.float32).clamp(1e-8, 1 - 1e-8)
        logits = torch.stack([torch.log1p(-prob), torch.log(prob)], dim=1)
        criterion = ClassWeightedFocalLoss(gamma_f=self.focal_gamma, reduction="mean")
        focal = criterion(logits, torch.tensor(y_true), class_weights=self.class_weights.cpu())
        metrics["eval_focal_loss"] = float(focal)
        return loss, num_examples, metrics


def smoke_rows(n: int, fraction: float, batch_size: int = 20) -> int:
    """Smoke tests only: a fraction of a hospital's training rows, at least one batch, and never a final
    batch of one row (BatchNorm1d cannot train on it). Sizes stay proportional to the hospitals' sizes, so
    fit times, and therefore arrival order, keep the real run's pattern."""
    rows = max(batch_size, int(n * fraction))
    return rows - 1 if rows % batch_size == 1 else rows


def truncate_dataset(dataset, n: int) -> None:
    """Smoke tests only: keep the first n rows (images stay aligned, the cache is indexed by row)."""
    dataset.df = dataset.df.iloc[:n].reset_index(drop=True)
    dataset.rows = dataset.rows[:n]
    dataset.labels = dataset.labels[:n]
    dataset.metadata_cache = dataset.metadata_cache[:n]


def build_clients(cfg: Dict, hospitals: List[str], model: torch.nn.Module, device: torch.device,
                  proximal_mu: float = 0.0, smoke_fraction: float = 0.0) -> List[InstrumentedClient]:
    configure_multiprocessing_env()
    data, local, run = cfg["data"], cfg["local_training"], cfg["runtime"]
    clients = []
    for hospital in hospitals:
        loaders = {}
        for split, shuffle in (("train", True), ("test", False)):
            dataset = build_cached_dataset(data["partitions_root"], data["cache_dir"], hospital, split)
            if smoke_fraction and split == "train":   # smoke tests only; test partitions stay complete
                truncate_dataset(dataset, smoke_rows(len(dataset), smoke_fraction, local["batch_size"]))
            loaders[split] = build_loader(dataset, local["batch_size"], shuffle, run["dataloader_workers"], run["pin_memory"])
        kwargs = dict(model=model, train_loader=loaders["train"], test_loader=loaders["test"], device=device,
                      local_epochs=cfg["federation"]["local_epochs"], learning_rate=local["learning_rate"],
                      weight_decay=local["weight_decay"], focal_gamma=local["focal_gamma"])
        if proximal_mu:
            kwargs["proximal_mu"] = proximal_mu
        clients.append(InstrumentedClient(**kwargs))
    return clients


def timed_fit(client: InstrumentedClient, arrays: List[np.ndarray], local_epochs: int) -> Tuple[List[np.ndarray], int, Dict, float, float]:
    """Train one client from `arrays`; return (arrays, n, metrics, wall seconds, peak GPU MB).

    The wall time covers loading the global model, all local epochs and returning the weights, with the
    GPU synchronised at both ends, so it is the client's real compute time on this machine.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    new_arrays, n, metrics = client.fit(arrays, {"local_epochs": local_epochs})
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
    return new_arrays, n, metrics, seconds, peak
