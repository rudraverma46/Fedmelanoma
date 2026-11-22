"""In-process Flower transport (difference D2) and the instrumented client (difference D9).

The Ray simulation is replaced by a ClientProxy that calls the client directly. Parameters still
pass through Flower's NumPyClient wrapper, so ndarrays are serialised and deserialised exactly
as they are for any other transport, and the strategy cannot tell the difference.

Proxies are registered with cid "00".."09", so FedMelanomaStrategy's sorted(cid) -> client_id
mapping is the identity: client_id k is hospital k (difference D3).
"""
from __future__ import annotations

import time
from typing import Dict, List

import torch
from flwr.common import (Code, DisconnectRes, EvaluateIns, EvaluateRes, FitIns, FitRes, GetParametersIns,
                         GetParametersRes, GetPropertiesIns, GetPropertiesRes, ReconnectIns, Status)
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy

from client.client import FedMelanomaClient
from client.focal_loss import ClassWeightedFocalLoss
from client.local_trainer import configure_multiprocessing_env, run_inference
from runtime.image_cache import build_cached_dataset, build_loader


class InstrumentedClient(FedMelanomaClient):
    """FedMelanomaClient plus one read-only diagnostic: focal loss on the local test partition.

    The reference's evaluation "loss" is 1 - ROC-AUC. This adds the real loss for the evaluation-loss
    curve. It uses the parent's parameters as loaded, and nothing reads it back into training,
    selection, aggregation or checkpoint choice.
    """

    def evaluate(self, parameters, config):
        loss, num_examples, metrics = super().evaluate(parameters, config)
        y_true, _, y_prob = run_inference(self.model, self.test_loader, self.device)
        prob = torch.tensor(y_prob, dtype=torch.float32).clamp(1e-8, 1 - 1e-8)
        logits = torch.stack([torch.log1p(-prob), torch.log(prob)], dim=1)
        criterion = ClassWeightedFocalLoss(gamma_f=self.focal_gamma, reduction="mean")
        focal = criterion(logits, torch.tensor(y_true), class_weights=self.class_weights.cpu())
        metrics["eval_focal_loss"] = float(focal)
        return loss, num_examples, metrics


class InProcessClientProxy(ClientProxy):
    """Delivers Flower instructions to a client in this process and logs per-client measurements."""

    def __init__(self, cid: str, client_id: int, hospital: str, client, clients_log) -> None:
        super().__init__(cid)
        self.client_id = client_id
        self.hospital = hospital
        self.client = client
        self.clients_log = clients_log

    def fit(self, ins: FitIns, timeout, group_id) -> FitRes:
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        res = self.client.fit(ins)
        self.clients_log({
            "event": "client_fit", "round": group_id, "client_id": self.client_id, "hospital": self.hospital,
            "num_examples": res.num_examples, "train_loss": res.metrics.get("train_loss"),
            "wall_seconds": time.time() - start,
            "gpu_peak_mb": torch.cuda.max_memory_allocated() / 2**20,
        })
        return res

    def evaluate(self, ins: EvaluateIns, timeout, group_id) -> EvaluateRes:
        start = time.time()
        res = self.client.evaluate(ins)
        numeric = {k: v for k, v in res.metrics.items() if isinstance(v, (int, float))}
        self.clients_log({"event": "client_evaluate", "round": group_id, "client_id": self.client_id,
                          "hospital": self.hospital, "num_examples": res.num_examples,
                          "wall_seconds": time.time() - start, **numeric})
        return res

    def get_properties(self, ins: GetPropertiesIns, timeout, group_id) -> GetPropertiesRes:
        return GetPropertiesRes(status=Status(code=Code.OK, message=""), properties={"hospital": self.hospital})

    def get_parameters(self, ins: GetParametersIns, timeout, group_id) -> GetParametersRes:
        return self.client.get_parameters(ins)

    def reconnect(self, ins: ReconnectIns, timeout, group_id) -> DisconnectRes:
        return DisconnectRes(reason="")


def build_federation(cfg: Dict, hospitals: List[str], model: torch.nn.Module, device: torch.device,
                     clients_log) -> SimpleClientManager:
    """Ten clients sharing one model object (difference D6), registered in hospital order."""
    configure_multiprocessing_env()
    data, local, run = cfg["data"], cfg["local_training"], cfg["runtime"]
    manager = SimpleClientManager()
    for client_id, hospital in enumerate(hospitals):
        loaders = {}
        for split, shuffle in (("train", True), ("test", False)):
            dataset = build_cached_dataset(data["partitions_root"], data["cache_dir"], hospital, split)
            loaders[split] = build_loader(dataset, local["batch_size"], shuffle, run["dataloader_workers"], run["pin_memory"])
        numpy_client = InstrumentedClient(
            model=model, train_loader=loaders["train"], test_loader=loaders["test"], device=device,
            local_epochs=cfg["federation"]["local_epochs"], learning_rate=local["learning_rate"],
            weight_decay=local["weight_decay"], focal_gamma=local["focal_gamma"],
        )
        cid = f"{client_id:02d}"
        manager.register(InProcessClientProxy(cid, client_id, hospital, numpy_client.to_client(), clients_log))
    return manager


def assert_identity_mapping(strategy, hospitals: List[str]) -> None:
    """The strategy's lazily built cid map must be hospital k -> client_id k."""
    for client_id in range(len(hospitals)):
        if strategy._client_id_to_cid[client_id] != f"{client_id:02d}":
            raise RuntimeError("client/hospital mapping is not the identity")

