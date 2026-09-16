"""
client/client.py
=================
Phase 3A — Flower Client Wrapper.

This module wraps the already-refactored, already-verified reusable
training components from client/local_trainer.py (Phase 2) inside a
`flwr.client.NumPyClient`. It introduces ZERO new training or evaluation
math: every numeric operation (forward pass, loss, backward pass,
optimizer step, metric computation) is delegated to functions that were
already unit-tested against the original centralized train.py /
evaluate.py in Phase 2.

Scope (Phase 3A only)
----------------------
IN scope:
  - FedMelanomaClient(NumPyClient): get_parameters / set_parameters /
    fit / evaluate
  - Minimal per-hospital DataLoader construction, mirroring exactly how
    train.py's main() builds DataLoaders (same MelanomaDataset class,
    same batch size, same shuffle flags, same num_workers/pin_memory) —
    the only difference is that one federated client trains on ONE
    hospital's partition instead of the ConcatDataset of all ten.
  - build_client(): a factory that instantiates a FedMelanomaClient for
    a given hospital, so this file is self-contained and testable.

OUT of scope (explicitly NOT implemented here):
  - ServerApp
  - Strategy (FedAvg or the custom FedMelanoma aggregation in
    server/aggregation.py)
  - Client selection (server/selector.py is untouched and NOT invoked
    from this file)
  - Straggler handling (server/straggler.py is untouched and NOT invoked
    from this file)
  - Any network communication (no ClientApp, no start_client /
    start_numpy_client). This file defines the Client class only; wiring
    it into a runnable Flower app is left to a later phase alongside the
    ServerApp.

Untouched research files
-------------------------
This module imports, but never modifies, the following unchanged
research modules:
  - client/dataset.py          (MelanomaDataset)
  - client/focal_loss.py       (ClassWeightedFocalLoss, compute_class_weights)
  - models/model.py            (FedMelanomaModel)
  - client/local_trainer.py    (train_one_epoch, evaluate_model,
                                 build_optimizer, configure_multiprocessing_env)
selector.py, aggregation.py, straggler.py, emd.py are not imported here
at all — this file has no client-selection or aggregation concerns.
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from typing import Dict, Iterator, List, Tuple

import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

import flwr as fl
from flwr.common import Context, NDArrays, Scalar

from client.dataset import MelanomaDataset
from client.focal_loss import ClassWeightedFocalLoss, compute_class_weights
from models.model import FedMelanomaModel
from client.local_trainer import (
    configure_multiprocessing_env,
    build_optimizer,
    train_one_epoch,
    evaluate_model,
)

# Adapter/monitoring modules (Phase 5) -- NOT research modules. Provide
# parameter fingerprinting (to prove the client trained on exactly the
# broadcast global parameters) and local, privacy-preserving subgroup
# metric computation (confusion counts only -- never raw samples/labels).
from monitoring.param_hash import hash_ndarrays
from monitoring.subgroup_metrics import local_all_axes_confusion_counts

# ---------------------------------------------------------------------------
# Defaults mirrored EXACTLY from train.py's module-level hyperparameters, so
# that one fit() round (at local_epochs=1) reproduces one epoch of the
# centralized loop bit-for-bit. Callers may override any of these via the
# constructor, or override local_epochs per-round via Flower's `config` dict
# (config["local_epochs"]) once a ServerApp/Strategy exists to send it.
# ---------------------------------------------------------------------------
DEFAULT_LOCAL_EPOCHS: int = 1        # per-round epoch count; Blueprint canonical E=5
                                      # is a ServerApp/Strategy round-scheduling concern,
                                      # not something this client-only phase hardcodes.
DEFAULT_BATCH_SIZE: int = 20         # == train.py BATCH_SIZE
DEFAULT_LEARNING_RATE: float = 1e-4  # == train.py LEARNING_RATE
DEFAULT_WEIGHT_DECAY: float = 1e-4   # == train.py WEIGHT_DECAY
DEFAULT_FOCAL_GAMMA: float = 2.0     # == train.py FOCAL_GAMMA
POSITIVE_CLASS_IDX: int = 1          # == train.py / evaluate.py POSITIVE_CLASS_IDX


# ===========================================================================
# Per-hospital DataLoader construction
# ===========================================================================

def build_hospital_dataloaders(
    hospital_name: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    partitions_root: str = "hospital_partitions",
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/test DataLoaders for a single hospital partition.

    Mirrors train.py's main() dataset construction (same MelanomaDataset
    class, same CSV/image-directory path convention, same batch size,
    same shuffle flags, same num_workers/pin_memory) — scoped to one
    hospital instead of ConcatDataset-ing all ten, since a federated
    client only ever sees its own local partition.
    """
    configure_multiprocessing_env()

    base = f"{partitions_root}/{hospital_name}"
    train_dataset = MelanomaDataset(
        f"{base}/metadata_train.csv", f"{base}/images/Train"
    )
    test_dataset = MelanomaDataset(
        f"{base}/metadata_test.csv", f"{base}/images/Test"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, test_loader


def _iter_underlying_datasets(dataset) -> Iterator[MelanomaDataset]:
    """Yield the MelanomaDataset(s) backing a (possibly ConcatDataset) loader dataset."""
    if isinstance(dataset, ConcatDataset):
        yield from dataset.datasets
    else:
        yield dataset


_METADATA_COLUMNS = ["fitzpatrick_skin_type", "image_type", "diagnosis_confirm_type"]


def _extract_metadata_df(loader: DataLoader) -> pd.DataFrame:
    """
    Row-order-aligned metadata for every sample the loader will yield,
    used only for LOCAL subgroup metric computation (Requirement 4/5) --
    never transmitted. Valid only when the loader is not shuffled (both
    build_hospital_dataloaders' test_loader and evaluate.py's test loader
    use shuffle=False), so iteration order here matches
    evaluate_model()/run_inference()'s y_true/y_pred/y_prob order exactly.
    """
    frames = [ds.df[_METADATA_COLUMNS] for ds in _iter_underlying_datasets(loader.dataset)]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0].reset_index(drop=True)


def _compute_local_class_weights(train_loader: DataLoader, device: torch.device) -> torch.Tensor:
    """
    Compute this client's local class weights via the untouched
    compute_class_weights() from focal_loss.py, from labels visible only
    to this client's train_loader.

    This mirrors exactly how train.py derives class_weights from
    dataset.df["diagnosis_1"], and satisfies Blueprint §3.4.3's privacy
    constraint: "raw class counts are never transmitted" — the weights
    are computed locally and never leave this client object.
    """
    train_targets = torch.tensor(
        [
            1 if str(label).lower() == "malignant" else 0
            for ds in _iter_underlying_datasets(train_loader.dataset)
            for label in ds.df["diagnosis_1"]
        ]
    )
    return compute_class_weights(train_targets).to(device)


# ===========================================================================
# FedMelanomaClient
# ===========================================================================

class FedMelanomaClient(fl.client.NumPyClient):
    """
    Flower NumPyClient wrapping the existing centralized local-training
    pipeline for one hospital partition.

    This class contains NO training or evaluation math of its own. It only:
      - marshals parameters between Flower's wire format (list[np.ndarray])
        and the model's state_dict (get_parameters / set_parameters)
      - builds a fresh optimizer + criterion each round and calls the
        already-verified train_one_epoch() / evaluate_model() from
        client/local_trainer.py
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        local_epochs: int = DEFAULT_LOCAL_EPOCHS,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        focal_gamma: float = DEFAULT_FOCAL_GAMMA,
        positive_class_idx: int = POSITIVE_CLASS_IDX,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.focal_gamma = focal_gamma
        self.positive_class_idx = positive_class_idx

        # Local-only class weights (Blueprint §3.4.3 privacy constraint —
        # computed here, never transmitted to the server).
        self.class_weights = _compute_local_class_weights(train_loader, device)

    # ------------------------------------------------------------------
    # Parameter marshaling
    # ------------------------------------------------------------------

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """
        Return model weights as a list of NumPy ndarrays (Flower's wire format).

        `.copy()` is required: `.cpu().numpy()` alone returns an array that
        ALIASES the model's live parameter memory (zero-copy view). Without
        it, a snapshot returned here would silently change if the model is
        later mutated in place (e.g. by a subsequent fit() call's
        optimizer.step(), or by a later set_parameters() call) — violating
        the Flower contract that get_parameters() returns an independent,
        stable snapshot the caller can hold onto.
        """
        return [val.detach().cpu().numpy().copy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Load a list of NumPy ndarrays (as received from the server) into the model."""
        param_names = self.model.state_dict().keys()
        state_dict = OrderedDict(
            {name: torch.tensor(value) for name, value in zip(param_names, parameters)}
        )
        self.model.load_state_dict(state_dict, strict=True)

    # ------------------------------------------------------------------
    # Local training — delegates to client.local_trainer.train_one_epoch()
    # ------------------------------------------------------------------

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        One federated round of local training on this client's hospital
        partition.

        Numerically identical, per epoch, to train.py's centralized loop:
        same criterion (ClassWeightedFocalLoss), same optimizer
        construction (build_optimizer -> AdamW with train.py's LR/weight
        decay), same train_one_epoch() call — just scoped to this
        client's own data and repeated for `local_epochs` epochs instead
        of train.py's NUM_EPOCHS.

        The optimizer is rebuilt fresh every call (Blueprint §4.3.3:
        "AdamW momentum and variance buffers are reset at the start of
        each round's local training") using the untouched
        build_optimizer() from Phase 2.

        Phase 5 addition: hashes `parameters` BEFORE set_parameters() is
        called and echoes that hash back in the returned metrics
        (`client_received_param_hash`). This is what lets the server
        prove -- not assume -- that this client actually trained on the
        exact global parameters it broadcast that round: the server's own
        copy of what it *sent* can be wrong in ways a client-independent
        echo cannot (e.g. a stale cached reference); comparing the
        server's broadcast hash against this client-computed hash closes
        that gap.
        """
        received_hash = hash_ndarrays(parameters)
        self.set_parameters(parameters)

        optimizer = build_optimizer(self.model, self.learning_rate, self.weight_decay)
        criterion = ClassWeightedFocalLoss(gamma_f=self.focal_gamma, reduction="mean")

        local_epochs = int(config.get("local_epochs", self.local_epochs))

        avg_loss = 0.0
        for _ in range(local_epochs):
            epoch_start = time.time()
            avg_loss = train_one_epoch(
                self.model,
                self.train_loader,
                criterion,
                optimizer,
                self.class_weights,
                epoch_start,
                self.device,
            )

        updated_parameters = self.get_parameters(config={})
        num_examples = len(self.train_loader.dataset)
        metrics: Dict[str, Scalar] = {
            "train_loss": float(avg_loss),
            "client_received_param_hash": received_hash,
            "client_returned_param_hash": hash_ndarrays(updated_parameters),
        }

        return updated_parameters, num_examples, metrics

    # ------------------------------------------------------------------
    # Local validation — delegates to client.local_trainer.evaluate_model()
    # ------------------------------------------------------------------

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Validate the given (global) parameters on this client's local
        test partition.

        Delegates to client.local_trainer.evaluate_model(), which is the
        same validation computation as train.py's centralized evaluate()
        (identical metrics: accuracy, precision, recall, f1, roc_auc).

        NOTE on the returned `loss` scalar: the centralized pipeline
        never computes an explicit validation loss value — it uses
        ROC-AUC as its sole model-selection criterion (see train.py's
        checkpointing logic). Flower's Client.evaluate() contract
        requires a float loss regardless. `loss = 1.0 - roc_auc` is used
        here purely as an order-preserving placeholder to satisfy that
        protocol requirement; it is derived from, not additional to, the
        centralized metrics, and it is not consumed by any research
        algorithm in this phase (no Strategy/aggregation exists yet to
        read it). All real validation metrics are returned unmodified in
        `metrics`.

        Phase 5 additions (all still local-only computation; only small
        aggregate numbers are added to the returned metrics dict, never
        raw predictions/metadata):
          - `client_received_param_hash`: same broadcast-verification
            echo as fit() above.
          - A few extra standard metrics (sensitivity/specificity/ppv/
            npv/balanced_accuracy/mcc/pr_auc), computed by calling the
            SAME evaluate.compute_metrics() the centralized pipeline
            uses (reused, not reimplemented), from the y_true/y_pred/
            y_prob already computed locally by evaluate_model() above.
          - `subgroup_confusion_counts_json`: this client's LOCAL
            per-category confusion-matrix counts (tn/fp/fn/tp -- small
            integers) for the blueprint's three metadata axes (skin
            type, imaging modality, confirmation type), via
            monitoring.subgroup_metrics.local_all_axes_confusion_counts.
            Privacy constraint: raw metadata/images/labels never leave
            this method; only these aggregate counts are serialised.
        """
        from evaluate import compute_metrics  # local import: reuse, not duplicate

        received_hash = hash_ndarrays(parameters)
        self.set_parameters(parameters)

        metrics, y_true, y_pred, y_prob = evaluate_model(
            self.model,
            self.test_loader,
            self.device,
            positive_class_idx=self.positive_class_idx,
        )

        loss = 1.0 - float(metrics["roc_auc"])
        num_examples = len(self.test_loader.dataset)

        full_metrics = compute_metrics(y_true, y_pred, y_prob)
        result_metrics: Dict[str, Scalar] = {
            "accuracy":  float(metrics["accuracy"]),
            "precision": float(metrics["precision"]),
            "recall":    float(metrics["recall"]),
            "f1":        float(metrics["f1"]),
            "roc_auc":   float(metrics["roc_auc"]),
            "sensitivity":       float(full_metrics["sensitivity"]),
            "specificity":       float(full_metrics["specificity"]),
            "ppv":               float(full_metrics["ppv"]),
            "npv":               float(full_metrics["npv"]),
            "balanced_accuracy": float(full_metrics["balanced_accuracy"]),
            "mcc":               float(full_metrics["mcc"]),
            "pr_auc":            float(full_metrics["pr_auc"]),
            "client_received_param_hash": received_hash,
        }

        try:
            metadata_df = _extract_metadata_df(self.test_loader)
            subgroup_counts = local_all_axes_confusion_counts(
                y_true, y_pred, metadata_df, positive_class_idx=self.positive_class_idx
            )
            result_metrics["subgroup_confusion_counts_json"] = json.dumps(subgroup_counts)
        except Exception as exc:  # local-only diagnostics must never fail a round
            result_metrics["subgroup_confusion_counts_json"] = json.dumps({})
            result_metrics["subgroup_metrics_error"] = str(exc)

        return loss, num_examples, result_metrics


# ===========================================================================
# Factory
# ===========================================================================

def build_client(
    hospital_name: str,
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
    local_epochs: int = DEFAULT_LOCAL_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    focal_gamma: float = DEFAULT_FOCAL_GAMMA,
    partitions_root: str = "hospital_partitions",
) -> FedMelanomaClient:
    """
    Convenience factory: build a FedMelanomaClient for a given hospital
    partition, wiring together the (untouched) FedMelanomaModel, this
    hospital's DataLoaders, and the client wrapper above.

    Not a ClientApp / client_fn for Flower's runtime — that wiring
    (needed to actually run simulations or a ServerApp-driven session)
    is deliberately left for the phase that introduces the ServerApp,
    per the Phase 3A scope.
    """
    model = FedMelanomaModel().to(device)
    train_loader, test_loader = build_hospital_dataloaders(
        hospital_name, batch_size=batch_size, partitions_root=partitions_root
    )
    return FedMelanomaClient(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        focal_gamma=focal_gamma,
    )


# ===========================================================================
# ClientApp -- Phase 4 Flower runtime wiring
# ===========================================================================
#
# Everything above this point is unchanged Phase 3A code. The ClientApp
# entry point below adds NO new training/evaluation math of its own -- it
# only resolves "which hospital is this simulated client node" from
# Flower's own per-node partition-id config, then delegates entirely to
# the existing build_client() factory above.
#
# No hospital name is hardcoded here: hospitals are discovered dynamically
# via hospital_registry.discover_hospitals() (repo-root adapter module),
# so this file has no knowledge of any specific hospital and needs no
# changes when the dataset is added or hospitals are renamed/added/removed.

DEFAULT_PARTITIONS_ROOT: str = "hospital_partitions"


def client_fn(context: Context) -> fl.client.Client:
    """
    Flower ClientApp entry point (flwr>=1.8 API).

    Resolution order:
      1. context.node_config["partitions-root"] (falls back to
         hospital_registry.DEFAULT_PARTITIONS_ROOT) tells us where to look.
      2. hospital_registry.discover_hospitals() enumerates the hospital
         folders there, sorted -- the same deterministic order
         server/server.py's server_fn uses to build FedMelanomaStrategy's
         per-hospital client list, so partition-id N here and
         selector.Client(client_id=N) on the server side always refer to
         the same hospital.
      3. context.node_config["partition-id"] (set by Flower's simulation
         runtime, one value per simulated client node) indexes into that
         list to pick this node's hospital.

    Raises only when actually invoked without a dataset present -- import
    of this module, and construction of `app` below, do not require
    hospital_partitions/ to exist.
    """
    from hospital_registry import discover_hospitals

    partitions_root = str(context.node_config.get("partitions-root", DEFAULT_PARTITIONS_ROOT))
    hospitals = discover_hospitals(partitions_root)
    if not hospitals:
        raise RuntimeError(
            f"No hospital partitions discovered under '{partitions_root}'. "
            "client_fn requires hospital_partitions/<hospital>/... to exist "
            "before a ClientApp node can be started."
        )

    partition_id = int(context.node_config["partition-id"])
    if not 0 <= partition_id < len(hospitals):
        raise ValueError(
            f"partition-id={partition_id} is out of range for "
            f"{len(hospitals)} discovered hospitals under '{partitions_root}'."
        )
    hospital_name = hospitals[partition_id]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_epochs = int(context.run_config.get("local-epochs", DEFAULT_LOCAL_EPOCHS))

    numpy_client = build_client(
        hospital_name=hospital_name,
        device=device,
        local_epochs=local_epochs,
        partitions_root=partitions_root,
    )
    return numpy_client.to_client()


# Building the ClientApp only registers client_fn as a callback -- it does
# NOT invoke it, so this line succeeds whether or not hospital_partitions/
# exists.
app = fl.client.ClientApp(client_fn=client_fn)
