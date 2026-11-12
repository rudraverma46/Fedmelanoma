"""Tests P1-P9: the proximal term, on small models and on the real model.

Each test targets one way FedProx is commonly implemented wrongly: the wrong norm, a reference
that moves, a reference taken from the wrong point, or BatchNorm buffers counted as parameters.
"""
import copy
import importlib.util
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
from flwr.common import FitIns, ndarrays_to_parameters, parameters_to_ndarrays
from torch.utils.data import DataLoader, TensorDataset

from client.client import FedMelanomaClient
from client.focal_loss import ClassWeightedFocalLoss
from client.local_trainer import build_optimizer, train_one_epoch
from client.proximal import parameter_distance, proximal_penalty, snapshot_global_reference, trainable_parameters

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEDAVG = os.path.abspath(os.path.join(ROOT, "..", "resnet50_fedavg_textbook"))
CPU = torch.device("cpu")


class Tiny(nn.Module):
    """Same call signature as FedMelanomaModel, with BatchNorm and dropout."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(13, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Dropout(0.3), nn.Linear(8, 2))

    def forward(self, images, meta):
        return self.net(meta)


def tiny_loader(seed=0, n=40):
    generator = torch.Generator().manual_seed(seed)
    meta = torch.randn(n, 13, generator=generator)
    labels = (torch.arange(n) % 4 == 0).long()
    return DataLoader(TensorDataset(torch.zeros(n, 1), meta, labels), batch_size=10, shuffle=True)


def weights():
    return torch.tensor([0.25, 0.75])


def tiny_client(mu, seed=0):
    torch.manual_seed(seed)
    model = Tiny()
    loader = tiny_loader(seed)
    client = FedMelanomaClient.__new__(FedMelanomaClient)
    client.model, client.train_loader, client.test_loader, client.device = model, loader, loader, CPU
    client.local_epochs, client.learning_rate, client.weight_decay = 3, 1e-2, 1e-4
    client.focal_gamma, client.positive_class_idx, client.proximal_mu = 2.0, 1, mu
    client.class_weights = weights()
    return client


def test_p1_penalty_value_by_hand():
    model = nn.Linear(2, 1, bias=True)
    reference = snapshot_global_reference(model)
    with torch.no_grad():
        model.weight += torch.tensor([[3.0, 4.0]])
        model.bias += 12.0
    # ||w - w_t||^2 = 9 + 16 + 144 = 169; (0.01 / 2) * 169 = 0.845. Squared, not the plain norm 13.
    assert proximal_penalty(model, reference, 0.01).item() == pytest.approx(0.845, rel=1e-5)
    assert parameter_distance(model, reference) == pytest.approx(13.0, rel=1e-5)


def test_p2_penalty_is_zero_at_the_reference_and_gradient_is_mu_times_difference():
    model = Tiny()
    reference = snapshot_global_reference(model)
    assert proximal_penalty(model, reference, 0.5).item() == 0.0
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p))
    mu = 0.37
    proximal_penalty(model, reference, mu).backward()
    for p, r in zip(trainable_parameters(model), reference):
        assert torch.allclose(p.grad, mu * (p.detach() - r), atol=1e-6)
    assert all(not r.requires_grad for r in reference)


def test_p3_reference_is_a_copy_not_an_alias():
    model = Tiny()
    reference = snapshot_global_reference(model)
    before = [r.clone() for r in reference]
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    assert all(torch.equal(a, b) for a, b in zip(reference, before))
    assert all(p.data_ptr() != r.data_ptr() for p, r in zip(model.parameters(), reference))


def test_p4_buffers_excluded_and_every_trainable_parameter_included():
    model = Tiny()
    reference = snapshot_global_reference(model)
    assert len(reference) == len(list(model.parameters())) < len(model.state_dict())
    with torch.no_grad():
        for buffer in model.buffers():
            buffer.add_(100)
    assert proximal_penalty(model, reference, 1.0).item() == 0.0

    from models.model import FedMelanomaModel
    real = FedMelanomaModel()
    assert len(trainable_parameters(real)) == len(list(real.parameters())) == 167  # of 329 state tensors; 162 are buffers
    assert sum(p.numel() for p in trainable_parameters(real)) == 23_515_298


def test_p5_reference_stays_fixed_across_every_step_and_equals_the_received_global():
    """The reference passed to train_one_epoch, read at every step of every epoch, equals w_t."""
    client = tiny_client(0.1)
    global_arrays = [v.detach().numpy().copy() for v in client.model.state_dict().values()]
    keys = list(client.model.state_dict().keys())
    expected = [torch.tensor(a) for k, a in zip(keys, global_arrays) if k in dict(client.model.named_parameters())]

    seen = []
    from client import local_trainer
    original = local_trainer.proximal_penalty

    def spy(model, reference, mu):
        seen.append([r.clone() for r in reference])
        return original(model, reference, mu)

    local_trainer.proximal_penalty = spy
    try:
        client.fit(global_arrays, {"local_epochs": 3})
    finally:
        local_trainer.proximal_penalty = original
    assert len(seen) == 3 * 4  # 3 epochs, 4 batches of 10
    for snapshot in seen:
        assert all(torch.equal(a, b) for a, b in zip(snapshot, expected))


def test_p6_second_round_uses_the_new_global_not_the_old_one():
    client = tiny_client(0.1)
    first = [v.detach().numpy().copy() for v in client.model.state_dict().values()]
    trained, _, _ = client.fit(first, {"local_epochs": 1})
    seen = []
    from client import local_trainer
    original = local_trainer.proximal_penalty

    def spy(model, reference, mu):
        seen.append([r.clone() for r in reference])
        return original(model, reference, mu)

    local_trainer.proximal_penalty = spy
    try:
        client.fit(trained, {"local_epochs": 1})
    finally:
        local_trainer.proximal_penalty = original
    names = list(client.model.state_dict().keys())
    params = dict(client.model.named_parameters())
    want = [torch.tensor(a) for k, a in zip(names, trained) if k in params]
    assert all(torch.equal(a, b) for a, b in zip(seen[0], want))


def load_fedavg_trainer():
    """The FedAvg experiment's local_trainer.py, compiled in memory, so nothing is written to its folder."""
    path = os.path.join(FEDAVG, "client", "local_trainer.py")
    spec = importlib.util.spec_from_loader("fedavg_local_trainer", loader=None)
    module = importlib.util.module_from_spec(spec)
    with open(path, encoding="utf-8") as fh:
        exec(compile(fh.read(), path, "exec"), module.__dict__)
    return module


@pytest.mark.skipif(not os.path.isdir(FEDAVG), reason="FedAvg folder not present")
def test_p7_mu_zero_is_bit_identical_to_the_fedavg_experiments_trainer():
    fedavg = load_fedavg_trainer()

    def run(train, expect_tuple):
        torch.manual_seed(5)
        model = Tiny()
        loader = tiny_loader(1)
        optimizer = fedavg.build_optimizer(model, 1e-2, 1e-4)
        criterion = ClassWeightedFocalLoss(gamma_f=2.0, reduction="mean")
        out = None
        for _ in range(3):
            out = train(model, loader, criterion, optimizer, weights(), 0.0, CPU)
        loss = out[0] if expect_tuple else out
        return loss, [v.clone() for v in model.state_dict().values()]

    loss_a, state_a = run(fedavg.train_one_epoch, False)
    loss_b, state_b = run(train_one_epoch, True)
    assert loss_a == loss_b
    assert all(torch.equal(a, b) for a, b in zip(state_a, state_b))


def test_p8_gradient_is_task_gradient_plus_proximal_gradient():
    """The optimizer sees the gradient of the combined objective, not of the task loss alone."""
    torch.manual_seed(3)
    model = Tiny().train()
    reference = snapshot_global_reference(model)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1 * torch.randn_like(p))
    meta, labels = torch.randn(16, 13), (torch.arange(16) % 4 == 0).long()
    criterion = ClassWeightedFocalLoss(gamma_f=2.0, reduction="mean")
    state = copy.deepcopy(model.state_dict())

    torch.manual_seed(9)
    model.zero_grad()
    criterion(model(None, meta), labels, class_weights=weights()).backward()
    task = [p.grad.clone() for p in model.parameters()]

    model.load_state_dict(state)
    torch.manual_seed(9)
    model.zero_grad()
    mu = 0.2
    (criterion(model(None, meta), labels, class_weights=weights()) + proximal_penalty(model, reference, mu)).backward()
    for g, t, p, r in zip((p.grad for p in model.parameters()), task, model.parameters(), reference):
        assert torch.allclose(g, t + mu * (p.detach() - r), atol=1e-6)


def test_p9_larger_mu_keeps_the_client_closer_to_the_global_model():
    drift = {}
    for mu in (1e-3, 1.0, 100.0):
        client = tiny_client(mu)
        arrays = [v.detach().numpy().copy() for v in client.model.state_dict().values()]
        _, _, metrics = client.fit(arrays, {"local_epochs": 3})
        drift[mu] = metrics["local_drift_l2"]
        assert metrics["proximal_mu"] == mu and metrics["proximal_penalty"] > 0
    assert drift[1e-3] > drift[1.0] > drift[100.0]


def test_p10_mu_reaches_clients_from_the_config_and_is_required(cfg):
    import inspect

    from runtime import in_process
    assert cfg["fedprox"]["mu"] == 0.01
    assert 'proximal_mu=cfg["fedprox"]["mu"]' in inspect.getsource(in_process.build_federation)


def test_p11_fit_metrics_keep_train_loss_as_the_task_loss():
    client = tiny_client(0.5)
    arrays = [v.detach().numpy().copy() for v in client.model.state_dict().values()]
    _, _, with_prox = client.fit(arrays, {"local_epochs": 1})
    assert with_prox["train_loss"] > 0 and with_prox["local_drift_l2"] > 0
    client = tiny_client(0.0)
    arrays = [v.detach().numpy().copy() for v in client.model.state_dict().values()]
    _, _, without = client.fit(arrays, {"local_epochs": 1})
    assert without["proximal_penalty"] == 0.0 and without["local_drift_l2"] == 0.0


def test_p12_mu_above_zero_through_flower_message_path(cfg):
    """Same as test 19, with the proximal term on: both routes return the same bytes."""
    from flwr.client.message_handler.message_handler import handle_legacy_message_from_msgtype
    from flwr.common import Context, Message, Metadata, RecordSet
    from flwr.common import recordset_compat as compat
    from flwr.common.constant import MessageType

    numpy_client = tiny_client(0.01)
    client = numpy_client.to_client()
    arrays = [v.detach().numpy().copy() for v in numpy_client.model.state_dict().values()]
    ins = FitIns(ndarrays_to_parameters(arrays), {"local_epochs": 1})
    torch.manual_seed(7)
    direct = client.fit(ins)

    metadata = Metadata(run_id=0, message_id="", src_node_id=1, dst_node_id=2, reply_to_message="", group_id="1",
                        ttl=3600.0, message_type=MessageType.TRAIN)
    context = Context(run_id=0, node_id=2, node_config={}, state=RecordSet(), run_config={})
    torch.manual_seed(7)
    message = Message(metadata=metadata, content=compat.fitins_to_recordset(ins, keep_input=True))
    reply = handle_legacy_message_from_msgtype(lambda ctx: client, message, context)
    via_message = compat.recordset_to_fitres(reply.content, keep_input=False)

    a, b = parameters_to_ndarrays(direct.parameters), parameters_to_ndarrays(via_message.parameters)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert direct.metrics == via_message.metrics
