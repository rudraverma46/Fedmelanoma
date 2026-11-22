"""Test 19: the in-process proxy gives the same result as Flower's real message path.

Under `flwr run`, DriverClientProxy turns FitIns into a RecordSet, wraps it in a Message, and the
ClientApp handles it with handle_legacy_message_from_msgtype; the reply is converted back. That
is everything between strategy and client except the Ray/gRPC delivery of the Message.

Here a real FedMelanomaClient with the real ResNet-50 model trains on a real hospital through
both routes, from the same parameters and the same RNG state, and the returned bytes must match.
It runs on the CPU, where training is deterministic, on the smallest hospital.
"""
import numpy as np
import pytest
import torch
from flwr.client.message_handler.message_handler import handle_legacy_message_from_msgtype
from flwr.common import Context, EvaluateIns, FitIns, Message, Metadata, RecordSet, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common import recordset_compat as compat
from flwr.common.constant import MessageType

from models.model import FedMelanomaModel
from monitoring.param_hash import hash_ndarrays
from runtime.image_cache import build_cached_dataset, build_loader
from runtime.in_process import InProcessClientProxy, InstrumentedClient

HOSPITAL = "TeleDerm_National_Screening_Network"


def send(content: RecordSet, message_type: str, client) -> RecordSet:
    metadata = Metadata(run_id=0, message_id="", src_node_id=1, dst_node_id=2, reply_to_message="",
                        group_id="1", ttl=3600.0, message_type=message_type)
    context = Context(run_id=0, node_id=2, node_config={}, state=RecordSet(), run_config={})
    reply = handle_legacy_message_from_msgtype(lambda ctx: client, Message(metadata=metadata, content=content), context)
    return reply.content


@pytest.fixture(scope="module")
def setup(cfg):
    torch.manual_seed(42)
    model = FedMelanomaModel()
    root, cache = cfg["data"]["partitions_root"], cfg["data"]["cache_dir"]
    train = build_loader(build_cached_dataset(root, cache, HOSPITAL, "train"), 20, True, 0, False)
    test = build_loader(build_cached_dataset(root, cache, HOSPITAL, "test"), 20, False, 0, False)
    numpy_client = InstrumentedClient(model=model, train_loader=train, test_loader=test, device=torch.device("cpu"))
    arrays = [v.detach().cpu().numpy().copy() for v in model.state_dict().values()]
    return numpy_client, ndarrays_to_parameters(arrays)


def test_19_fit_identical_through_both_routes(setup):
    numpy_client, parameters = setup
    client = numpy_client.to_client()
    ins = FitIns(parameters, {"local_epochs": 1})
    proxy = InProcessClientProxy("00", 0, HOSPITAL, client, lambda record: None)

    torch.manual_seed(7)
    proxy_fit = client.fit  # proxy.fit also records CUDA peaks, which this CPU test must not touch
    direct = proxy_fit(ins)
    torch.manual_seed(7)
    via_message = compat.recordset_to_fitres(
        send(compat.fitins_to_recordset(ins, keep_input=True), MessageType.TRAIN, client), keep_input=False)

    a, b = parameters_to_ndarrays(direct.parameters), parameters_to_ndarrays(via_message.parameters)
    assert hash_ndarrays(a) != hash_ndarrays(parameters_to_ndarrays(parameters)), "training changed nothing"
    assert len(a) == len(b) == 329
    assert all(x.dtype == y.dtype and np.array_equal(x, y) for x, y in zip(a, b))
    assert direct.num_examples == via_message.num_examples == 155
    assert direct.metrics == via_message.metrics
    assert proxy.client is client


def test_19_evaluate_identical_through_both_routes(setup):
    numpy_client, parameters = setup
    client = numpy_client.to_client()
    ins = EvaluateIns(parameters, {})
    proxy = InProcessClientProxy("00", 0, HOSPITAL, client, lambda record: None)

    direct = proxy.evaluate(ins, None, 1)
    via_message = compat.recordset_to_evaluateres(send(compat.evaluateins_to_recordset(ins, keep_input=True), MessageType.EVALUATE, client))

    assert direct.loss == via_message.loss
    assert direct.num_examples == via_message.num_examples == 39
    assert direct.metrics == via_message.metrics
