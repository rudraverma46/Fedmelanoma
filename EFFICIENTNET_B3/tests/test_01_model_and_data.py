"""Tests 1-10: model, loss, backpropagation, GPU, serialisation, client construction."""
import copy
import os

import numpy as np
import pytest
import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
import timm

from client.client import FedMelanomaClient
from client.dataset import MelanomaDataset
from client.focal_loss import ClassWeightedFocalLoss, compute_class_weights
from models.model import FedMelanomaModel, MetadataMLP
from monitoring.param_hash import hash_ndarrays
from runtime.image_cache import build_cached_dataset, split_paths


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return FedMelanomaModel()


def batch(n=4):
    g = torch.Generator().manual_seed(1)
    image = torch.randn(n, 3, 224, 224, generator=g)
    meta = torch.zeros(n, 13)
    rows = torch.arange(n)
    meta[rows, rows % 6] = 1              # varied rows: identical rows have zero BatchNorm variance
    meta[rows, 6 + rows % 3] = 1
    meta[rows, 9 + rows % 4] = 1
    return image, meta


def test_01_model_initialisation(model):
    assert len(model.state_dict()) == 583
    assert sum(p.numel() for p in model.parameters()) == 10_702_474
    assert all(p.requires_grad for p in model.parameters())
    assert model.classifier.in_features == 1568 and model.classifier.out_features == 2
    assert model.dropout.p == 0.3


def test_02_efficientnet_forward_matches_timm(model):
    model.eval()
    image, _ = batch()
    assert timm.__version__ == "1.0.19"
    reference = timm.create_model("efficientnet_b3", pretrained=True, num_classes=0)
    reference.eval()
    with torch.no_grad():
        features, expected = model.backbone(image), reference(image)
    assert features.shape == (4, 1536)
    assert torch.isfinite(features).all()
    assert torch.equal(features, expected)


def test_03_metadata_mlp(model):
    model.eval()
    _, meta = batch()
    with torch.no_grad():
        out = model.metadata_mlp(meta)
    assert out.shape == (4, 32) and (out >= 0).all()
    layers = [type(m).__name__ for m in MetadataMLP().network]
    assert layers == ["Linear", "BatchNorm1d", "ReLU", "Linear", "ReLU"]
    assert model.metadata_mlp.network[0].in_features == 13
    assert model.metadata_mlp.network[0].out_features == 64
    assert model.metadata_mlp.network[3].out_features == 32


def test_04_feature_concatenation_order(model):
    """Recompute logits by hand; only [image ; metadata] in this order gives the same result."""
    model.eval()
    image, meta = batch()
    with torch.no_grad():
        fused = torch.cat([model.backbone(image), model.metadata_mlp(meta)], dim=1)
        assert fused.shape == (4, 1568)
        assert torch.allclose(model.classifier(fused), model(image, meta), atol=1e-5)
        swapped = torch.cat([model.metadata_mlp(meta), model.backbone(image)], dim=1)
        assert not torch.allclose(model.classifier(swapped), model(image, meta), atol=1e-3)


def test_05_classification_output(model):
    model.eval()
    with torch.no_grad():
        logits = model(*batch())
    assert logits.shape == (4, 2)
    assert torch.allclose(torch.softmax(logits, 1).sum(1), torch.ones(4))


def test_06_loss_matches_independent_calculation():
    logits = np.array([[2.0, -1.0], [0.3, 0.1], [-1.5, 2.5], [0.0, 0.0], [1.0, 3.0]])
    targets = np.array([0, 1, 1, 0, 0], dtype=np.int64)
    n0, n1 = 3, 2
    a0, a1 = 5 / (2 * n0), 5 / (2 * n1)
    alpha = np.array([a0 / (a0 + a1), a1 / (a0 + a1)])
    exp = np.exp(logits)
    pt = (exp / exp.sum(1, keepdims=True))[np.arange(5), targets]
    expected = np.mean(-alpha[targets] * (1 - pt) ** 2 * np.log(pt))

    t = torch.tensor(targets)
    weights = compute_class_weights(t)
    assert np.allclose(weights.numpy(), alpha, atol=1e-6)
    got = ClassWeightedFocalLoss(gamma_f=2.0, reduction="mean")(torch.tensor(logits, dtype=torch.float32), t, class_weights=weights)
    assert abs(got.item() - expected) < 1e-6


def test_07_backpropagation_reaches_all_branches():
    torch.manual_seed(0)
    model = FedMelanomaModel().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    image, meta = batch()
    targets = torch.tensor([0, 1, 0, 1])
    before = model.classifier.weight.detach().clone()
    conv_before = model.backbone.conv_stem.weight.detach().clone()
    mlp_before = model.metadata_mlp.network[0].weight.detach().clone()
    loss = ClassWeightedFocalLoss()(model(image, meta), targets, class_weights=compute_class_weights(targets))
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    assert model.backbone.conv_stem.weight.grad.abs().sum() > 0
    assert model.metadata_mlp.network[0].weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, model.classifier.weight)
    assert not torch.equal(conv_before, model.backbone.conv_stem.weight)
    assert not torch.equal(mlp_before, model.metadata_mlp.network[0].weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_08_gpu_training_step_at_real_batch_size():
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    model = FedMelanomaModel().to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    image, meta = batch(20)
    targets = torch.randint(0, 2, (20,), device=device)
    for _ in range(2):
        optimizer.zero_grad()
        loss = ClassWeightedFocalLoss()(model(image.to(device), meta.to(device)), targets, class_weights=torch.tensor([0.05, 0.95], device=device))
        loss.backward()
        optimizer.step()
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    print(f"peak GPU memory at batch 20: {peak_gb:.2f} GB")
    assert torch.isfinite(loss) and peak_gb < 3.9
    del model, optimizer
    torch.cuda.empty_cache()


def test_09_serialisation_round_trip():
    model = FedMelanomaModel()
    arrays = [v.detach().cpu().numpy().copy() for v in model.state_dict().values()]
    restored = parameters_to_ndarrays(ndarrays_to_parameters(arrays))
    assert hash_ndarrays(arrays) == hash_ndarrays(restored)
    assert all(a.dtype == b.dtype and np.array_equal(a, b) for a, b in zip(arrays, restored))

    # After aggregation num_batches_tracked is float64, as in the reference; it must still load.
    keys = list(model.state_dict().keys())
    tracked = [i for i, k in enumerate(keys) if k.endswith("num_batches_tracked")]
    assert len(tracked) == 79
    aggregated = [a.copy() for a in arrays]
    for i in tracked:
        aggregated[i] = np.asarray(12.7, dtype=np.float64)
    state = {k: torch.tensor(v) for k, v in zip(keys, aggregated)}
    other = copy.deepcopy(model)
    other.load_state_dict(state, strict=True)
    assert other.state_dict()[keys[tracked[0]]].dtype == torch.int64
    assert int(other.state_dict()[keys[tracked[0]]]) == 12


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_10_clients_build_and_cache_is_pixel_identical(cfg, hospitals):
    from runtime.in_process import build_federation

    assert len(hospitals) == 10
    root, cache = cfg["data"]["partitions_root"], cfg["data"]["cache_dir"]
    for hospital in hospitals:
        for split in ("train", "test"):
            cached = build_cached_dataset(root, cache, hospital, split)
            plain = MelanomaDataset(*split_paths(root, hospital, split))
            assert len(cached) == len(plain)
            assert torch.equal(cached.labels, plain.labels) and torch.equal(cached.metadata_cache, plain.metadata_cache)
            for idx in (0, len(plain) // 2, len(plain) - 1):
                a, b = cached[idx], plain[idx]
                assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])

    model = FedMelanomaModel()
    manager = build_federation(cfg, hospitals, model, torch.device("cuda"), lambda record: None)
    proxies = manager.all()
    assert sorted(proxies) == [f"{i:02d}" for i in range(10)]
    total = 0
    for i, hospital in enumerate(hospitals):
        proxy = proxies[f"{i:02d}"]
        assert proxy.hospital == hospital
        numpy_client = proxy.client.numpy_client
        assert isinstance(numpy_client, FedMelanomaClient)
        assert numpy_client.model is model
        dataset = numpy_client.train_loader.dataset
        assert numpy_client.train_loader.batch_size == 20
        total += len(dataset)
        labels = dataset.labels
        n1, n0 = int(labels.sum()), int((labels == 0).sum())
        a0, a1 = len(labels) / (2 * n0), len(labels) / (2 * n1)
        expected = torch.tensor([a0 / (a0 + a1), a1 / (a0 + a1)])
        assert torch.allclose(numpy_client.class_weights.cpu(), expected, atol=1e-6)
        assert len(labels) % 20 != 1, "a final batch of one row would crash BatchNorm1d"
    assert total == 7788
    del manager, model
    torch.cuda.empty_cache()


def test_config_paths_exist(cfg):
    assert os.path.isdir(cfg["data"]["partitions_root"])
