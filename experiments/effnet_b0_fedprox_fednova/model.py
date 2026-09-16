"""
model.py — EfficientNet-B0 + Metadata MLP fusion model.

This mirrors the fusion architecture used across the rest of the FedMelanoma
repo (client/model.py, without_federated/client/model.py), but swaps the
image backbone from EfficientNet-B3 (1536-d features) to EfficientNet-B0
(1280-d features).

Why EfficientNet-B0 specifically:
    Selected to reproduce the backbone reported as the strongest published
    result for federated melanoma/skin-cancer classification found during
    a literature review (Springer ICACECS 2025, "Federated Learning for
    Skin Cancer Detection with Optimized Client Selection and Hybrid
    Aggregation Strategies"), which pairs EfficientNet-B0 with a hybrid
    FedProx + FedNova aggregation strategy to reach ~93.56% accuracy.
    See ../../docs/FedMelanoma_Blueprint_v4.md Section 3.3 for the B3
    variant used by this repo's canonical v4 architecture — this experiment
    is a deliberate, isolated alternative for comparison, not a replacement.

Verified empirically (see experiments/effnet_b0_fedprox_fednova/README.md):
    timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
    produces a 1280-dim pooled feature vector, confirmed via a live forward
    pass (not just the model's `num_features` attribute, which can be
    misleading for architectures with a separate expansion head — this bit
    us when swapping in MobileNetV3 earlier in this project).
"""

import torch
import torch.nn as nn
import timm


class MetadataMLP(nn.Module):
    """
    Two-layer projection of the 13-dim one-hot metadata vector
    (6 Fitzpatrick skin types + 3 imaging modalities + 4 confirmation types)
    into a 32-dim embedding. Identical to the MetadataMLP used elsewhere in
    this repo (client/model.py, without_federated/client/model.py).
    """

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(13, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU()
        )

    def forward(self, x):
        return self.network(x)


class FedMelanomaModelB0(nn.Module):
    """
    Image + metadata fusion model for melanoma binary classification,
    using an EfficientNet-B0 backbone.

    Fusion width: 1280 (EfficientNet-B0 image features) + 32 (metadata) = 1312.
    """

    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0
        )

        self.metadata_mlp = MetadataMLP()

        self.dropout = nn.Dropout(0.3)

        self.classifier = nn.Linear(
            1280 + 32,
            2
        )

    def forward(
        self,
        image,
        metadata
    ):
        image_features = self.backbone(
            image
        )

        metadata_features = self.metadata_mlp(
            metadata
        )

        fused = torch.cat(
            [
                image_features,
                metadata_features
            ],
            dim=1
        )

        fused = self.dropout(fused)

        logits = self.classifier(
            fused
        )

        return logits
