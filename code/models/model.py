"""ResNet-50 variant of the reference models/model.py (commit 1351f63).

This is the only reference file whose content changes. Against the reference:
  - imports: `timm` -> `torchvision.models`
  - self.backbone: efficientnet_b3 (num_classes=0) -> resnet50 with fc = Identity
  - classifier input: 1536 + 32 -> 2048 + 32
  - MetadataMLP input: 13 -> 9 (diagnosis_confirm_type removed; it leaks the label)
The dropout, forward() and the class names are unchanged, so every vendored module that imports
FedMelanomaModel works without edits.
"""
import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

# Must equal client.dataset.MelanomaDataset.METADATA_DIM (6 skin types + 3 image types).
METADATA_DIM = 9


class MetadataMLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(METADATA_DIM, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU()
        )

    def forward(self, x):

        return self.network(x)


class FedMelanomaModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V2
        )

        # Same effect as timm's num_classes=0: the backbone returns the
        # globally pooled feature vector instead of ImageNet logits.
        self.backbone.fc = nn.Identity()

        self.metadata_mlp = MetadataMLP()

        self.dropout = nn.Dropout(float(__import__("os").environ.get("FEDMEL_DROPOUT", "0.3")))

        self.classifier = nn.Linear(
            2048 + 32,
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
