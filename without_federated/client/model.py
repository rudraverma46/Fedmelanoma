import torch
import torch.nn as nn
import timm


class MetadataMLP(nn.Module):

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


class FedMelanomaModel(nn.Module):

    def __init__(self):

        super().__init__()

        # --- CHANGED: EfficientNet-B3 -> ResNet50 ---------------------------
        # ResNet50 (ImageNet pretrained), classifier head stripped via
        # num_classes=0 so timm applies global average pooling and returns
        # the 2048-dim penultimate feature vector directly.
        self.backbone = timm.create_model(
            "resnet50",
            pretrained=True,
            num_classes=0
        )

        self.metadata_mlp = MetadataMLP()

        self.dropout = nn.Dropout(0.3)

        # --- CHANGED: fusion input dim 1536 (EffNet-B3) -> 2048 (ResNet50) --
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
