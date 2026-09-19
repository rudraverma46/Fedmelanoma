import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


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

        backbone = mobilenet_v3_large(
            weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2
        )

        self.backbone_features = backbone.features
        self.backbone_pool = backbone.avgpool
        backbone_out_dim = backbone.classifier[0].in_features  # 960

        self.metadata_mlp = MetadataMLP()

        self.dropout = nn.Dropout(0.3)

        self.classifier = nn.Linear(
            backbone_out_dim + 32,
            2
        )

    def backbone(self, image):

        features = self.backbone_features(image)
        pooled = self.backbone_pool(features)

        return torch.flatten(pooled, 1)

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