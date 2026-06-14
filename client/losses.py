import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(labels):

    labels = torch.tensor(labels)

    n0 = (labels==0).sum().item()
    n1 = (labels==1).sum().item()

    total = n0+n1

    alpha0 = total/(2*n0)
    alpha1 = total/(2*n1)

    weights = torch.tensor(
        [alpha0,alpha1],
        dtype=torch.float32
    )

    weights = (
        weights /
        weights.sum()
    )

    return weights


class ClassWeightedFocalLoss(
    nn.Module
):

    def __init__(
        self,
        alpha,
        gamma=2
    ):

        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        logits,
        targets
    ):

        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction='none'
        )

        pt = torch.exp(
            -ce_loss
        )

        focal_loss = (
            self.alpha[targets]
            *
            ((1-pt)**self.gamma)
            *
            ce_loss
        )

        return focal_loss.mean()