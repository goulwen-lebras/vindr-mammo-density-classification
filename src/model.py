import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes=2, pretrained=True):
    """
    Build a ResNet18 model for breast density classification.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
        For binary density classification: 2.
        For four-class BI-RADS density classification: 4.

    pretrained : bool
        If True, use weights pretrained on ImageNet.

    Returns
    -------
    model : torch.nn.Module
        ResNet18 model with a modified final classification layer.
    """

    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model