import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SRC_DIR))

from model import build_resnet18


def test_resnet18_binary_output_shape():
    model = build_resnet18(num_classes=2, pretrained=False)

    dummy_batch = torch.randn(4, 3, 224, 224)

    outputs = model(dummy_batch)

    assert outputs.shape == (4, 2)


def test_resnet18_four_class_output_shape():
    model = build_resnet18(num_classes=4, pretrained=False)

    dummy_batch = torch.randn(4, 3, 224, 224)

    outputs = model(dummy_batch)

    assert outputs.shape == (4, 4)