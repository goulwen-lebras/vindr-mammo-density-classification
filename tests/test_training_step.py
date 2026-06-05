import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SRC_DIR))

from dataset import MammographyDensityDataset
from model import build_resnet18


def test_one_training_step_runs():
    csv_path = PROJECT_ROOT / "data" / "dummy_metadata.csv"

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    dataset = MammographyDensityDataset(
        csv_path=csv_path,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
    )

    model = build_resnet18(num_classes=2, pretrained=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    images, labels = next(iter(dataloader))

    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert outputs.shape == (4, 2)
    assert loss.item() > 0