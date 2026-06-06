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
from train import train_one_epoch
from evaluate import evaluate_model


def test_evaluate_model_returns_metrics():
    csv_path = PROJECT_ROOT / "data" / "dummy_metadata.csv"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = MammographyDensityDataset(
        csv_path=csv_path,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
    )

    device = torch.device("cpu")

    model = build_resnet18(num_classes=2, pretrained=False)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_one_epoch(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
    )

    print(metrics)

    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "balanced_accuracy" in metrics
    assert "confusion_matrix" in metrics

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
    assert metrics["confusion_matrix"].shape == (2, 2)