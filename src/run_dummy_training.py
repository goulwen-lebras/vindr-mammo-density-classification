from pathlib import Path
import sys

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


def main():
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
        shuffle=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = build_resnet18(num_classes=2, pretrained=False)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 3

    for epoch in range(epochs):
        average_loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {average_loss:.4f}")

    metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
    )

    print("Evaluation metrics:")
    print("Accuracy:", metrics["accuracy"])
    print("Macro-F1:", metrics["macro_f1"])
    print("Balanced accuracy:", metrics["balanced_accuracy"])
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])


if __name__ == "__main__":
    main()