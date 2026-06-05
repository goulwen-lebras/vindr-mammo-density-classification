import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SRC_DIR))

from dataset import MammographyDensityDataset


def test_dataloader_returns_batch():
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

    images, labels = next(iter(dataloader))

    assert isinstance(images, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert images.shape == (4, 3, 224, 224)
    assert labels.shape == (4,)
    assert labels.dtype == torch.int64