import sys
from pathlib import Path
from torchvision import transforms
import torch

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SRC_DIR))

from dataset import MammographyDensityDataset


def test_dataset_loads_dummy_metadata():
    csv_path = PROJECT_ROOT / "data" / "dummy_metadata.csv"

    dataset = MammographyDensityDataset(csv_path=csv_path)

    assert len(dataset) == 20


def test_dataset_returns_image_and_label():
    csv_path = PROJECT_ROOT / "data" / "dummy_metadata.csv"

    dataset = MammographyDensityDataset(csv_path=csv_path)

    image, label = dataset[0]

    assert isinstance(image, Image.Image)
    assert image.size == (224, 224)
    assert label in [0, 1]

def test_dataset_returns_tensor_when_transform_is_used():
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

    image, label = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)
    assert image.min() >= 0.0
    assert image.max() <= 1.0
    assert label in [0, 1]