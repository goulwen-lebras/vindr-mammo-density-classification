import sys
from pathlib import Path

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