from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class MammographyDensityDataset(Dataset):
    """
    PyTorch Dataset for breast density classification.

    Expected CSV columns:
    - image_path
    - density
    - density_binary
    """

    def __init__(self, csv_path, transform=None, target_column="density_binary"):
        self.csv_path = Path(csv_path)
        self.dataframe = pd.read_csv(self.csv_path)
        self.transform = transform
        self.target_column = target_column

        required_columns = ["image_path", "density", target_column]

        for column in required_columns:
            if column not in self.dataframe.columns:
                raise ValueError(f"Missing required column: {column}")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_path = Path(row["image_path"])
        image = Image.open(image_path).convert("RGB")

        label = int(row[self.target_column])

        if self.transform is not None:
            image = self.transform(image)

        return image, label