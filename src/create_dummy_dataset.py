from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "dummy_images"
CSV_PATH = DATA_DIR / "dummy_metadata.csv"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

densities = ["A", "B", "C", "D"]
rows = []

for i in range(20):
    density = densities[i % 4]
    image_id = f"dummy_{i:03d}"
    image_path = IMAGE_DIR / f"{image_id}.png"

    image_array = np.random.randint(
        low=0,
        high=255,
        size=(224, 224),
        dtype=np.uint8
    )

    image = Image.fromarray(image_array)
    image.save(image_path)

    binary_label = 0 if density in ["A", "B"] else 1

    rows.append(
        {
            "image_id": image_id,
            "density": density,
            "density_binary": binary_label,
            "image_path": str(image_path),
        }
    )

df = pd.DataFrame(rows)
df.to_csv(CSV_PATH, index=False)

print(f"Created {len(df)} dummy samples")
print(f"CSV saved to: {CSV_PATH}")
print(df.head())