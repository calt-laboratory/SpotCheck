import polars as pl
from pathlib import Path
from typing import Final
from PIL import Image
from PIL.ImageFile import ImageFile

from functorch.dim import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


DATA_DIR: Final[Path] = Path(__file__).parent.parent.parent / "data" / "ham10000"
METADATA_PATH: Final[Path] = DATA_DIR / "HAM10000_metadata.csv"
IMAGE_DIR: Final[Path] = DATA_DIR / "images"
LABEL_MAP: Final[dict[str, int]] = {"nv": 0, "mel": 1}


def load_metadata() -> pl.DataFrame:
    df = pl.read_csv(METADATA_PATH)
    print(df)

    # Keep only rows where diagnosis col is "nv" = Nevus or "mel" = Melanoma
    return df.filter(pl.col("dx").is_in(["nv", "mel"]))


class NevusDataset(Dataset):
    def __init__(self, df: pl.DataFrame, transform: transforms | None = None):
        self._df = df
        self._images_dir = IMAGE_DIR
        self._transform = transform

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> tuple[ImageFile, int]:
        row = self._df[idx]
        img_path = self._images_dir / f"{row['image_id']}.jpg"
        img = Image.open(img_path).convert("RGB")
        label = LABEL_MAP[row["dx"].item()]

        if self._transform:
            img = self._transform(img)

        return img, label


if __name__ == "__main__":
    load_metadata()
