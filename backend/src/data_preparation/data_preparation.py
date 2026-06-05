import polars as pl
from pathlib import Path
from typing import Final, Any
from PIL import Image
from PIL.ImageFile import ImageFile

from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torchvision.transforms import Compose

DATA_DIR: Final[Path] = Path(__file__).parent.parent.parent / "data" / "ham10000"
METADATA_PATH: Final[Path] = DATA_DIR / "HAM10000_metadata.csv"
IMAGE_DIR: Final[Path] = DATA_DIR / "images"
LABEL_MAP: Final[dict[str, int]] = {"nv": 0, "mel": 1}

# Standard normalization values for ImageNet-pretrained models.
# These are used because most PyTorch models (ResNet, EfficientNet, etc.) were pretrained on ImageNet, and normalizing
# w/ these mean/std values ensures compatibility w/ the pretrained weights and stable training.
IMAGENET_MEAN: Final[list[float]] = [
    0.485,
    0.456,
    0.406,
]  # RGB channel means from ImageNet
IMAGENET_STD: Final[list[float]] = [
    0.229,
    0.224,
    0.225,
]  # RGB channel std from ImageNet

# Default input size for most pretrained models (ResNet, EfficientNet, etc.)
# 224x224 is the standard resolution used in ImageNet pretraining, balancing accuracy and computational efficiency.
DEFAULT_IMAGE_SIZE: Final[tuple[int, int]] = (224, 224)


def split_datasets(
    test_size: float = 0.2,
    validation_size: float = 0.1,
) -> tuple[Dataset, Dataset, Dataset]:
    df = _load_metadata()
    print(f"Number of samples: {df.shape[0]}")

    transform = _get_transforms()
    dataset = NevusDataset(df, transform)

    test_dataset, temp_dataset = random_split(
        dataset=dataset, lengths=[test_size, 1 - test_size]
    )

    validation_dataset, train_dataset = random_split(
        dataset=temp_dataset,
        lengths=[
            validation_size / (1 - test_size),
            1 - validation_size / (1 - test_size),
        ],
    )

    return train_dataset, validation_dataset, test_dataset



def _load_metadata() -> pl.DataFrame:
    df = pl.read_csv(METADATA_PATH)
    print(df.head())

    # Keep only rows where diagnosis col is "nv" = Nevus or "mel" = Melanoma
    return df.filter(pl.col("dx").is_in(["nv", "mel"]))


class NevusDataset(Dataset):
    def __init__(self, df: pl.DataFrame, transform: Compose | None = None):
        self._df = df
        self._images_dir = IMAGE_DIR
        self._transform = transform

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> tuple[ImageFile | Any, int]:
        row = self._df[idx]
        img_path = self._images_dir / f"{row['image_id'].item()}.jpg"
        img = Image.open(img_path).convert("RGB")  # removes Alpha channel
        label = LABEL_MAP[row["dx"].item()]

        if self._transform:
            img = self._transform(img)

        return img, label


def _get_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(DEFAULT_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


if __name__ == "__main__":
    split_datasets()
