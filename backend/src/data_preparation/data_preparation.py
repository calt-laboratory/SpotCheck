import polars as pl
from pathlib import Path
from typing import Final
from torch.utils.data import Dataset


DATA_DIR: Final[Path] = Path(__file__).parent.parent.parent / "data" / "ham10000"
METADATA_DIR: Final[Path] = DATA_DIR / "HAM10000_metadata.csv"
IMAGE_DIR: Final[Path] = DATA_DIR / "images"

def load_metadata() -> pl.DataFrame:
    df = pl.read_csv(METADATA_DIR)
    print(df)

    # Keep only rows where diagnosis col is "nv" = Nevus or "mel" = Melanoma
    return df.filter(pl.col("dx").is_in(["nv", "mel"]))


if __name__ == "__main__":
    load_metadata()
