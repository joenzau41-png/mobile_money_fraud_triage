"""Data loading utilities for the PaySim fraud triage project."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
RAW_CSV = RAW_DIR / "PS_20174392719_1491204439457_log.csv"
FULL_PARQUET = PROC_DIR / "paysim.parquet"
SAMPLE_PARQUET = PROC_DIR / "paysim_sample.parquet"

KAGGLE_SLUG = "ealaxi/paysim1"
SAMPLE_FRAC = 0.10
SEED = 42


def download_raw() -> None:
    """Download PaySim via the Kaggle API (needs ~/.kaggle/kaggle.json)."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_SLUG, path=RAW_DIR, unzip=True)
    print(f"Downloaded to {RAW_DIR}")


def to_parquet() -> None:
    """Convert raw CSV to Parquet and write a 10% development sample."""
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"{RAW_CSV} not found. See data/README.md for download steps."
        )
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_CSV)
    df.to_parquet(FULL_PARQUET, index=False)

    # Random 10% sample, kept in time order, for fast EDA.
    # (Time-based train/val/test splitting comes in Step 4.)
    sample = df.sample(frac=SAMPLE_FRAC, random_state=SEED).sort_values("step")
    sample.to_parquet(SAMPLE_PARQUET, index=False)
    print(f"Full: {len(df):,} rows -> {FULL_PARQUET.name}")
    print(f"Sample: {len(sample):,} rows -> {SAMPLE_PARQUET.name}")


def load(sample: bool = True) -> pd.DataFrame:
    """Load the processed dataset (10% sample by default)."""
    path = SAMPLE_PARQUET if sample else FULL_PARQUET
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run: python -m src.data convert")
    return pd.read_parquet(path)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "download":
        download_raw()
    elif cmd == "convert":
        to_parquet()
    else:
        print("Usage: python -m src.data [download|convert]")