"""
Metadata handling utilities.
"""

import pandas as pd
from pathlib import Path

from .config import DEFAULT_METADATA_ROWS


def load_or_create_metadata(path: Path) -> pd.DataFrame:
    """Load metadata from CSV or create default."""
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(DEFAULT_METADATA_ROWS)


def save_metadata(df: pd.DataFrame, path: Path) -> None:
    """Save metadata to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

