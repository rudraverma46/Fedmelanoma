"""
metadata_loader.py
==================
Loads the metadata CSV, the blueprint file (if available),
and helps locate image files on disk.
"""

from pathlib import Path
import pandas as pd
from config import (
    INPUT_DIR, IMAGE_EXTENSIONS,
    ID_COLUMN_CANDIDATES, FITZPATRICK_COLUMN_CANDIDATES,
    IMAGE_TYPE_COLUMN_CANDIDATES, CONFIRM_COLUMN_CANDIDATES,
)


def find_metadata_csv(input_dir: Path = INPUT_DIR) -> Path | None:
    """Find the first .csv file inside the input directory."""
    csvs = list(input_dir.glob("*.csv"))
    if not csvs:
        return None
    # Prefer one that contains 'metadata' in the name
    for c in csvs:
        if "metadata" in c.name.lower():
            return c
    return csvs[0]


def find_blueprint_file(input_dir: Path = INPUT_DIR) -> Path | None:
    """Find a blueprint / method-description file (txt / md)."""
    for ext in ("*.md", "*.txt"):
        for f in input_dir.glob(ext):
            name = f.name.lower()
            if any(k in name for k in ("blueprint", "method", "description", "plan")):
                return f
    return None


def load_metadata(csv_path: Path) -> pd.DataFrame:
    """Load metadata CSV into a pandas DataFrame."""
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"[metadata_loader] Loaded metadata: {csv_path.name} | rows={len(df)} | cols={len(df.columns)}")
    return df


def detect_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column found in df (case-insensitive)."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def detect_image_id_column(df: pd.DataFrame) -> str | None:
    return detect_column(df, ID_COLUMN_CANDIDATES)


def find_image_file(image_id: str, images_dir: Path = INPUT_DIR) -> Path | None:
    """
    Try to locate an image on disk given its ISIC id / filename stem.
    Tries common extensions.
    """
    if image_id is None or (isinstance(image_id, float) and pd.isna(image_id)):
        return None

    image_id = str(image_id).strip()
    # If extension already present, try direct
    p_direct = images_dir / image_id
    if p_direct.is_file():
        return p_direct

    # Try with extensions
    stem = Path(image_id).stem
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def detect_all_columns(df: pd.DataFrame) -> dict:
    """Detect all needed columns in one call."""
    return {
        "id":          detect_image_id_column(df),
        "fitzpatrick": detect_column(df, FITZPATRICK_COLUMN_CANDIDATES),
        "image_type":  detect_column(df, IMAGE_TYPE_COLUMN_CANDIDATES),
        "confirm":     detect_column(df, CONFIRM_COLUMN_CANDIDATES),
    }