"""Download, extract, and parse the ADP National Employment Report archive.

The raw ADP data is copyrighted and must not be committed to the repo, so the
download lands in a gitignored ``data/`` directory and is fetched exactly once.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DEFAULT_URL = (
    "https://adpemploymentreport.com/artifacts/us_ner/20260603/ADP_NER_history.zip"
)

CSV_MEMBER = "ADP_NER_history.csv"
ZIP_NAME = "ADP_NER_history.zip"

REQUIRED_COLUMNS = ["timestep", "agg_RIS", "category", "date", "NER", "NER_SA"]


class DataError(Exception):
    """Raised when downloading, extracting, or parsing the data fails."""


def fetch(
    url: str = DEFAULT_URL, data_dir: str = "data", local_path: str | None = None
) -> Path:
    """Return a path to the ADP archive, downloading it once if needed.

    If ``local_path`` is given it is used directly and the network is never
    touched. Otherwise the zip is cached in ``data_dir`` and downloaded only on
    the first run when it is not already present.
    """
    if local_path is not None:
        path = Path(local_path)
        if not path.exists():
            raise DataError(f"Local input file not found: {path}")
        return path

    dest_dir = Path(data_dir)
    zip_path = dest_dir / ZIP_NAME
    if zip_path.exists():
        return zip_path

    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = zip_path.with_suffix(zip_path.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp_path)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise DataError(
            f"Failed to download the ADP data from {url}: {exc}. "
            "Check your network connection or pass --input with a local zip or csv."
        ) from exc

    os.replace(tmp_path, zip_path)
    return zip_path


def extract(zip_path: str | os.PathLike) -> Path:
    """Extract the history CSV from the archive and return its path.

    The CSV is written next to the zip. Raises a clear error if the file is not
    a valid zip or does not contain the expected CSV member.
    """
    zip_path = Path(zip_path)
    if not zipfile.is_zipfile(zip_path):
        raise DataError(
            f"{zip_path} is not a valid zip archive. "
            "Pass --input with a real ADP zip or the extracted csv."
        )

    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        if CSV_MEMBER not in names:
            raise DataError(
                f"Archive {zip_path} does not contain the expected member "
                f"{CSV_MEMBER}. Found: {names}."
            )
        out_dir = zip_path.parent
        archive.extract(CSV_MEMBER, out_dir)
    return out_dir / CSV_MEMBER


def load_raw(csv_path: str | os.PathLike) -> pd.DataFrame:
    """Read the long-format history CSV into a validated DataFrame.

    Parses ``date`` as datetime and coerces ``NER`` and ``NER_SA`` to numeric.
    Raises a clear error naming any missing or non-numeric column.
    """
    csv_path = Path(csv_path)
    try:
        df = pd.read_csv(csv_path)
    except (OSError, pd.errors.ParserError) as exc:
        raise DataError(f"Could not read CSV {csv_path}: {exc}") from exc

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(
            f"CSV {csv_path} is missing required column(s): {', '.join(missing)}."
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise DataError(f"Column 'date' in {csv_path} has unparseable values.")

    for col in ("NER", "NER_SA"):
        coerced = pd.to_numeric(df[col], errors="coerce")
        # A value that became NaN but was not originally blank was non-numeric.
        newly_bad = coerced.isna() & df[col].notna()
        if newly_bad.any():
            raise DataError(f"Column '{col}' in {csv_path} has non-numeric values.")
        # Store as float so downstream diffing and forecasting math is uniform.
        df[col] = coerced.astype(float)

    return df


def load(
    url: str = DEFAULT_URL,
    data_dir: str = "data",
    local_path: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Convenience pipeline: fetch, extract if needed, and parse to a DataFrame."""
    if refresh and local_path is None:
        zip_path = Path(data_dir) / ZIP_NAME
        if zip_path.exists():
            zip_path.unlink()

    source = fetch(url=url, data_dir=data_dir, local_path=local_path)
    source = Path(source)
    if source.suffix.lower() == ".csv":
        csv_path = source
    else:
        csv_path = extract(source)
    return load_raw(csv_path)
