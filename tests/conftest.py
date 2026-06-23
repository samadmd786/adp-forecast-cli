"""Shared pytest fixtures.

The committed ``mini_history.csv`` is a tiny hand made long-format CSV (not real
ADP data). When a test needs a zip, it is built on the fly into a tmp dir.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"
MINI_CSV = FIXTURE_DIR / "mini_history.csv"


@pytest.fixture
def mini_csv_path() -> Path:
    """Path to the committed hand made fixture CSV."""
    return MINI_CSV


@pytest.fixture
def mini_zip(tmp_path: Path) -> Path:
    """Zip the fixture CSV into a tmp dir under the expected member name."""
    zip_path = tmp_path / "ADP_NER_history.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(MINI_CSV, arcname="ADP_NER_history.csv")
    return zip_path
