"""Tests for the data layer: fetch, extract, load_raw."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from adp_ner import data
from adp_ner.data import DataError, ZIP_NAME


def test_fetch_downloads_once(tmp_path, mini_zip, monkeypatch):
    """First run with no data dir downloads exactly once and caches the zip."""
    data_dir = tmp_path / "data"
    calls = {"n": 0}

    def fake_retrieve(url, filename):
        calls["n"] += 1
        Path(filename).write_bytes(mini_zip.read_bytes())

    monkeypatch.setattr(data.urllib.request, "urlretrieve", fake_retrieve)

    path = data.fetch(url="http://example.test/x.zip", data_dir=str(data_dir))
    assert path == data_dir / ZIP_NAME
    assert path.exists()
    assert calls["n"] == 1


def test_fetch_uses_cache_second_run(tmp_path, mini_zip, monkeypatch):
    """Second run with the zip present does not touch the network."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / ZIP_NAME).write_bytes(mini_zip.read_bytes())

    def boom(url, filename):
        raise AssertionError("network should not be called when cached")

    monkeypatch.setattr(data.urllib.request, "urlretrieve", boom)
    path = data.fetch(url="http://example.test/x.zip", data_dir=str(data_dir))
    assert path == data_dir / ZIP_NAME


def test_fetch_download_failure_is_clean(tmp_path, monkeypatch):
    """A failed download raises a clear error and leaves no partial zip."""
    data_dir = tmp_path / "data"

    def fail(url, filename):
        Path(filename).write_bytes(b"partial")
        raise OSError("connection reset")

    monkeypatch.setattr(data.urllib.request, "urlretrieve", fail)
    with pytest.raises(DataError) as exc:
        data.fetch(url="http://example.test/x.zip", data_dir=str(data_dir))
    assert "Failed to download" in str(exc.value)
    assert "--input" in str(exc.value)
    assert not (data_dir / ZIP_NAME).exists()
    assert not (data_dir / (ZIP_NAME + ".part")).exists()


def test_fetch_local_path(mini_zip):
    """A local path is returned directly without any network."""
    assert data.fetch(local_path=str(mini_zip)) == mini_zip


def test_fetch_local_path_missing(tmp_path):
    """A missing local path raises a clear error."""
    with pytest.raises(DataError):
        data.fetch(local_path=str(tmp_path / "nope.zip"))


def test_extract_valid_zip(mini_zip):
    """Extract on a valid zip returns the CSV path."""
    csv_path = data.extract(mini_zip)
    assert csv_path.name == "ADP_NER_history.csv"
    assert csv_path.exists()


def test_extract_non_zip(tmp_path):
    """Extract on a non-zip file raises a clear error."""
    bad = tmp_path / "not_a_zip.zip"
    bad.write_text("hello, not a zip")
    with pytest.raises(DataError) as exc:
        data.extract(bad)
    assert "not a valid zip" in str(exc.value)


def test_extract_missing_member(tmp_path):
    """Extract on a zip without the expected CSV raises a clear error."""
    zip_path = tmp_path / "wrong.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("other.csv", "a,b\n1,2\n")
    with pytest.raises(DataError) as exc:
        data.extract(zip_path)
    assert "ADP_NER_history.csv" in str(exc.value)


def test_load_raw_dtypes(mini_csv_path):
    """load_raw parses the fixture with correct dtypes."""
    df = data.load_raw(mini_csv_path)
    assert str(df["date"].dtype).startswith("datetime64")
    assert df["NER"].dtype == float
    assert df["NER_SA"].dtype == float


def test_load_raw_missing_column(tmp_path):
    """A CSV missing a required column raises an error naming it."""
    csv = tmp_path / "bad.csv"
    csv.write_text("timestep,agg_RIS,category,date,NER\nM,National,U.S.,2024-01-01,1\n")
    with pytest.raises(DataError) as exc:
        data.load_raw(csv)
    assert "NER_SA" in str(exc.value)


def test_load_raw_non_numeric(tmp_path):
    """A non-numeric NER value raises a clear error."""
    csv = tmp_path / "bad.csv"
    csv.write_text(
        "timestep,agg_RIS,category,date,NER,NER_SA\n"
        "M,National,U.S.,2024-01-01,oops,130000\n"
    )
    with pytest.raises(DataError) as exc:
        data.load_raw(csv)
    assert "NER" in str(exc.value)


def test_load_pipeline_with_zip(mini_zip):
    """The load convenience pipeline works from a local zip."""
    df = data.load(local_path=str(mini_zip))
    assert not df.empty
    assert set(["timestep", "agg_RIS", "category", "date", "NER", "NER_SA"]).issubset(df.columns)
