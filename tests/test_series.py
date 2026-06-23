"""Tests for the National monthly series builder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adp_ner import data
from adp_ner.series import SeriesError, national_monthly, select_column


def test_national_monthly_filters_and_sorts(mini_csv_path):
    """Only National U.S. monthly rows survive, sorted with increasing dates."""
    df = data.load_raw(mini_csv_path)
    series = national_monthly(df)

    # 18 monthly National U.S. points in the fixture.
    assert len(series) == 18
    assert series.index.is_monotonic_increasing
    assert series.index.is_unique
    # No weekly or other agg_RIS rows leaked in.
    assert series.index.min() == pd.Timestamp("2024-01-01")
    assert series.index.max() == pd.Timestamp("2025-06-01")


def test_chg_sa_matches_manual_diff(mini_csv_path):
    """chg_sa equals the manual NER_SA.diff() on the fixture."""
    df = data.load_raw(mini_csv_path)
    series = national_monthly(df)
    expected = series["NER_SA"].diff()
    pd.testing.assert_series_equal(series["chg_sa"], expected, check_names=False)
    assert np.isnan(series["chg_sa"].iloc[0])


def test_dedup_keeps_last():
    """Duplicate date rows are de-duplicated, keeping the last."""
    df = pd.DataFrame({
        "timestep": ["M", "M", "M"],
        "agg_RIS": ["National", "National", "National"],
        "category": ["U.S.", "U.S.", "U.S."],
        "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-02-01"]),
        "NER": [100.0, 200.0, 250.0],
        "NER_SA": [100.0, 200.0, 250.0],
    })
    series = national_monthly(df)
    assert len(series) == 2
    assert series.loc[pd.Timestamp("2024-02-01"), "NER_SA"] == 250.0


def test_single_row_yields_nan_change():
    """A single row yields a NaN change and does not crash."""
    df = pd.DataFrame({
        "timestep": ["M"],
        "agg_RIS": ["National"],
        "category": ["U.S."],
        "date": pd.to_datetime(["2024-01-01"]),
        "NER": [100.0],
        "NER_SA": [100.0],
    })
    series = national_monthly(df)
    assert len(series) == 1
    assert np.isnan(series["chg_sa"].iloc[0])


def test_empty_selection_raises():
    """No matching rows raises a clear error."""
    df = pd.DataFrame({
        "timestep": ["W"],
        "agg_RIS": ["Industry"],
        "category": ["Manufacturing"],
        "date": pd.to_datetime(["2024-01-01"]),
        "NER": [1.0],
        "NER_SA": [1.0],
    })
    with pytest.raises(SeriesError):
        national_monthly(df)


def test_select_column(mini_csv_path):
    """select_column picks the right column per measure and SA flag."""
    df = data.load_raw(mini_csv_path)
    series = national_monthly(df)
    assert select_column(series, "level", sa=True).name == "NER_SA"
    assert select_column(series, "level", sa=False).name == "NER"
    assert select_column(series, "change", sa=True).name == "chg_sa"
    assert select_column(series, "change", sa=False).name == "chg"
    with pytest.raises(SeriesError):
        select_column(series, "bogus")
