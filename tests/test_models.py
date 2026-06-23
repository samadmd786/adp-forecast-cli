"""Tests for the baseline forecast models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adp_ner import models
from adp_ner.models import (
    ModelError,
    ewma,
    mean_window,
    naive,
    predict_next,
    prepare_changes,
    seasonal_naive,
)


def _changes(values):
    """Build a monthly indexed change series from a list of values."""
    idx = pd.date_range("2024-01-01", periods=len(values), freq="MS")
    return pd.Series(values, index=idx, dtype=float)


def test_prepare_drops_leading_nan():
    """The leading NaN from .diff() is dropped before any model sees the series."""
    levels = _changes([100, 110, 130, 145])
    changes = levels.diff()
    assert np.isnan(changes.iloc[0])
    clean = prepare_changes(changes)
    assert len(clean) == 3
    assert not clean.isna().any()


def test_naive_returns_last_change():
    """naive returns the last change exactly."""
    s = _changes([10, 20, 35, 50])
    assert naive(s) == 50.0


def test_seasonal_naive_returns_lag_12():
    """seasonal_naive returns the value from 12 steps back."""
    # 13 points so the prediction target lines up with the first element.
    values = list(range(1, 14))  # 1..13
    s = _changes(values)
    assert seasonal_naive(s) == float(values[-12])  # == 1.0


def test_seasonal_naive_too_short_raises():
    """A series with 12 or fewer points raises a clear error."""
    s = _changes(list(range(12)))
    with pytest.raises(ModelError):
        seasonal_naive(s)


def test_mean_window_matches_manual():
    """mean_window equals the manual trailing mean."""
    s = _changes([10, 20, 30, 40, 50, 60, 70])
    assert mean_window(s, window=3) == pytest.approx(np.mean([50, 60, 70]))


def test_ewma_matches_pandas():
    """ewma matches a direct pandas .ewm(...).mean() computation."""
    s = _changes([5, 8, 13, 21, 34])
    expected = s.ewm(span=3, adjust=False).mean().iloc[-1]
    assert ewma(s, span=3) == pytest.approx(expected)


@pytest.mark.parametrize("model", models.MODEL_NAMES)
def test_constant_level_gives_zero_change(model):
    """A constant level series has zero change, so every model predicts zero."""
    # 14 points of constant changes (all zero) so seasonal_naive can run.
    s = _changes([0.0] * 14)
    assert predict_next(s, model=model) == pytest.approx(0.0)


@pytest.mark.parametrize("model", ["naive", "mean", "ewma"])
def test_linear_trend_returns_slope(model):
    """A linear level trend yields a constant slope change for these models."""
    slope = 7.0
    s = _changes([slope] * 14)  # diff of a linear level is the constant slope
    assert predict_next(s, model=model) == pytest.approx(slope)


def test_predict_next_unknown_model_raises():
    """An unknown model name raises a clear error."""
    s = _changes([1, 2, 3])
    with pytest.raises(ModelError):
        predict_next(s, model="bogus")


def test_prepare_empty_raises():
    """An all-NaN series raises after dropping."""
    s = pd.Series([np.nan, np.nan])
    with pytest.raises(ModelError):
        prepare_changes(s)
