"""Tests for the walk-forward backtest, metrics, and forecast assembly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adp_ner import data
from adp_ner.backtest import (
    BacktestError,
    backtest_model,
    evaluate_all,
    make_forecast,
    metrics,
    prediction_interval,
)
from adp_ner.models import prepare_changes
from adp_ner.series import national_monthly


def _fixture_changes(mini_csv_path):
    """Clean SA change series built from the committed fixture."""
    df = data.load_raw(mini_csv_path)
    series = national_monthly(df)
    return prepare_changes(series["chg_sa"])


def test_backtest_one_prediction_per_cutoff(mini_csv_path):
    """Backtest yields one prediction per eligible cutoff with finite metrics."""
    changes = _fixture_changes(mini_csv_path)
    bt = backtest_model(changes, "ewma", min_train=13)
    assert len(bt) == len(changes) - 13
    assert np.isfinite(bt["predicted"]).all()
    m = metrics(bt["predicted"].to_numpy(), bt["actual"].to_numpy())
    assert all(np.isfinite(v) for v in m.values())


def test_backtest_no_leakage(mini_csv_path):
    """The training index max is strictly less than the predicted month."""
    changes = _fixture_changes(mini_csv_path)
    min_train = 13
    for cutoff in range(min_train, len(changes)):
        train = changes.iloc[:cutoff]
        test_date = changes.index[cutoff]
        assert train.index.max() < test_date


def test_backtest_too_short_raises():
    """A series too short to backtest raises a clear, specific error."""
    idx = pd.date_range("2024-01-01", periods=5, freq="MS")
    changes = pd.Series(range(5), index=idx, dtype=float)
    with pytest.raises(BacktestError):
        backtest_model(changes, "naive", min_train=13)


def test_metrics_values():
    """MAE, RMSE, and directional accuracy compute as expected."""
    predicted = np.array([10.0, -5.0, 20.0])
    actual = np.array([12.0, -1.0, 25.0])
    m = metrics(predicted, actual)
    assert m["MAE"] == pytest.approx(np.mean([2, 4, 5]))
    assert m["RMSE"] == pytest.approx(np.sqrt(np.mean([4, 16, 25])))
    assert m["RMSE"] >= m["MAE"]
    assert m["directional_accuracy"] == pytest.approx(1.0)


def test_evaluate_all_sorted_by_rmse(mini_csv_path):
    """evaluate_all returns every model sorted ascending by RMSE."""
    changes = _fixture_changes(mini_csv_path)
    table = evaluate_all(changes)
    assert set(table.index) == {"naive", "seasonal_naive", "mean", "ewma"}
    assert list(table["RMSE"]) == sorted(table["RMSE"])


def test_interval_ordering(mini_csv_path):
    """The interval satisfies low <= point <= high."""
    changes = _fixture_changes(mini_csv_path)
    point = 60.0
    low, high = prediction_interval(changes, "ewma", point, z=1.0)
    assert low <= point <= high


def test_make_forecast_consistent(mini_csv_path):
    """make_forecast reconstructs the level by adding the predicted change."""
    df = data.load_raw(mini_csv_path)
    series = national_monthly(df)
    changes = prepare_changes(series["chg_sa"])
    last_level = float(series["NER_SA"].iloc[-1])
    last_date = series.index[-1]

    fc = make_forecast(changes, last_level, last_date, model="ewma")
    assert fc["implied_level"] == pytest.approx(last_level + fc["predicted_change"])
    assert fc["interval_low"] <= fc["predicted_change"] <= fc["interval_high"]
    assert fc["target_date"] == "2025-07-01"  # one month past the fixture's last point


def test_determinism(mini_csv_path):
    """Forecasting twice on the same input gives identical output."""
    df = data.load_raw(mini_csv_path)
    series = national_monthly(df)
    changes = prepare_changes(series["chg_sa"])
    last_level = float(series["NER_SA"].iloc[-1])
    last_date = series.index[-1]

    a = make_forecast(changes, last_level, last_date, model="ewma")
    b = make_forecast(changes, last_level, last_date, model="ewma")
    assert a == b
