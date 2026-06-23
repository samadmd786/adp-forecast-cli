"""Walk-forward backtesting, metrics, and prediction intervals.

Everything here uses an expanding window, one step ahead scheme: for each cutoff
we train on the history up to that point and predict the very next change, then
compare against what actually happened. This is the honest way to measure how a
model does out of sample, and it is what the prediction interval is built from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from adp_ner.models import MODEL_NAMES, ModelError, _predict, predict_next

# Minimum training points before the first prediction. seasonal_naive needs a
# full year of lag plus one, so 13 lets every model run from the same start.
DEFAULT_MIN_TRAIN = 13


class BacktestError(Exception):
    """Raised when a series is too short or otherwise cannot be backtested."""


def backtest_model(changes: pd.Series, model: str, min_train: int = DEFAULT_MIN_TRAIN,
                   window: int = 6, span: int = 6) -> pd.DataFrame:
    """Run an expanding window, one step ahead backtest for a single model.

    Returns a DataFrame indexed by the predicted month with ``predicted`` and
    ``actual`` columns. Raises if the series is too short to make even one
    prediction. Asserts that no training slice ever includes the test point.
    """
    if len(changes) <= min_train:
        raise BacktestError(
            f"Need more than {min_train} change points to backtest, "
            f"got {len(changes)}. Provide a longer history."
        )

    rows = []
    # Predict each point from min_train onward using only earlier data.
    for cutoff in range(min_train, len(changes)):
        train = changes.iloc[:cutoff]
        test_date = changes.index[cutoff]
        actual = float(changes.iloc[cutoff])

        # Leakage guard: training must end strictly before the predicted month.
        assert train.index.max() < test_date, "leakage: train includes test point"

        predicted = _predict(model, train, window=window, span=span)
        rows.append({"date": test_date, "predicted": predicted, "actual": actual})

    result = pd.DataFrame(rows).set_index("date")
    return result


def metrics(predicted: np.ndarray, actual: np.ndarray) -> dict:
    """Compute MAE, RMSE, and directional accuracy from prediction arrays.

    MAE and RMSE are in the same units as the target (people). RMSE squares the
    misses so a few large errors push it well above MAE. Directional accuracy is
    the share of months where the predicted sign matched the actual sign.
    """
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    errors = predicted - actual

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    directional = float(np.mean(np.sign(predicted) == np.sign(actual)))
    return {"MAE": mae, "RMSE": rmse, "directional_accuracy": directional}


def evaluate_all(changes: pd.Series, min_train: int = DEFAULT_MIN_TRAIN,
                 window: int = 6, span: int = 6) -> pd.DataFrame:
    """Backtest every model and return a tidy metrics table sorted by RMSE."""
    rows = []
    for name in MODEL_NAMES:
        bt = backtest_model(changes, name, min_train=min_train, window=window, span=span)
        m = metrics(bt["predicted"].to_numpy(), bt["actual"].to_numpy())
        m["model"] = name
        rows.append(m)

    table = pd.DataFrame(rows).set_index("model")
    table = table[["MAE", "RMSE", "directional_accuracy"]]
    return table.sort_values("RMSE")


def prediction_interval(changes: pd.Series, model: str, point: float, z: float = 1.0,
                        min_train: int = DEFAULT_MIN_TRAIN, window: int = 6,
                        span: int = 6) -> tuple[float, float]:
    """Build a band around ``point`` from the model's own backtest errors.

    The width is ``z`` times the standard deviation of the one step ahead
    backtest errors, so it reflects how far this model has actually missed out of
    sample. The return is guaranteed to satisfy low <= point <= high.
    """
    bt = backtest_model(changes, model, min_train=min_train, window=window, span=span)
    errors = bt["predicted"].to_numpy() - bt["actual"].to_numpy()
    std = float(np.std(errors)) if len(errors) > 0 else 0.0
    half = abs(z) * std
    return point - half, point + half


def make_forecast(changes: pd.Series, last_level: float, last_date: pd.Timestamp,
                  model: str = "ewma", window: int = 6, span: int = 6,
                  z: float = 1.0, min_train: int = DEFAULT_MIN_TRAIN) -> dict:
    """Produce the full next month forecast: point, level, and interval.

    Forecasts the SA change with the chosen model, reconstructs the implied SA
    level by adding that change to the last known level, and wraps it in an
    interval drawn from the model's own backtest errors. This single function is
    the source of truth so the ``forecast`` and ``explain`` commands agree.
    """
    predicted_change = predict_next(changes, model=model, window=window, span=span)
    implied_level = last_level + predicted_change
    low, high = prediction_interval(
        changes, model, predicted_change, z=z, min_train=min_train,
        window=window, span=span,
    )
    target_date = last_date + pd.offsets.MonthBegin(1)

    return {
        "model": model,
        "target_date": target_date.strftime("%Y-%m-%d"),
        "predicted_change": predicted_change,
        "last_level": float(last_level),
        "implied_level": implied_level,
        "interval_low": low,
        "interval_high": high,
        "z": z,
    }
