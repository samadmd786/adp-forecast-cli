"""Baseline forecast models for the month over month SA change series.

The forecast target is the month over month change in the seasonally adjusted
employment level (``chg_sa``). Every model is a small, pure function that takes
the change series (leading NaN already dropped) and returns the next change.
Only pandas is used here, by design: the series is a single, already seasonally
adjusted monthly line, so simple baselines are the right tool. (The backtest
metrics in backtest.py use numpy; the models themselves do not need it.)
"""

from __future__ import annotations

import pandas as pd

# Defaults shared by the models and the CLI.
DEFAULT_WINDOW = 6
DEFAULT_SPAN = 6
SEASONAL_LAG = 12


class ModelError(Exception):
    """Raised when a model cannot produce a prediction for the given series."""


def prepare_changes(series: pd.Series) -> pd.Series:
    """Drop the leading NaN that ``.diff()`` produces so models see clean data.

    The first change is always NaN (no prior month to difference against). If it
    is not removed it would poison means and exponential weights, so every model
    consumes the output of this helper rather than the raw column.
    """
    clean = series.dropna()
    if clean.empty:
        raise ModelError("Change series is empty after dropping NaN values.")
    return clean


def naive(changes: pd.Series) -> float:
    """Predict the next change as the last observed change."""
    return float(changes.iloc[-1])


def seasonal_naive(changes: pd.Series, lag: int = SEASONAL_LAG) -> float:
    """Predict the next change as the change ``lag`` months earlier.

    Needs at least ``lag + 1`` points: a full year of lag plus the current
    observation, so the lagged value lines up with the month being predicted.
    """
    if len(changes) <= lag:
        raise ModelError(
            f"seasonal_naive needs more than {lag} points to look back "
            f"{lag} months, got {len(changes)}."
        )
    return float(changes.iloc[-lag])


def mean_window(changes: pd.Series, window: int = DEFAULT_WINDOW) -> float:
    """Predict the next change as the trailing mean of the last ``window`` changes."""
    if window <= 0:
        raise ModelError(f"window must be positive, got {window}.")
    return float(changes.iloc[-window:].mean())


def ewma(changes: pd.Series, span: int = DEFAULT_SPAN) -> float:
    """Predict the next change as the exponentially weighted mean of past changes.

    Recent months carry more weight than older ones, which makes this the most
    expressive baseline while staying easy to explain. ``adjust=False`` gives the
    standard recursive EWMA so the result is reproducible by hand.
    """
    if span <= 0:
        raise ModelError(f"span must be positive, got {span}.")
    return float(changes.ewm(span=span, adjust=False).mean().iloc[-1])


# Registry mapping CLI model names to a function of (changes, **params) -> float.
def _predict(name: str, changes: pd.Series, window: int = DEFAULT_WINDOW,
             span: int = DEFAULT_SPAN) -> float:
    """Dispatch to the named model, passing only the parameters it uses."""
    if name == "naive":
        return naive(changes)
    if name == "seasonal_naive":
        return seasonal_naive(changes)
    if name == "mean":
        return mean_window(changes, window=window)
    if name == "ewma":
        return ewma(changes, span=span)
    raise ModelError(f"Unknown model: {name!r}.")


MODEL_NAMES = ("naive", "seasonal_naive", "mean", "ewma")


def predict_next(changes: pd.Series, model: str = "ewma", window: int = DEFAULT_WINDOW,
                 span: int = DEFAULT_SPAN) -> float:
    """Public entry point: predict the next change with the named model."""
    return _predict(model, changes, window=window, span=span)
