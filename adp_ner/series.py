"""Build the National headline monthly series and its change columns."""

from __future__ import annotations

import pandas as pd


class SeriesError(Exception):
    """Raised when the headline series cannot be built."""


def national_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Return the National U.S. monthly headline series with change columns.

    Filters to ``agg_RIS == "National"``, ``category == "U.S."`` and
    ``timestep == "M"``, sorts by date, drops exact duplicate dates keeping the
    last, and adds ``chg`` (``NER.diff()``) and ``chg_sa`` (``NER_SA.diff()``).
    The first row's changes are NaN, which is expected.
    """
    mask = (
        (df["agg_RIS"] == "National")
        & (df["category"] == "U.S.")
        & (df["timestep"] == "M")
    )
    sub = df.loc[mask].copy()
    if sub.empty:
        raise SeriesError(
            "No National U.S. monthly rows found "
            "(agg_RIS == 'National', category == 'U.S.', timestep == 'M')."
        )

    sub = sub.sort_values("date")
    sub = sub.drop_duplicates(subset="date", keep="last")
    sub = sub.set_index("date")
    sub.index.name = "date"

    sub["chg"] = sub["NER"].diff()
    sub["chg_sa"] = sub["NER_SA"].diff()
    return sub


def select_column(df: pd.DataFrame, measure: str = "level", sa: bool = True) -> pd.Series:
    """Select a column by measure (``level`` or ``change``) and SA flag."""
    if measure not in ("level", "change"):
        raise SeriesError(f"Unknown measure: {measure!r}. Use 'level' or 'change'.")
    if measure == "level":
        col = "NER_SA" if sa else "NER"
    else:
        col = "chg_sa" if sa else "chg"
    return df[col]
