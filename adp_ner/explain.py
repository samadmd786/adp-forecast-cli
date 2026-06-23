"""Human readable rationale for the current forecast.

Everything here is built on top of ``make_forecast`` and ``evaluate_all`` so the
numbers quoted in the explanation are guaranteed to match what ``forecast`` and
``evaluate`` print. The goal is prose a person can follow and reproduce by hand.
"""

from __future__ import annotations

import pandas as pd

from adp_ner.backtest import DEFAULT_MIN_TRAIN, evaluate_all, make_forecast
from adp_ner.models import SEASONAL_LAG, naive, prepare_changes

# Section headings, kept as constants so tests can assert each one is present.
SECTION_MODEL = "Model"
SECTION_TREND = "Recent trend"
SECTION_METHOD = "How the forecast was made"
SECTION_SEASONALITY = "Seasonality"
SECTION_WALK = "Step by step"
SECTION_UNCERTAINTY = "Uncertainty"
SECTION_VS_NAIVE = "Versus the naive guess"

SECTIONS = (
    SECTION_MODEL,
    SECTION_TREND,
    SECTION_METHOD,
    SECTION_SEASONALITY,
    SECTION_WALK,
    SECTION_UNCERTAINTY,
    SECTION_VS_NAIVE,
)


def _k(value: float) -> str:
    """Format a signed people count (a change) as thousands of jobs."""
    return f"{value / 1000:+,.1f}k"


def _mag(value: float) -> str:
    """Format a magnitude (such as RMSE) as thousands of jobs, no sign."""
    return f"{value / 1000:,.1f}k"


def build_explanation(series: pd.DataFrame, model: str = "naive", window: int = 6,
                      span: int = 6, z: float = 1.0,
                      min_train: int = DEFAULT_MIN_TRAIN, recent_n: int = 6) -> dict:
    """Assemble structured explanation fields for the current forecast.

    Returns a dict of plain fields (numbers in people, plus a few formatted
    strings) that both the JSON output and the prose renderer consume.
    """
    changes = prepare_changes(series["chg_sa"])
    last_level = float(series["NER_SA"].iloc[-1])
    last_date = series.index[-1]

    fc = make_forecast(changes, last_level, last_date, model=model,
                       window=window, span=span, z=z, min_train=min_train)

    # Backtest every model so we can say what the chosen model beat.
    table = evaluate_all(changes, min_train=min_train, window=window, span=span)
    winner = str(table["RMSE"].idxmin())
    chosen_rmse = float(table.loc[model, "RMSE"])
    others = table.drop(index=model)
    runner_up = str(others["RMSE"].idxmin())
    runner_up_rmse = float(others.loc[runner_up, "RMSE"])

    # Recent history and a simple momentum read (last third vs the third before).
    recent = changes.iloc[-recent_n:]
    recent_changes = [
        {"date": d.strftime("%Y-%m-%d"), "change": float(v)}
        for d, v in recent.items()
    ]
    recent_average = float(recent.mean())
    third = max(1, recent_n // 3)
    momentum = "building" if recent.iloc[-third:].mean() >= recent.iloc[:third].mean() else "fading"

    naive_change = naive(changes)
    vs_naive = fc["predicted_change"] - naive_change

    # The exact value seasonal_naive copies forward: the change SEASONAL_LAG
    # months back. This is what the seasonal_naive explanation must quote, not
    # the oldest entry in the recent window.
    if len(changes) > SEASONAL_LAG:
        src_date = changes.index[-SEASONAL_LAG]
        seasonal_source = {
            "date": src_date.strftime("%Y-%m-%d"),
            "change": float(changes.iloc[-SEASONAL_LAG]),
        }
    else:
        seasonal_source = None

    return {
        "model": model,
        "target_date": fc["target_date"],
        "predicted_change": fc["predicted_change"],
        "implied_level": fc["implied_level"],
        "interval_low": fc["interval_low"],
        "interval_high": fc["interval_high"],
        "z": z,
        "window": window,
        "span": span,
        "is_winner": model == winner,
        "winner": winner,
        "chosen_rmse": chosen_rmse,
        "runner_up": runner_up,
        "runner_up_rmse": runner_up_rmse,
        "recent_changes": recent_changes,
        "recent_average": recent_average,
        "momentum": momentum,
        "naive_change": naive_change,
        "vs_naive": vs_naive,
        "seasonal_source": seasonal_source,
    }


def _method_sentence(fields: dict) -> str:
    """Describe how the chosen model turns history into the point forecast."""
    model = fields["model"]
    recent = fields["recent_changes"]
    point = fields["predicted_change"]

    if model == "naive":
        last = recent[-1]
        return (
            f"The naive model simply carries the last observed change forward. "
            f"The change for {last['date']} was {_k(last['change'])}, so that is "
            f"the forecast."
        )
    if model == "mean":
        w = fields["window"]
        return (
            f"The mean model averages the last {w} monthly changes. Their average "
            f"is {_k(point)}, which becomes the forecast."
        )
    if model == "ewma":
        vals = ", ".join(_k(r["change"]) for r in recent)
        return (
            f"The ewma model takes an exponentially weighted mean of past changes "
            f"with the most recent months weighted most heavily. Recent changes "
            f"were {vals}, giving {_k(point)}."
        )
    if model == "seasonal_naive":
        copied = fields["seasonal_source"]
        return (
            f"The seasonal_naive model copies the change from twelve months "
            f"earlier, which was {copied['date']} at {_k(copied['change'])}."
        )
    return f"The {model} model produced {_k(point)}."


def render_explanation(fields: dict) -> str:
    """Render the structured fields as plain language prose with section headings."""
    point = fields["predicted_change"]
    recent = fields["recent_changes"]

    # Model section.
    if fields["is_winner"]:
        model_text = (
            f"The {fields['model']} model produced this number. It is the default "
            f"because it had the lowest backtest RMSE ({_mag(fields['chosen_rmse'])} "
            f"of jobs), beating the next best model {fields['runner_up']} "
            f"({_mag(fields['runner_up_rmse'])})."
        )
    else:
        model_text = (
            f"The {fields['model']} model produced this number. Note it is not the "
            f"backtest winner: {fields['winner']} had the lowest RMSE, while "
            f"{fields['model']} scored {_mag(fields['chosen_rmse'])}."
        )

    # Recent trend section.
    trend_list = ", ".join(f"{r['date']} {_k(r['change'])}" for r in recent)
    trend_text = (
        f"The last {len(recent)} monthly changes were: {trend_list}. Their average "
        f"is {_k(fields['recent_average'])} and momentum is {fields['momentum']}."
    )

    # Step by step walk.
    walk_text = (
        f"Start from the recent changes above, apply the {fields['model']} rule, "
        f"and you arrive at {_k(point)} for {fields['target_date']}. Add that to "
        f"the last SA level to get an implied level of "
        f"{fields['implied_level'] / 1000:,.1f}k."
    )

    # Seasonality note.
    seasonality_text = (
        "We do not model month of year effects. The series is already seasonally "
        "adjusted, so there is little seasonal signal left to capture and any month "
        "of year component would come out small."
    )

    # Uncertainty section.
    uncertainty_text = (
        f"The interval is {_k(fields['interval_low'])} to {_k(fields['interval_high'])} "
        f"(z = {fields['z']:g}). Its width comes from how far this model's past one "
        f"step ahead forecasts have missed in the backtest, not from the in sample fit."
    )

    # Versus naive.
    if fields["model"] == "naive":
        vs_text = (
            "This forecast is the naive guess, since the chosen model is naive "
            "itself. It is the baseline every other model is measured against."
        )
    else:
        vs_text = (
            f"The naive guess (carry the last change forward) would be "
            f"{_k(fields['naive_change'])}. This forecast differs from it by "
            f"{_k(fields['vs_naive'])}."
        )

    blocks = [
        (SECTION_MODEL, model_text),
        (SECTION_TREND, trend_text),
        (SECTION_METHOD, _method_sentence(fields)),
        (SECTION_SEASONALITY, seasonality_text),
        (SECTION_WALK, walk_text),
        (SECTION_UNCERTAINTY, uncertainty_text),
        (SECTION_VS_NAIVE, vs_text),
    ]

    lines = [f"Forecast for {fields['target_date']}: {_k(point)} of jobs", ""]
    for heading, text in blocks:
        lines.append(f"{heading}:")
        lines.append(f"  {text}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
