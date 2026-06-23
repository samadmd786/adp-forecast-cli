"""Tests for the explanation builder and its consistency with the forecast."""

from __future__ import annotations

import pytest

from adp_ner import data
from adp_ner.backtest import make_forecast
from adp_ner.explain import SECTIONS, build_explanation, render_explanation
from adp_ner.models import prepare_changes
from adp_ner.series import national_monthly


@pytest.fixture
def fixture_series(mini_csv_path):
    """National monthly series built from the committed fixture."""
    return national_monthly(data.load_raw(mini_csv_path))


def test_explanation_has_each_section(fixture_series):
    """The rendered explanation contains every required section heading."""
    fields = build_explanation(fixture_series, model="ewma")
    text = render_explanation(fields)
    for section in SECTIONS:
        assert f"{section}:" in text


def test_seasonality_note_present(fixture_series):
    """The seasonality note states the series is already seasonally adjusted."""
    text = render_explanation(build_explanation(fixture_series, model="naive"))
    assert "already seasonally adjusted" in text


def test_explain_matches_forecast(fixture_series):
    """The number explain quotes equals what make_forecast returns."""
    changes = prepare_changes(fixture_series["chg_sa"])
    last_level = float(fixture_series["NER_SA"].iloc[-1])
    last_date = fixture_series.index[-1]

    fc = make_forecast(changes, last_level, last_date, model="ewma")
    fields = build_explanation(fixture_series, model="ewma")

    assert fields["predicted_change"] == pytest.approx(fc["predicted_change"])
    assert fields["implied_level"] == pytest.approx(fc["implied_level"])
    assert fields["interval_low"] == pytest.approx(fc["interval_low"])
    assert fields["interval_high"] == pytest.approx(fc["interval_high"])
    assert fields["target_date"] == fc["target_date"]


def test_explain_naive_equals_naive_baseline(fixture_series):
    """For the naive model, the forecast equals the naive baseline change."""
    fields = build_explanation(fixture_series, model="naive")
    assert fields["vs_naive"] == pytest.approx(0.0)


def test_explain_fields_have_expected_keys(fixture_series):
    """The structured fields expose the keys the JSON output relies on."""
    fields = build_explanation(fixture_series, model="ewma")
    expected = {
        "model", "target_date", "predicted_change", "implied_level",
        "interval_low", "interval_high", "z", "winner", "chosen_rmse",
        "recent_changes", "recent_average", "momentum", "naive_change", "vs_naive",
    }
    assert expected.issubset(fields.keys())
