"""End to end CLI tests on the committed fixture (fully offline)."""

from __future__ import annotations

import json

import pytest

from adp_ner.cli import main

# The fixture's last SA change is 131020 - 130960 = 60.0 people. naive (the
# default) carries it forward, so this is the golden forecast for the fixture.
GOLDEN_CHANGE = 60.0


def _run(argv, capsys):
    """Run the CLI with argv and return (exit_code, stdout)."""
    code = main(argv)
    out = capsys.readouterr().out
    return code, out


def _src(*extra):
    """Build an argv list pointed at the committed fixture."""
    return [*extra, "--input", "tests/fixtures/mini_history.csv"]


def test_history_runs(capsys):
    """history runs and prints a dated row."""
    code, out = _run(["history", *_src(), "--rows", "3"], capsys)
    assert code == 0
    assert "2025-06-01" in out


def test_forecast_golden(capsys):
    """A golden test pins the forecast on the fixture so the math cannot drift."""
    code, out = _run(["forecast", *_src(), "--json"], capsys)
    assert code == 0
    payload = json.loads(out)
    assert payload["model"] == "naive"
    assert payload["predicted_change"] == pytest.approx(GOLDEN_CHANGE)
    assert payload["target_date"] == "2025-07-01"


def test_smoke_history_forecast_explain_consistent(capsys):
    """history, forecast, and explain run and quote a consistent forecast."""
    _run(["history", *_src()], capsys)

    _, fc_out = _run(["forecast", *_src(), "--json"], capsys)
    fc = json.loads(fc_out)

    _, ex_out = _run(["explain", *_src(), "--json"], capsys)
    ex = json.loads(ex_out)

    assert fc["predicted_change"] == pytest.approx(ex["predicted_change"])
    assert fc["implied_level"] == pytest.approx(ex["implied_level"])
    assert fc["target_date"] == ex["target_date"]


@pytest.mark.parametrize("command", ["history", "forecast", "evaluate", "explain"])
def test_json_parses(command, capsys):
    """--json output for each subcommand parses as valid JSON."""
    _, out = _run([command, *_src(), "--json"], capsys)
    json.loads(out)  # raises if not valid JSON


def test_bad_model_exits_nonzero(capsys):
    """An invalid model name exits non-zero via argparse."""
    with pytest.raises(SystemExit) as exc:
        main(["forecast", *_src(), "--model", "bogus"])
    assert exc.value.code != 0


def test_negative_rows_exits_nonzero(capsys):
    """A non-positive --rows exits non-zero with a readable message."""
    with pytest.raises(SystemExit) as exc:
        main(["history", *_src(), "--rows", "-3"])
    assert exc.value.code != 0


def test_missing_input_is_clean_error(capsys):
    """A missing --input file gives a clear error and exit 1, not a stack trace."""
    code = main(["forecast", "--input", "does_not_exist.csv"])
    err = capsys.readouterr().err
    assert code == 1
    assert "error:" in err
    assert "not found" in err
