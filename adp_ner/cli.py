"""Command line interface for the ADP NER tracker.

Ships ``history``, ``forecast``, ``evaluate``, and ``explain``.
"""

from __future__ import annotations

import argparse
import json
import sys

from adp_ner.backtest import BacktestError, evaluate_all, make_forecast
from adp_ner.data import DEFAULT_URL, DataError, load
from adp_ner.explain import build_explanation, render_explanation
from adp_ner.models import DEFAULT_SPAN, DEFAULT_WINDOW, MODEL_NAMES, ModelError, prepare_changes
from adp_ner.series import SeriesError, national_monthly

# Chosen once by the backtest during development. On the real ADP series naive
# has the lowest RMSE (157.1k) and MAE (82.6k), beating ewma, mean, and
# seasonal_naive, so it is the hardcoded default. See the README for the table.
DEFAULT_MODEL = "naive"


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    """Add the data source flags shared across subcommands."""
    parser.add_argument("--input", dest="input_path", metavar="PATH",
                        help="Use a local zip or csv instead of downloading.")
    parser.add_argument("--url", default=DEFAULT_URL,
                        help="Override the default download URL.")
    parser.add_argument("--refresh", action="store_true",
                        help="Force a fresh download even if this month's data is cached.")


def build_parser() -> argparse.ArgumentParser:
    """Construct the top level argument parser."""
    parser = argparse.ArgumentParser(
        prog="adp",
        description="Track and forecast the ADP National Employment Report.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    hist = sub.add_parser("history", help="Show recent historical numbers.")
    hist.add_argument("--rows", type=int, default=12,
                    help="Number of most recent rows to show (default 12).")
    hist.add_argument("--measure", choices=["level", "change", "both"], default="both",
                    help="Which measure to display (default both).")
    sa_group = hist.add_mutually_exclusive_group()
    sa_group.add_argument("--sa", dest="sa", action="store_true", default=True,
                        help="Use seasonally adjusted values (default).")
    sa_group.add_argument("--nsa", dest="sa", action="store_false",
                        help="Use not seasonally adjusted values.")
    hist.add_argument("--json", dest="as_json", action="store_true",
                    help="Emit machine readable JSON.")
    _add_source_args(hist)
    hist.set_defaults(func=cmd_history)

    fc = sub.add_parser("forecast", help="Predict next month's change.")
    fc.add_argument("--model", choices=list(MODEL_NAMES), default=DEFAULT_MODEL,
                    help=f"Model to use (default {DEFAULT_MODEL}, chosen by backtest).")
    fc.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                    help=f"Trailing window for the mean model (default {DEFAULT_WINDOW}).")
    fc.add_argument("--span", type=int, default=DEFAULT_SPAN,
                    help=f"Span for the ewma model (default {DEFAULT_SPAN}).")
    fc.add_argument("--interval", type=float, default=1.0,
                    help="Band multiplier z on the backtest error std (default 1.0).")
    fc.add_argument("--json", dest="as_json", action="store_true",
                    help="Emit machine readable JSON.")
    _add_source_args(fc)
    fc.set_defaults(func=cmd_forecast)

    ev = sub.add_parser("evaluate", help="Backtest all models and show metrics.")
    ev.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                    help=f"Trailing window for the mean model (default {DEFAULT_WINDOW}).")
    ev.add_argument("--span", type=int, default=DEFAULT_SPAN,
                    help=f"Span for the ewma model (default {DEFAULT_SPAN}).")
    ev.add_argument("--json", dest="as_json", action="store_true",
                    help="Emit machine readable JSON.")
    _add_source_args(ev)
    ev.set_defaults(func=cmd_evaluate)

    ex = sub.add_parser("explain", help="Explain the current forecast in plain language.")
    ex.add_argument("--model", choices=list(MODEL_NAMES), default=DEFAULT_MODEL,
                    help=f"Model to explain (default {DEFAULT_MODEL}).")
    ex.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                    help=f"Trailing window for the mean model (default {DEFAULT_WINDOW}).")
    ex.add_argument("--span", type=int, default=DEFAULT_SPAN,
                    help=f"Span for the ewma model (default {DEFAULT_SPAN}).")
    ex.add_argument("--interval", type=float, default=1.0,
                    help="Band multiplier z on the backtest error std (default 1.0).")
    ex.add_argument("--json", dest="as_json", action="store_true",
                    help="Emit machine readable JSON.")
    _add_source_args(ex)
    ex.set_defaults(func=cmd_explain)

    return parser


def _load_series(args: argparse.Namespace):
    """Load the National monthly series from the chosen data source."""
    df = load(url=args.url, local_path=args.input_path, refresh=args.refresh)
    return national_monthly(df)


# Column widths for the plain text history table.
DATE_WIDTH = 10
NUM_WIDTH = 22


def _fmt_thousands(value: float | None) -> str:
    """Format a level or change (people) as thousands of jobs with one decimal.

    The ADP headline figure is published in thousands of jobs, so dividing by
    1000 makes the column read like the number people expect to see.
    """
    if value is None or value != value:  # None or NaN
        return "n/a"
    return f"{value / 1000:,.1f}"


def _cell(value: float) -> float | None:
    """Convert a possibly-NaN pandas value into a JSON friendly float or None."""
    return None if value != value else float(value)


def _build_records(tail, measure: str, level_col: str, chg_col: str) -> list[dict]:
    """Turn the tail of the series into a list of row dicts for output.

    Each record always carries a ``date`` and, depending on ``measure``, a
    ``level`` and/or ``change`` field in raw people (not thousands).
    """
    want_level = measure in ("level", "both")
    want_change = measure in ("change", "both")

    records = []
    for date, row in tail.iterrows():
        rec = {"date": date.strftime("%Y-%m-%d")}
        if want_level:
            rec["level"] = _cell(row[level_col])
        if want_change:
            rec["change"] = _cell(row[chg_col])
        records.append(rec)
    return records


def _render_table(records: list[dict], measure: str, adj: str) -> str:
    """Render records as a right aligned plain text table with a header row."""
    want_level = measure in ("level", "both")
    want_change = measure in ("change", "both")

    header = [f"{'date':<{DATE_WIDTH}}"]
    if want_level:
        header.append(f"{f'level {adj} (thousands)':>{NUM_WIDTH}}")
    if want_change:
        header.append(f"{f'change {adj} (thousands)':>{NUM_WIDTH}}")

    lines = ["  ".join(header)]
    for rec in records:
        cells = [f"{rec['date']:<{DATE_WIDTH}}"]
        if want_level:
            cells.append(f"{_fmt_thousands(rec['level']):>{NUM_WIDTH}}")
        if want_change:
            cells.append(f"{_fmt_thousands(rec['change']):>{NUM_WIDTH}}")
        lines.append("  ".join(cells))
    return "\n".join(lines)


def cmd_history(args: argparse.Namespace) -> int:
    """Print the most recent rows of the headline series as a table or JSON."""
    if args.rows <= 0:
        raise SystemExit("--rows must be a positive integer.")

    series = _load_series(args)
    tail = series.tail(args.rows)

    # Pick the seasonally adjusted or raw columns based on the --sa / --nsa flag.
    level_col = "NER_SA" if args.sa else "NER"
    chg_col = "chg_sa" if args.sa else "chg"
    adj = "SA" if args.sa else "NSA"

    records = _build_records(tail, args.measure, level_col, chg_col)

    if args.as_json:
        print(json.dumps({"adjustment": adj, "rows": records}, indent=2))
    else:
        print(_render_table(records, args.measure, adj))
    return 0


def _series_changes(series):
    """Return the clean SA change series and the last SA level and date."""
    changes = prepare_changes(series["chg_sa"])
    last_level = float(series["NER_SA"].iloc[-1])
    last_date = series.index[-1]
    return changes, last_level, last_date


def cmd_forecast(args: argparse.Namespace) -> int:
    """Predict next month's SA change and the implied level, with an interval."""
    series = _load_series(args)
    changes, last_level, last_date = _series_changes(series)

    fc = make_forecast(
        changes, last_level, last_date, model=args.model,
        window=args.window, span=args.span, z=args.interval,
    )

    if args.as_json:
        print(json.dumps(fc, indent=2))
        return 0

    pc = fc["predicted_change"] / 1000
    lo = fc["interval_low"] / 1000
    hi = fc["interval_high"] / 1000
    print(f"Forecast for {fc['target_date']} (model: {fc['model']})")
    print(f"  predicted change : {pc:+,.1f} thousand jobs")
    print(f"  interval (z={fc['z']:g}) : {lo:+,.1f} to {hi:+,.1f} thousand jobs")
    print(f"  implied SA level : {fc['implied_level'] / 1000:,.1f} thousand")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Backtest every model and print the metrics table sorted by RMSE."""
    series = _load_series(args)
    changes = prepare_changes(series["chg_sa"])
    table = evaluate_all(changes, window=args.window, span=args.span)

    if args.as_json:
        records = []
        for name, row in table.iterrows():
            records.append({
                "model": name,
                "MAE": float(row["MAE"]),
                "RMSE": float(row["RMSE"]),
                "directional_accuracy": float(row["directional_accuracy"]),
            })
        print(json.dumps({"metrics": records}, indent=2))
        return 0

    # Metrics are in people; show MAE and RMSE in thousands to read like jobs.
    print(f"{'model':<16}{'MAE (k)':>10}{'RMSE (k)':>10}{'dir. acc.':>12}")
    for name, row in table.iterrows():
        print(f"{name:<16}{row['MAE'] / 1000:>10.1f}{row['RMSE'] / 1000:>10.1f}"
            f"{row['directional_accuracy']:>12.0%}")
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    """Print a plain language rationale for the current forecast."""
    series = _load_series(args)
    fields = build_explanation(
        series, model=args.model, window=args.window,
        span=args.span, z=args.interval,
    )

    if args.as_json:
        print(json.dumps(fields, indent=2))
    else:
        print(render_explanation(fields), end="")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``adp`` console script and ``python -m adp_ner``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (DataError, SeriesError, ModelError, BacktestError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
