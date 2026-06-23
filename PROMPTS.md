# PROMPTS

A log of the AI sessions used to build this project. Tool: Claude Code running
Claude Opus 4.8. The prompts below are the real ones sent, in order, with notes
on what was done with each output, including the dead ends.

## Session 1: scaffolding and Phase 1 (data layer and history view)

**Prompts sent:**

> use PLAN.md to build this project add requirments.txt too

> only build phase 1 for now

**What was done:** Claude read `PLAN.md`, then built the project layout,
`requirements.txt`, `pyproject.toml` (console script `adp`), gitignored `data/`,
and the Phase 1 code: `data.py` (fetch once / cache / extract / load_raw),
`series.py` (`national_monthly` with `chg` and `chg_sa`), and the `history`
subcommand. It also wrote the hand made `tests/fixtures/mini_history.csv` and the
`test_data.py` / `test_series.py` suites. Used mostly as is.

**Dead ends and fixes:**

- The system Python was externally managed (PEP 668), so `pip install -e .`
  refused. Switched to a `.venv`. Not a code problem, just environment.
- First test run failed: `test_load_raw_dtypes` expected float but the fixture's
  all integer `NER` column parsed as `int64`. Fixed by having `load_raw` coerce
  `NER` and `NER_SA` to float explicitly, which is also what the spec asked for.
  Output kept after the fix.

## Session 2: readability pass

**Prompt sent:**

> add comments and doc strings where required to incrase readability, write clear functions

**What was done:** Claude refactored the dense table building in `cmd_history`
into small helpers (`_fmt_thousands`, `_cell`, `_build_records`, `_render_table`)
and added docstrings and short comments across the modules. Used as is.

## Session 3: Phase 2 (forecasting engine and evaluation)

**Prompt sent:**

> phase 1 worked, update pytest to use -v option, now start phase 2

**What was done:** Added `addopts = "-v"` to the pytest config. Built `models.py`
(four pure baselines plus `prepare_changes`), `backtest.py` (walk-forward backtest
with a leakage assertion, MAE / RMSE / directional accuracy, prediction interval,
and a `make_forecast` source of truth), and the `forecast` and `evaluate`
subcommands. Wrote `test_models.py` and `test_backtest.py`. Used as is.

**Design decision worth recording:** `seasonal_naive` requires more than 12
points (a full year of lag plus the current observation) so the lagged value
lines up with the month being predicted. The plan's "shorter than 13 points
raises" was implemented literally.

## Session 4: settling the default model

**Prompt sent (real backtest output pasted in):**

> adp evaluate
> model              MAE (k)  RMSE (k)   dir. acc.
> naive                 82.6     157.1         93%
> ewma                 111.1     221.5         90%
> mean                 124.8     261.3         89%
> seasonal_naive       203.2     400.3         84%

**What was done:** The plan said to pick the default once from the real backtest.
The starting default was `ewma`, but `naive` had the lowest RMSE on the real
series, so `DEFAULT_MODEL` was changed to `naive` and the reasoning written into
the code comment and the README. This is the "rejected and replaced" case: the
`ewma` default was a reasonable prior that the data overruled.

## Session 5: Phase 3 (explanation, docs, polish)

**Prompt sent:**

> start phase3

**What was done:** Built `explain.py` (`build_explanation` on top of
`make_forecast` and `evaluate_all` so the numbers match `forecast`, plus a prose
renderer with the required sections) and the `explain` subcommand. Wrote
`test_explain.py` (section coverage, forecast consistency, JSON keys) and
`test_cli.py` (end to end smoke, a golden forecast pin on the fixture, `--json`
validity, and argument error exit codes). Wrote this `README.md` and `PROMPTS.md`.

**Small fix:** the first explain output formatted RMSE with a leading `+` because
it reused the signed change formatter. Added a separate unsigned `_mag` helper for
magnitudes like RMSE.

## Session 6: tradeoffs document

**Prompt sent:**

> now write a tradeoff.md to include pro and con of this architecture, I used
> argparser, numpy and pandas because I have used them earlier and I am
> comfortable with the syntax and commands but I am sure there might be a better
> way to do it

**What was done:** Claude wrote `TRADEOFFS.md` organized into Tooling and
Architecture, each choice as pros and cons with a concrete "a better way might be"
note (typer for the CLI, statsmodels or scikit-learn for the models, cache
freshness checks). The user's framing, that the familiar stack was deliberate but
maybe not optimal, was carried through. Committed on its own.

## Session 7: surface the tradeoffs in the README

**Prompts sent:**

> add a tradeoff section in readme and add a link to this tradeoff.md

> write like this I deliberately chose to use standard library argparse and
> pandas to prioritize system stability, offline reliability, and minimal
> dependency bloat. A detailed analysis of these architectural choices, including
> why heavier ML libraries were omitted and how the backtest informed the system
> design, can be found in the tradeoff sectiom

**What was done:** Claude first added a short Tradeoffs section to the README
linking `TRADEOFFS.md`. The user then supplied their own preferred wording, so the
section was rewritten in that voice (and the "sectiom" typo corrected to
"section"). Used as dictated. Committed.

## Session 8: code review with CodeRabbit

**Prompt sent:**

> /code-review

**What was done:** Ran the CodeRabbit CLI (`coderabbit review --agent`) against
the initial commit to cover the whole codebase. It returned two findings, both
valid and both fixed:

- Major, `explain.py`: the seasonal_naive explanation quoted the oldest entry in
  the recent six month window (six months back) instead of the change twelve
  months back that the model actually copies. The forecast number was correct but
  the rationale cited the wrong source month. Fixed by adding a `seasonal_source`
  field built from `changes.iloc[-SEASONAL_LAG]` and quoting it, plus a regression
  test pinning the source to twelve months before the target.
- Minor, `models.py`: the module docstring claimed numpy and pandas were both used,
  but the module imports only pandas. Corrected the docstring.

Both fixes used as is. Test count went from 56 to 57. A verify re-run hit the free
tier rate limit, but the fixes were confirmed by the suite and by inspecting the
seasonal_naive output directly.

## Session 9: monthly cache freshness and curl download

**Prompt sent:**

> add a timestamp in data folder, check if tdata/ADP_NER_history.csv exist and if
> the timestamp is not of current month then download csv from internet using curl
> else use the downloaded csv

**What was done:** Reworked the caching in `data.py`. The old behavior reused the
data forever as long as the zip was present; now `fetch` reuses the extracted
`ADP_NER_history.csv` only while a sibling `ADP_NER_history.timestamp` file is from
the current calendar month, and otherwise downloads a fresh archive with `curl`
(replacing `urllib.request.urlretrieve`) and writes a new timestamp. Added the
helpers `_timestamp_is_current_month` and `_curl_download`, changed `--refresh` to
delete the timestamp so the cache reads as stale, and updated the CLI help text.
The three fetch tests were rewritten to monkeypatch `_curl_download` and cover the
fresh download, current-month reuse, and stale-month re-download cases. Full suite
stayed green at 58 tests. README's setup/run sections were rewritten to document
the `curl` requirement and the once-a-month refresh.

**Design note:** the ADP report is published monthly, so a calendar-month check is
the simplest freshness signal that avoids both serving stale data into a new month
and re-downloading needlessly within a month. This implements the "cache freshness
checks" idea that `TRADEOFFS.md` had flagged as a better way.
