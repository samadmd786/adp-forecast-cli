# Build Plan: ADP National Employment Report Tracker and Forecaster

This is the plan for Claude Code. Build it in three phases. Stop at the end of each phase so I can verify it manually before you move on. Do not start a later phase until I say the previous one passed.

## Ground truth about the data (already verified, do not re-derive blindly)

The download URL is:

```
https://adpemploymentreport.com/artifacts/us_ner/20260603/ADP_NER_history.zip
```

The zip contains two files: `ADP_NER_history.csv` and `license.txt`.

The CSV is long format with these exact columns:

| column   | meaning                                                        |
|----------|----------------------------------------------------------------|
| timestep | `M` (monthly) or `W` (weekly)                                  |
| agg_RIS  | one of `National`, `Industry`, `Census Divisions`, `Establishment Size` |
| category | the slice within agg_RIS, e.g. `U.S.`, `East North Central`    |
| date     | first of month, e.g. `2026-05-01`                              |
| NER      | employment level, not seasonally adjusted                      |
| NER_SA   | employment level, seasonally adjusted                          |

Facts to bake in as sanity checks:

- The headline series is the rows where `agg_RIS == "National"` and `category == "U.S."` and `timestep == "M"`. That is 197 monthly points, 2010-01-01 through 2026-05-01, no nulls.
- The file stores **levels**. The number ADP reports each month is the **month over month change in the seasonally adjusted level**, i.e. `NER_SA.diff()`. That is the forecast target.
- Expected reference values for the latest point: May 2026 SA level is about 132,624,000 and the implied monthly change is about +122,000. Recent six changes (Dec 2025 to May 2026) are roughly: 37k, 11k, 66k, 61k, 105k, 122k. Use these to confirm parsing in Phase 1 and forecasting in Phase 2.
- `NER_SA` is already seasonally adjusted, so any seasonal (month of year) component the model estimates on the SA change series should come out small. Treat that as an explainability sanity check, not a bug.

### Licensing constraint

`license.txt` states the data is ADP's property and may not be reproduced commercially. So **do not commit the raw CSV or the zip to the repo**. Gitignore the `data/` dir where the download is stored. For tests, commit one tiny hand made CSV fixture that we wrote ourselves (not real ADP data), so nothing copyrighted lands in the repo. Note this in the README.

## Hard requirements (apply to all phases)

- Python 3.10+. CLI built with `argparse`. Forecasting uses only `numpy` and `pandas`. No `statsmodels`, no `scikit-learn`, no `prophet`. Exponential smoothing uses `pandas` `.ewm`, and the metrics are plain numpy. If any regression model gets added later as an extension, it must be done with `numpy.linalg.lstsq`, not a stats library.
- The three user stories must all work by the end: see historical numbers, see the prediction for next month, understand why that prediction was made.
- Every phase ends in a runnable state with passing tests. Use `pytest`.
- Tests must run offline. Mock the network for the download tests and use the committed `mini_history.csv` fixture (zipped on the fly when a zip is needed) for everything else. No test may hit the real URL.
- Deterministic output. No unseeded randomness anywhere in the forecast path.
- Code style: clear names, small functions, docstrings on public functions. No em dashes in any output text, comments, or docs.

### Proposed project layout

```
adp-ner-forecast/
  README.md
  PROMPTS.md
  requirements.txt
  pyproject.toml            (console_script entry "adp")
  .gitignore                (ignore the data/ dir, __pycache__, .pytest_cache)
  data/                     (download lands here on first run, gitignored, not committed)
  adp_ner/
    __init__.py
    cli.py                  argparse wiring, subcommands, output formatting
    data.py                 download once, extract, parse, validate
    series.py               build the National monthly series and the change column
    models.py               the baseline models, all numpy/pandas
    backtest.py             walk-forward evaluation and metrics
    explain.py              human readable rationale
  tests/
    fixtures/
      mini_history.csv      small hand made long-format CSV we wrote, committed
    conftest.py             tiny pytest fixture that zips mini_history.csv into a tmp dir when a test needs a zip
    test_data.py
    test_series.py
    test_models.py
    test_backtest.py
    test_explain.py
    test_cli.py
```

CLI entry should work both as `python -m adp_ner ...` and as `adp ...` after `pip install -e .`.

---

# Phase 1: Data layer and history view

Goal: I can clone, install, run one command, and see the real historical ADP numbers that match the published series.

### Build

1. `data.py`
   - `fetch(url=DEFAULT_URL, data_dir="data", local_path=None) -> Path`
     - If `local_path` is given, use that file and skip the network entirely.
     - Otherwise look for the zip inside `data_dir`. If it is already there, use it and do not touch the network. The file gets downloaded exactly once, on the first run when `data_dir` does not exist yet.
     - When the zip is not present, create `data_dir`, download the zip, and save it there. Wrap the download in a try/except. On any network or HTTP error, raise a clear error that says what failed and suggests checking the connection or passing `--input` with a local file. Do not leave a half written zip behind if the download fails partway.
   - `extract(zip_path) -> Path` returns the path to the extracted `ADP_NER_history.csv`. Raise a clear error if the file is not a valid zip, or does not contain the expected CSV member.
   - `load_raw(csv_path) -> pd.DataFrame` reads the CSV, parses `date`, validates that all six required columns exist, coerces `NER` and `NER_SA` to numeric, and raises a clear error naming the missing or bad column if validation fails.

2. `series.py`
   - `national_monthly(df) -> pd.DataFrame` filters to `agg_RIS == "National"`, `category == "U.S."`, `timestep == "M"`, sorts by date, drops exact duplicate dates keeping the last, sets a clean monthly index, and adds two columns: `chg` ( `NER.diff()` ) and `chg_sa` ( `NER_SA.diff()` ). The first row's changes are NaN, which is expected.
   - A small helper to select level vs change and SA vs NSA based on flags.

3. `cli.py`
   - Subcommand `history` with options:
     - `--rows N` (default 12) number of most recent rows to show
     - `--measure {level,change,both}` (default both)
     - `--sa / --nsa` (default sa)
     - `--input PATH` use a local zip or csv instead of downloading
     - `--url URL` override the default URL
     - `--refresh` force a fresh download
     - `--json` machine readable output
   - Print a clean aligned table (plain text, no fancy unicode borders needed) with date, level, and change as requested. Show units in thousands of jobs for the change so it reads like the published figure, and label it clearly.

### Tests (test_data.py, test_series.py)

Required cases including edges:

- First run with no `data/` dir downloads once and stores the zip (mock the HTTP call, assert it is called once).
- Second run with the zip already present uses it and does not call the network at all (assert the mocked HTTP call is not invoked).
- Download failure raises a clear, actionable error and does not leave a partial zip behind (mock the HTTP call to raise, assert the error message and that no zip file remains).
- `extract` on a valid zip returns the CSV path (the conftest fixture zips `mini_history.csv` into a tmp dir).
- Edge: `extract` on a non-zip file raises a clear error.
- Edge: `extract` on a zip that does not contain the expected CSV member raises a clear error.
- `load_raw` parses the fixture and produces correct dtypes (datetime date, float NER and NER_SA).
- Edge: CSV missing a required column raises an error that names the column.
- Edge: CSV with a non-numeric value in NER raises a clear error (or coerces and flags, pick one and test it).
- `national_monthly` returns only National U.S. monthly rows, sorted ascending, dates strictly increasing.
- `chg_sa` equals the manual `NER_SA.diff()` on the fixture.
- Edge: duplicate date rows are de-duplicated.
- Edge: a series with a single row yields a NaN change and does not crash.

### Manual verification gate (what I will check)

I will run:

```
pip install -e .
adp history --rows 8
```

and confirm the last row is May 2026 with an SA level near 132,624,000 and a change near +122k, and that the prior months roughly match the reference changes listed above. I will also run `adp history --input <a local zip> ` to confirm the offline path works.

---

# Phase 2: Forecasting engine and evaluation

Goal: I can ask for next month's prediction and see a credible number, and I can see how the model was evaluated against simple baselines.

### Build (models.py, backtest.py)

Forecast target is the month over month change in the SA level, `chg_sa`. The level forecast is just the last known SA level plus the predicted change. The "next print" headline number is the predicted change in thousands.

Implement these models, all with numpy and pandas only. They are deliberately simple, which is the right call for a single seasonally adjusted monthly series:

1. `naive`: next change equals the last observed change.
2. `seasonal_naive`: next change equals the change 12 months earlier.
3. `mean_window`: next change equals the trailing mean of the last `window` changes (default 6).
4. `ewma`: next change equals the pandas exponentially weighted mean of past changes (expose `span`). This is the most expressive of the four, since it leans on recent months more than old ones, and it is easy to explain.

Each model is a small function that takes the change series and returns the next point. Keep them stateless and pure so they are trivial to test.

There is no `auto` model. Running an expanding window backtest over every model on each `forecast` call is wasteful and clutters the CLI routing for no real benefit. Instead, decide the default once during development: run `adp evaluate`, see which model has the lowest backtest RMSE, set that as the hardcoded default for `forecast`, and write down in the README why it was chosen. A sensible starting default is `ewma`, but let the one time evaluation settle it.

Prediction interval (lives in `backtest.py`):
- Build the band from the single chosen model's own one step ahead backtest errors, not from in sample fit. Backtesting one model is cheap, so this stays light. Take the standard deviation of those errors and report point plus or minus z times that std (z default 1.0 for a roughly 68 percent band, configurable via `--interval`). Guarantee low <= point <= high. This is honest about how the model actually does out of sample.

Backtesting (`backtest.py`):

- Walk-forward, expanding window, one step ahead. For each cutoff from some minimum training size to the second to last point, fit on data up to the cutoff and predict the next change, then compare to the actual. **Assert no leakage**: the training slice must never include the point being predicted.
- Metrics per model:
  - MAE, mean absolute error: the average absolute miss (predicted minus actual, sign dropped). It is in the same units as the target, so an MAE of 30 means a typical month is off by about 30k jobs.
  - RMSE, root mean squared error: square each miss, average them, take the square root. Same units as MAE, but squaring penalizes big misses harder, so RMSE is always greater than or equal to MAE and a large gap between the two flags a few months that blew up.
  - Directional accuracy: how often we got the sign of the change right.
  - Lower is better for MAE and RMSE, higher is better for directional accuracy. Return a tidy table.

### CLI

- Subcommand `forecast`:
  - `--model {naive,seasonal_naive,mean,ewma}` default is the model chosen during the one time evaluation (start with `ewma`)
  - `--window N` (mean), `--span N` (ewma), `--interval FLOAT` (band multiplier)
  - data source flags shared with Phase 1 (`--input`, `--url`, `--refresh`)
  - `--json`
  - Output: the target month, the predicted change in thousands, the implied SA level, and the interval. Name the model used.
- Subcommand `evaluate`:
  - Runs the backtest and prints the metrics table for all models, sorted by RMSE. `--json` supported.

### Tests (test_models.py, test_backtest.py)

Required cases including edges:

- `naive` on a hand built series returns the last change exactly.
- `seasonal_naive` returns the value from 12 steps back; edge: series shorter than 13 points raises a clear error.
- `mean_window` equals the manual trailing mean.
- `ewma` matches a direct pandas `.ewm(...).mean()` computation.
- Forecast on a constant level series returns a change of zero (diff of a constant is zero) for every model.
- Forecast on a perfectly linear level trend returns the constant slope as the change for naive, mean, and ewma.
- Interval ordering holds: low <= point <= high on real and synthetic data.
- Backtest produces one prediction per eligible cutoff and the metrics are finite.
- Leakage guard: add an assertion in the backtest loop and a test that deliberately checks the training index max is strictly less than the test index.
- Edge: a series too short to backtest (fewer points than the minimum training window) raises a clear, specific error.
- Edge: NaN handling, the leading NaN from `.diff()` is dropped before any model sees the series and does not poison the result.
- Determinism: running forecast twice on the same input gives identical output.

### Manual verification gate (what I will check)

I will run:

```
adp forecast
adp forecast --model ewma --span 6
adp evaluate
```

and confirm the predicted June 2026 change is a plausible number in the tens of thousands range (not negative hundreds of thousands, not in the millions), the implied level is just above 132.62M, and the evaluate table shows one of the models winning on RMSE with sensible MAE numbers. I want to understand the spread between models and confirm the default model matches the winner.

---

# Phase 3: Explanation, polish, and docs

Goal: clone and run works end to end per the README, the explanation reads like a person wrote it, and the two required docs are real.

### Build (explain.py, finish cli.py)

- Subcommand `explain`:
  - Produces a plain language rationale for the current forecast. It must cover, in prose with a few short supporting lines:
    - which model produced the number, and that it is the default because it won the one time backtest, naming the RMSE it beat
    - the recent trend: the last several monthly changes and their average, and whether momentum is building or fading
    - how the chosen model turns that history into the number. For ewma, name the recent values and note that the most recent months carry the most weight. For mean, name the window and its average. For naive, that it just carries the last change forward. For seasonal_naive, which month a year ago it copied.
    - the seasonality note: state plainly that we do not model month of year effects because the series is already seasonally adjusted, so there is little seasonal signal left to capture
    - a short, traceable walk from the recent history to the point forecast, so the reader can reproduce it by hand
    - the uncertainty: the interval and that its width comes from how far the model's past one step ahead forecasts have missed
    - how this compares to the naive guess
  - `--json` returns the same content as structured fields.
  - The numbers in `explain` must be consistent with what `forecast` prints. Cross-check this in a test.

- Final CLI polish: shared argument handling, consistent `--json` across subcommands, helpful `--help` text, sensible error messages and non-zero exit codes on failure.

### Docs

- `README.md` must contain:
  - what the tool is, in two or three sentences
  - install and run instructions that actually work from a clean clone (`pip install -e .`, then example commands for `history`, `forecast`, `explain`, `evaluate`)
  - the approach and the key tradeoffs (why forecast the SA change rather than the level, why simple baselines with the default picked once by backtest rather than a heavier model or a per run auto select, why SA over NSA, the download once and offline story, the licensing note about not committing data)
  - how forecast accuracy was evaluated and the actual results: paste the real backtest metrics table produced by `adp evaluate`, name the winning model, and state the MAE and RMSE in thousands of jobs and the directional accuracy. Be honest about how hard this series is to beat with a naive baseline.
  - what you would build next with another week (ideas: try a small AR or lagged regression with numpy lstsq and see if it actually beats the baselines on the backtest, bring in the weekly series or the industry and size breakdowns as features, add real seasonal modeling on the NSA series, proper prediction intervals via residual bootstrapping, a small backtest plot, pulling the latest vintage automatically by discovering the newest artifact path)
- `PROMPTS.md`: the real, unsanitized session log. For each AI session record the tool used, the prompts sent, and a short note on what was done with the output (used as is, edited heavily, or rejected and why). Include dead ends and prompts that did not work. Prefer the raw Claude Code transcript over a summary.

### Tests (test_explain.py, test_cli.py)

Required cases including edges:

- `explain` output contains each required section.
- The forecast number quoted inside `explain` equals the number `forecast` returns for the same inputs (consistency test).
- `--json` output for each subcommand parses as valid JSON and has the expected keys.
- End to end smoke test on the committed fixture: `history` then `forecast` then `explain` run without error and produce consistent numbers.
- A golden test pins the forecast on the fixture to a known value so future refactors cannot silently change the math.
- CLI argument errors (bad model name, negative rows) exit non-zero with a readable message.
- Edge: running any subcommand with `--input` pointing at a missing file gives a clear error, not a stack trace.

### Manual verification gate (what I will check)

From a clean clone I will run the README steps verbatim, then `adp explain`, and confirm the explanation matches the forecast and reads naturally. I will run the full test suite and expect it green and offline.

---

## Notes for you, Claude Code

- The artifact URL has a dated path segment (`20260603`). ADP rotates these, so do not assume it is permanent. Keep it as the default but make `--url` and `--input` first class, and rely on the stored `data/` zip once it is downloaded.
- Write the static `mini_history.csv` fixture by hand first in Phase 1 so every later test has something to run against without the network. Do not write a generator script for it. It should be long format with all four agg_RIS values present but tiny, and a National U.S. monthly slice long enough (15 plus points) to exercise seasonal_naive and the backtest. When a test needs a zip, the conftest fixture zips this CSV into a tmp dir on the fly.
- Keep the forecast target decision visible in code and docs: model `chg_sa`, reconstruct the level by addition. Do not forecast the raw level directly with a trend line as the primary method, since the level is strongly trending and that hides the real signal.
- Stop after each phase and tell me what to run to verify. Do not roll phases together.
