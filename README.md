# ADP National Employment Report Tracker and Forecaster

A small command line tool that downloads the ADP National Employment Report,
shows the recent numbers, predicts next month's jobs change, and explains the
prediction in plain words. The forecasting uses only numpy and pandas. The data
is a single monthly series that is already seasonally adjusted, so simple methods
work well and a heavy library is not needed.

## First time setup

You need Python 3.10 or newer, and `curl` on your PATH (it ships with macOS and
most Linux distributions). From a fresh clone:

```bash
python3 -m venv .venv          # create an isolated environment
source .venv/bin/activate       # activate it (do this in every new shell)
pip install -e .                # install the `adp` command
```

That installs the `adp` command into the virtual environment. The system Python
on recent macOS and Linux is "externally managed" (PEP 668) and will refuse a
direct `pip install`, so the virtual environment above is the supported path.

## Running it

With the environment active, run any of the four commands:

```bash
adp history --rows 8          # recent published numbers
adp forecast                  # next month's predicted change and level
adp explain                   # why the forecast is what it is
adp evaluate                  # how the models score against each other
```

The first run downloads the data with `curl` into a `data/` folder (which is
gitignored) and records a timestamp. The ADP report comes out once a month, so
every later run that same calendar month reuses the saved copy and works
offline. The first run of a new month sees the timestamp is stale and downloads
the latest report automatically.

Every command also takes `--json` for machine readable output, plus three flags
for where the data comes from:

- `--input PATH` use a local zip or csv instead of downloading
- `--url URL` use a different download link
- `--refresh` download a fresh copy now, even if this month's data is saved

For example, to run fully offline from a file you already have:

```bash
adp history --input /path/to/ADP_NER_history.zip
```

## What the number means

The data file stores employment **levels** (how many people are employed). The
number ADP reports each month is the **change from last month in the seasonally
adjusted level**, which in code is `NER_SA.diff()`. The headline series is the
rows where `agg_RIS == "National"`, `category == "U.S."`, and `timestep == "M"`.
Every command works on that series.

## Approach and main choices

- **Predict the change, not the level.** The level keeps climbing over time, so a
  line drawn through it always looks good and hides what is really going on. We
  predict the monthly change instead, then add it back to the last known level to
  get the predicted level.
- **Simple models, default chosen once.** There are four models (`naive`,
  `seasonal_naive`, `mean`, `ewma`), all built with numpy and pandas. Instead of
  re-testing them on every run, we tested them once during development, saw which
  one was most accurate, and set that as the default. The results are below.
- **Use the seasonally adjusted numbers.** The `NER_SA` column already removes the
  usual month to month seasonal pattern, so the model does not have to learn it.
- **Download once a month, then work offline.** The data only changes once a
  month, so the saved copy is reused for the rest of the calendar month it was
  fetched in. A timestamp next to the data tracks when it was downloaded; the
  first run of a new month sees that the timestamp is stale and re-downloads
  automatically, while runs within the same month stay fast, offline, and easy
  on ADP's server. The `--refresh` flag forces a fresh download immediately, and
  `--input` and `--url` let you point at a saved file or a new link if the
  download address ever changes.
- **The raw data is not in this repo.** The license file inside the download says
  the data belongs to ADP and cannot be reproduced commercially, so the csv and
  zip are not committed and the `data/` folder is gitignored. The only data in the
  repo is `tests/fixtures/mini_history.csv`, a tiny made up file we wrote by hand
  so the tests can run offline. It is not real ADP data.

## How accuracy was measured, and the results

We test each model by walking through history one month at a time. At every step
the model only sees the past, predicts the next month, and we compare that
prediction to what actually happened. The model never sees the month it is trying
to predict (there is a check in the code that guarantees this). The scores, in
thousands of jobs:

```
model              MAE (k)  RMSE (k)   dir. acc.
naive                 82.6     157.1         93%
ewma                 111.1     221.5         90%
mean                 124.8     261.3         89%
seasonal_naive       203.2     400.3         84%
```

MAE is the average miss. RMSE is like MAE but punishes big misses more.
Directional accuracy is how often we got the up or down direction right.

**`naive` wins** on all three, so it is the default. Being honest: this series is
hard to beat. The monthly change moves almost at random, so just repeating last
month's change is a strong baseline. The `ewma` model is the best of the rest and
the easiest to explain, but it still does not beat `naive` here. Direction
accuracy is high for everyone because the number is positive most months.

You can recreate this table any time with `adp evaluate`.

## Tradeoffs

I deliberately chose to use standard library argparse and pandas to prioritize
system stability, offline reliability, and minimal dependency bloat. A detailed
analysis of these architectural choices, including why heavier ML libraries were
omitted and how the backtest informed the system design, can be found in the
[tradeoff section](TRADEOFFS.md).

## What I would add with more time

- A small regression using `numpy.linalg.lstsq` to see if it beats `naive`.
- The weekly series, or the industry and company size breakdowns, as extra inputs.
- Seasonal modeling on the raw (not adjusted) numbers, to compare.
- Better prediction ranges built by resampling past errors.
- A simple chart of predicted versus actual.
- Auto detecting the newest download link so the default never goes out of date.

## AI prompts

This project was built with help from AI, and the prompts are saved so the work
is reproducible:

- The prompts used with Claude (chat) are in
  [Claude-ADP employment forecasting CLI tool.md](Claude-ADP%20employment%20forecasting%20CLI%20tool.md).
- The prompts used with Claude Code are in [PROMPTS.md](PROMPTS.md).

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests run fully offline. The download is faked in the tests, and the small
hand made `mini_history.csv` file backs the rest. No test ever calls the real
download link.

Changed code is reviewed with the CodeRabbit CLI (`coderabbit review --agent`)
before being committed.
