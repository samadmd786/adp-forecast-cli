# ADP National Employment Report Tracker and Forecaster

A small command line tool that downloads the ADP National Employment Report,
shows the recent numbers, predicts next month's jobs change, and explains the
prediction in plain words. The forecasting uses only numpy and pandas. The data
is a single monthly series that is already seasonally adjusted, so simple methods
work well and a heavy library is not needed.

## Install and run

You need Python 3.10 or newer. From a fresh clone:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then try the four commands. The first run downloads the data once into a `data/`
folder (which is gitignored). After that, every run reuses the saved copy and
works offline.

```bash
adp history --rows 8          # recent published numbers
adp forecast                  # next month's predicted change and level
adp explain                   # why the forecast is what it is
adp evaluate                  # how the models score against each other
```

Every command also takes `--json` for machine readable output, plus three flags
for where the data comes from:

- `--input PATH` use a local zip or csv instead of downloading
- `--url URL` use a different download link
- `--refresh` download a fresh copy even if one is saved

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
- **Download once, then work offline.** The data only changes once a month, so
  downloading it again on every command is a waste. It makes the tool slow and
  puts needless load on ADP's server. Saving one copy and reusing it is faster,
  works without a connection, and is kinder to the host. The `--input` and `--url`
  flags also let you point at a saved file or a new link if the download address
  ever changes.
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
