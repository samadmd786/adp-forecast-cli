# ADP National Employment Report Tracker and Forecaster

A small command line tool that downloads the ADP National Employment Report
history, shows the published headline numbers, forecasts next month's headline
change, and explains in plain language how that forecast was made. Forecasting
uses only numpy and pandas, by design: the headline series is a single, already
seasonally adjusted monthly line, and simple baselines are the right tool for it.

## Install and run

From a clean clone (Python 3.10 or newer):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then try the four commands. The first run downloads the data once into a
gitignored `data/` directory; later runs reuse it offline.

```bash
adp history --rows 8          # recent published numbers
adp forecast                  # next month's predicted change and level
adp explain                   # why the forecast is what it is
adp evaluate                  # backtest metrics for every model
```

Every command accepts `--json` for machine readable output, and a shared set of
data source flags: `--input PATH` (use a local zip or csv instead of the
network), `--url URL` (override the download URL), and `--refresh` (force a fresh
download). For example, to run fully offline against a local file:

```bash
adp history --input /path/to/ADP_NER_history.zip
```

## What the headline number is

The CSV stores employment **levels**. The figure ADP publishes each month is the
**month over month change in the seasonally adjusted level**, that is
`NER_SA.diff()`. The headline series is the rows where `agg_RIS == "National"`,
`category == "U.S."`, and `timestep == "M"`. That is what every command works on.

## Approach and key tradeoffs

- **Forecast the SA change, not the level.** The level is strongly trending, so a
  trend line on the level would look accurate while hiding the real signal. We
  forecast the change (`chg_sa`) and reconstruct the level by adding the predicted
  change to the last known level.
- **Simple baselines, default picked once by backtest.** We ship four models
  (`naive`, `seasonal_naive`, `mean`, `ewma`), all numpy and pandas. Rather than
  re-running an auto select on every call, we ran the backtest once during
  development, saw which model had the lowest RMSE, and hardcoded it as the
  default for `forecast`. See the results below.
- **Seasonally adjusted over not adjusted.** `NER_SA` already removes month of
  year effects, so the model does not need to learn seasonality. Any seasonal
  component estimated on the SA change comes out small, which is a sanity check,
  not a bug.
- **Download once, then offline.** The archive is fetched a single time into
  `data/` and reused. `--input` and `--url` are first class so a rotated artifact
  path or an air gapped machine is no problem.
- **Licensing.** `license.txt` in the archive states the data is ADP's property
  and may not be reproduced commercially, so the raw CSV and zip are **not
  committed**. `data/` is gitignored. The only data in the repo is
  `tests/fixtures/mini_history.csv`, a tiny long format file we wrote by hand (not
  real ADP data) so the tests run offline.

## How accuracy was evaluated, and the results

Evaluation is a walk-forward, expanding window, one step ahead backtest: for each
month we train on all prior data, predict the next change, and compare to the
actual. No training slice ever includes the point being predicted (there is an
assertion in the loop guarding against leakage). Metrics, in thousands of jobs:

```
model              MAE (k)  RMSE (k)   dir. acc.
naive                 82.6     157.1         93%
ewma                 111.1     221.5         90%
mean                 124.8     261.3         89%
seasonal_naive       203.2     400.3         84%
```

**`naive` wins** on all three metrics, so it is the hardcoded default for
`forecast`. This is honest about how hard the series is to beat: an already
seasonally adjusted monthly change series behaves close to a random walk, so
carrying the last change forward is tough competition for anything more elaborate.
The exponentially weighted model is the strongest of the alternatives and is the
most explainable, but it does not beat the naive baseline here. Directional
accuracy is high across the board because the series spends most months positive.

(Regenerate this table any time with `adp evaluate`.)

## Tradeoffs

I deliberately chose to use standard library argparse and pandas to prioritize
system stability, offline reliability, and minimal dependency bloat. A detailed
analysis of these architectural choices, including why heavier ML libraries were
omitted and how the backtest informed the system design, can be found in the
[tradeoff section](TRADEOFFS.md).

## What I would build next with another week

- A small AR or lagged regression with `numpy.linalg.lstsq`, to test whether any
  linear structure actually beats the naive baseline on the backtest.
- Bring in the weekly series, or the industry and establishment size breakdowns,
  as features.
- Real seasonal modeling on the NSA series, for comparison against the SA path.
- Proper prediction intervals via residual bootstrapping instead of a single
  error standard deviation.
- A small backtest plot of predicted versus actual.
- Auto discovery of the newest artifact path so the default URL never goes stale.

## Development

```bash
pip install -e ".[dev]"
pytest
```

Tests are fully offline. The network is mocked for the download tests, and the
hand made `mini_history.csv` fixture (zipped on the fly when a test needs a zip)
backs everything else. No test hits the real URL.
