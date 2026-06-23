# Tradeoffs

An honest look at the architecture choices in this project, what they buy, and
what a different choice might have bought instead. The three core tools (argparse,
numpy, pandas) were picked partly because I have used them before and am
comfortable with the syntax. That familiarity is a real advantage for getting
something correct and shipped, but it is worth being clear about where a
different tool could have done better.

## Tooling

### argparse for the CLI

**Pros**

- In the standard library, so no dependency and nothing to install.
- Well understood, stable, and documented everywhere.
- Subcommands, mutually exclusive groups (`--sa / --nsa`), choices, and types are
  all built in, which covers everything this CLI needs.

**Cons**

- The wiring is verbose. Each subcommand repeats `--window`, `--span`, `--json`,
  and the source flags, so there is boilerplate even after factoring out
  `_add_source_args`.
- No type inference. Argument types and help text are written by hand rather than
  derived from a function signature.
- Error messages and help formatting are fixed unless you subclass.

**A better way might be:** `click` or `typer`. `typer` derives the CLI from type
annotated Python functions, which would cut most of the boilerplate and give
nicer help and shell completion for free. The cost is a third party dependency and
a bit of magic that is less transparent than argparse. For a small tool, argparse
is defensible; for a larger CLI, typer would likely pay off.

### numpy and pandas for the forecasting

**Pros**

- Forcing the models to be plain numpy and pandas keeps them tiny, pure, and
  trivial to unit test. Each model is one line of intent.
- No heavy dependency, no model serialization, no hidden state. The whole forecast
  path is reproducible by hand, which is exactly what the `explain` command needs.
- For a single, already seasonally adjusted monthly series, simple baselines are
  genuinely competitive. The backtest confirmed the naive model wins, so a heavier
  library would not have improved accuracy here.

**Cons**

- The ceiling is low. There is no principled way to add ARIMA, state space models,
  or proper confidence intervals without reimplementing statistics by hand.
- The prediction interval is a single error standard deviation times z. It is
  honest about past misses but assumes roughly symmetric, roughly constant
  variance errors, which is cruder than a model based interval.
- Some operations (the EWMA recurrence, the seasonal lag) lean on pandas
  conventions that a reader has to know to verify.

**A better way might be:** `statsmodels` for real ARIMA or exponential smoothing
with proper intervals, or `scikit-learn` for a lagged regression with
cross-validation. The plan deliberately ruled these out, and given that naive wins
the backtest, that was the right call for this dataset. On a noisier or
multivariate problem, a stats library would be worth the dependency.

## Architecture

### Module split: data, series, models, backtest, explain, cli

**Pros**

- Each module has one job and a small surface, so the boundaries are easy to test
  in isolation (one test file per module).
- The CLI is a thin layer over library functions, so the same logic is callable
  from Python, not just the terminal.
- Swapping the data source, adding a model, or changing the output format each
  touch one place.

**Cons**

- For a project this size the split is arguably more structure than strictly
  needed. A single module would also have worked and been shorter to read.
- A few cross module couplings exist (`backtest` imports `models`, `explain`
  imports both `backtest` and `models`), so the dependency direction has to be
  kept straight to avoid import cycles.

### make_forecast as the single source of truth

**Pros**

- `forecast` and `explain` both call `make_forecast`, so the number in the
  explanation can never drift from the number in the forecast. A test pins this.
- Adding a new consumer (a web endpoint, a report) reuses the same function.

**Cons**

- It couples the point forecast and the interval into one call, so a caller who
  wants only the point still triggers a backtest for the interval. That is cheap
  here but would matter on a very long series.

### Forecasting the SA change, reconstructing the level

**Pros**

- Models the quantity that is actually published and that carries the signal,
  instead of a strongly trending level that any line fits well.
- The level is recovered by simple addition, which is transparent and exact.

**Cons**

- Errors in the change accumulate if the tool were ever extended to multi step
  forecasts. For the one step ahead case this is not an issue.

### Default model chosen once by backtest, not per run

**Pros**

- The CLI stays simple and fast: no auto select runs on every `forecast` call.
- The choice is explained and reproducible (`adp evaluate` regenerates the table).

**Cons**

- The default is a snapshot. If ADP revises the series or a new vintage shifts the
  ranking, the hardcoded default could go stale until someone re-runs `evaluate`
  and updates it. A periodic check, or a documented cadence, would mitigate this.

### Download once, cache in a gitignored data dir

**Pros**

- One network call ever, then fully offline. Friendly to repeated use and to CI.
- Keeps copyrighted data out of the repo, which the license requires.
- `--input` and `--url` make a rotated artifact path or an air gapped run trivial.

**Cons**

- The cache has no freshness check. Once the zip is present it is reused until
  `--refresh`, so a stale local copy can hide a newer release. A timestamp or
  checksum check would make this safer.

### Testing on a hand made fixture

**Pros**

- Tests run offline and commit no copyrighted data. The network is mocked, and the
  tiny fixture exercises parsing, the series build, every model, the backtest, and
  the end to end CLI.
- A golden test pins the forecast so a refactor cannot silently change the math.

**Cons**

- The fixture's numbers are tiny and synthetic, so reading the test output does
  not resemble the real headline figures, and the fixture cannot catch issues that
  only appear at real scale or with real volatility (such as the 2020 outliers).
  The real data sanity checks have to be run manually.

## Summary

The familiar stack (argparse, numpy, pandas) was the right call for a small,
explainable tool where the data itself rewards simple methods. The main places I
would revisit with more time are the CLI ergonomics (typer), a model based
prediction interval, and cache freshness. None of these are correctness problems
today; they are ceilings I chose not to raise because the dataset did not demand
it.
