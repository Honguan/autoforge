# timeseries-forecast — knowledge for AI planners

## When to use
- Predict the next value(s) of numeric series measured at regular intervals: daily sales, hourly
  electricity load, weekly demand, website traffic.
- Data: CSV with a time column (`time_column`, default `date`) and the value column as `task.target`.
  Many series in one file (long format): add a series column (`series_column`, default `series_id`),
  one row per series and time step, and use `global_lag_gbm`:

  ```
  date,series_id,sales
  2025-01-01,store_1,120
  2025-01-01,store_2,35
  ```

## When NOT to use
- Irregular event logs (aggregate them to a regular interval first).
- Classification of whole series, or anomaly flags → `anomaly-detection`.

## Recipes
| Situation | Recipe |
|---|---|
| one series | `lag_gbm` |
| many related series (stores, products, meters), including short ones | `global_lag_gbm` |

`lag_gbm`: features = the last `lags` values + rolling mean/std over `windows`; HistGradientBoosting.
`global_lag_gbm`: the same features, but every series is divided by its own mean absolute value so one
shared model learns patterns across series of different sizes (the level is passed as an extra feature).
Each series is split chronologically on its own; series too short for `lags` + the test part are left out
of evaluation (listed in the log). Multi-step forecasts are recursive (each prediction feeds the next step).

## Evaluation (read this before reporting results)
- The split is chronological: the last `test_fraction` of the series is held out. Never shuffled.
- `naive_mae` is the error of "next value = last value". **`skill_vs_naive` > 0 means the model beats it.**
  If it is ≤ 0, report that the naive forecast is as good and the model adds nothing.
- `mape` is skipped for zero values; prefer `mae` (same unit as the data) for HPO (minimize).
- `global_lag_gbm` also reports `wape` (total absolute error / total actual, comparable across data sets)
  and `n_series`. Its `mae` is pooled over all series, so large series dominate; use `wape` when sizes differ a lot.

## Tuning guide
- `lags` should cover at least one full season: daily data with weekly pattern → ≥ 7 (default 14);
  hourly data with daily pattern → ≥ 24 (try 48).
- `windows`: a short and a seasonal window, e.g. `[3, 7]` daily, `[6, 24]` hourly.
- Strong trend (values keep growing beyond the training range): tree models cannot extrapolate; forecast
  differences instead (planned option) or keep horizons short.

## Common failures
| Symptom | Cause | Fix |
|---|---|---|
| `series too short` | not enough points for `lags` | lower `lags`, add data |
| skill_vs_naive ≤ 0 | series is mostly noise / random walk | report naive; add drivers (planned) |
| flat forecast far ahead | recursive forecasts regress to the mean | forecast fewer steps |
| `series column ... not in CSV` | wrong column name for global_lag_gbm | set `series_column` |
| global model worse than per-series | series behave differently | train `lag_gbm` per group, or split the CSV |

## Serving
`{"horizon": 7}` continues from the end of the training data; `{"history": [...recent values...], "horizon": 7}`
forecasts after new data.
Global model: `{"series": "store_1", "horizon": 7}` continues a known series; `{"history": [...], "horizon": 7}`
forecasts a new series (at least max(lags, windows) values).
