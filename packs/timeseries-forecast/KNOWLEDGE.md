# timeseries-forecast — knowledge for AI planners

## When to use
- Predict the next value(s) of one numeric series measured at regular intervals: daily sales, hourly
  electricity load, weekly demand, website traffic.
- Data: CSV with a time column (`time_column`, default `date`) and the value column as `task.target`.

## When NOT to use
- Irregular event logs (aggregate them to a regular interval first).
- Hundreds of related series to forecast together (a global model is planned).
- Classification of whole series, or anomaly flags → `anomaly-detection`.

## Recipe
`lag_gbm`: features = the last `lags` values + rolling mean/std over `windows`; HistGradientBoosting.
Multi-step forecasts are recursive (each prediction feeds the next step).

## Evaluation (read this before reporting results)
- The split is chronological: the last `test_fraction` of the series is held out. Never shuffled.
- `naive_mae` is the error of "next value = last value". **`skill_vs_naive` > 0 means the model beats it.**
  If it is ≤ 0, report that the naive forecast is as good and the model adds nothing.
- `mape` is skipped for zero values; prefer `mae` (same unit as the data) for HPO (minimize).

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

## Serving
`{"horizon": 7}` continues from the end of the training data; `{"history": [...recent values...], "horizon": 7}`
forecasts after new data.
