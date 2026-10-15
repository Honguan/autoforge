# anomaly-detection — knowledge for AI planners

## When to use
- Flag unusual rows when you have **no (or very few) labelled anomalies**: suspicious transactions, sensor
  faults, odd log records, data-quality outliers.
- Data: CSV, one row per event. Numeric and categorical columns are used; ID/time columns should be listed
  in `exclude`.
- Optional `task.target`: a column of known labels (1/true/yes/anomaly/fraud) used **only to evaluate**.

## When NOT to use
- Many labelled anomalies (hundreds+) → train a classifier (`tabular-classic`, `class_weight: balanced`):
  supervised beats unsupervised when labels exist.
- Anomalies in a time series' shape → forecast it (`timeseries-forecast`) and flag large errors.

## Recipe
`isolation_forest`: rows that are isolated with few random splits score high. `contamination` = expected
share of anomalies; it sets the decision threshold.

## Metrics
- Always: `anomaly_rate` (≈ contamination).
- With labels: `precision`, `recall`, `f1`, `roc_auc` (threshold-free; best single number for HPO, maximize).
- Without labels there is nothing to optimise: show the user the top-scored rows and ask whether they are real.

## Tuning guide
- `contamination`: set to the true anomaly rate if known (e.g. 0.01 for 1 % fraud). Too high → many false alarms.
- `exclude` identifiers (customer_id, timestamp): unique values look "anomalous" and add noise.
- Scale does not matter (features are standardised); skewed amounts often work better log-transformed upstream.

## Common failures
| Symptom | Cause | Fix |
|---|---|---|
| everything with a new category flagged | high-cardinality ID column used | add it to `exclude` |
| roc_auc ≈ 0.5 | anomalies are not unusual in these features | add features (ratios, time since last event) |
| f1 low but roc_auc high | threshold wrong | adjust `contamination` |

## Serving
Each row returns `{"prediction": "anomaly"|"normal", "score": ...}` (higher score = more unusual).
