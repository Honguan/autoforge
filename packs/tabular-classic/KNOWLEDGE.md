# tabular-classic — knowledge for AI planners

## When to use
- The data is a table (CSV) where each row is one example and one column is the target.
- You need a quick, reliable baseline before trying anything heavier.
- No GPU is available, or the dataset is under ~1M rows.

## When NOT to use
- Text, images, audio or sequences as the main input. Use a modality-specific pack.
- Millions of rows with many features: prefer a gradient-boosting pack (planned).

## Choosing a recipe
| Situation | Recipe |
|---|---|
| Classification, first baseline, need explainability | `logistic_regression` |
| Classification or regression, best accuracy here | `random_forest` |

Always train `logistic_regression` first for classification; if `random_forest`
does not beat it by a clear margin, prefer the simpler model.

## Data requirements
- `target` must name an existing column. Classification targets may be strings or integers.
- Numeric columns are median-imputed and standardized; other columns are one-hot encoded.
- High-cardinality ID-like columns (unique per row) should be dropped before training.

## Metrics
- Classification: `f1_macro` for imbalanced classes, otherwise `accuracy`.
- Regression: `rmse` (same unit as the target), plus `r2`.

## Tuning guide
- Logistic regression overfits rarely. If accuracy is low, the problem is non-linear:
  switch to `random_forest`, don't tune `C` for long.
- Imbalanced classes (minority < 20%): set `class_weight: balanced`.
- Random forest overfitting (train >> validation score): raise `min_samples_leaf` (5–20)
  or set `max_depth` (6–12).
- More `n_estimators` rarely hurts accuracy, only time; 300 is a good default.

## Common failures
| Symptom | Cause | Fix |
|---|---|---|
| `ValueError: ... stratify` | a class has only 1 example | merge or drop rare classes |
| Memory error | one-hot on a high-cardinality column | drop or bucket that column |
| Perfect validation score | target leaked into a feature | find and drop the leaking column |
