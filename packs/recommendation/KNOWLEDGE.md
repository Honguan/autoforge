# recommendation — knowledge for AI planners

## When to use
- "What should this user see/buy/watch next?" from a log of who interacted with what: purchases, clicks,
  plays, views, ratings. Also "items similar to this one" (`{"item": id}`).
- Most users have several interactions (median ≥ 3) and items repeat across users.

## When NOT to use
- Nearly every user is new or has one interaction → popularity / content-based rules (use item attributes).
- You must predict an exact star rating → regression with `tabular-classic` on user/item features.
- Recommendations must use rich attributes (text, price, category) or the session order → not supported yet.

## Data format
One CSV (`.tsv` also works), one row per interaction; duplicates of a (user, item) pair are summed:
```
user_id,item_id,clicks,timestamp
u17,sku-204,3,1716200000
u17,sku-551,1,1716203600
u42,sku-204,1,2024-05-20 10:00
```
- `user_col` / `item_col` (defaults `user_id` / `item_id`). If missing, common names are detected:
  user, userId, customer_id, client_id / item, itemId, product_id, movieId, sku. Otherwise a clear error lists the columns.
- `weight_col` (rating, count, clicks; `task.target` also sets it). Null = detect `rating`/`count`/`clicks`…,
  else 1 per row; `"none"` forces 1 per row. Rows with weight ≤ 0 are dropped (negative feedback is not used).
- `time_col` (null = detect `timestamp`/`date`/`ts`…; epoch numbers or date strings).
- Validation: per user with ≥ 2 interactions, the **last one by time** is held out (random one without a
  time column). Users with 1 interaction stay in training only. The saved model is **refit on all data**.

## Recipe(s)
| recipe | how | pick when |
|---|---|---|
| `implicit_als` | Hu/Koren/Volinsky confidence-weighted ALS, confidence = 1 + alpha·weight | default; latent tastes, better coverage |
| `item_knn` | cosine item-item similarity with shrinkage, top `neighbors` per item | fast, explainable baseline, small/sparse logs |
Both are pure numpy/scipy on CPU. ALS checkpoints every iteration (`run retry` resumes); item_knn is one pass.

## Metrics
- `recall_at_k` (hit rate of the held-out item in the top `k`) — **HPO metric, maximize**.
- `ndcg_at_k` (rewards higher ranks), `coverage` (share of the catalog ever recommended).
- `popularity_recall_at_k`: the "most popular items" baseline. If the model does not beat it, personalisation
  is not helping (too little data per user) — say so to the user.
- ALS curves per iteration: `train_loss`, `recall_at_k`, `ndcg_at_k`; the best iteration is kept.

## Tuning guide
- `k`: what the UI shows (5–20). Compare runs only at the same `k`.
- ALS `factors` 32–128 (more data → more factors), `regularization` 0.01–1 (raise if recall drops with
  more iterations), `alpha` 1–40 for counts (log-scale heavy counts upstream), `iterations` 10–20.
- `run adjust --set regularization=...` changes ALS live between iterations.
- kNN `neighbors` 20–200, `shrinkage` 0–50 (higher = trust co-occurrence of few users less).
- Speed: ALS is a Python loop over users/items × iterations — ~1M interactions takes minutes per iteration at 64 factors.

## Common failures
| Symptom | Cause | Fix |
|---|---|---|
| `no user column found` / `item column ... not found` | unusual column names | set `user_col` / `item_col` |
| `no user has 2+ interactions` | log too sparse to evaluate | collect more history, or aggregate users |
| recall ≈ popularity_recall | few interactions per user / weights dominated by outliers | more data, item_knn, lower `alpha`, cap weights |
| recall falls after a few iterations | overfitting | raise `regularization`, fewer `factors` (best iteration is kept anyway) |
| coverage very low | only popular items recommended | lower `alpha`, more `factors`, try item_knn |

## Serving
- Recommend: `{"user": "u17", "k": 5, "exclude_seen": true}` → `{"prediction": ["sku-9", ...], "scores": [...],
  "cold_start": false}`. `k` defaults to the training `k`; `exclude_seen` defaults to true.
- Unknown user → the most popular items with `"cold_start": true` (scores = interaction counts).
- Similar items: `{"item": "sku-204", "k": 5}` → `{"prediction": [...], "scores": [...], "similar_to": "sku-204"}`
  (cosine of ALS item factors, or the kNN similarity). Ids are matched as strings, so 17 and "17" are the same user.
