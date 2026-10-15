"""Lag/rolling features + HistGradientBoostingRegressor, evaluated on the last part of the series
(chronological split, never shuffled) against the naive forecast (next value = last value)."""

from __future__ import annotations

import json
import math
from pathlib import Path


def _features(values: list[float], lags: int, windows: list[int]) -> list[float]:
    """Features for predicting the value right after `values` (needs len(values) >= max(lags, windows))."""
    feats = [values[-k] for k in range(1, lags + 1)]
    for w in windows:
        tail = values[-w:]
        mean = sum(tail) / w
        feats += [mean, math.sqrt(sum((x - mean) ** 2 for x in tail) / w)]
    return feats


def _xy(series, lags, windows):
    need = max(lags, *windows)
    xs = [_features(series[:t], lags, windows) for t in range(need, len(series))]
    return xs, series[need:]


def train(ctx) -> dict:
    import joblib
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor

    hp = ctx.hyperparameters
    lags, windows = int(hp.get("lags", 14)), [int(w) for w in hp.get("windows", [3, 7])]
    df = pd.read_csv(ctx.data_path)
    tcol = hp.get("time_column", "date")
    if tcol in df.columns:
        df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
        df = df.sort_values(tcol)
    series = pd.to_numeric(df[ctx.target], errors="coerce").interpolate(limit_direction="both").tolist()
    need = max(lags, *windows)
    n_test = max(1, int(len(series) * float(hp.get("test_fraction", 0.2))))
    if len(series) - n_test <= need + 10:
        raise ValueError(f"series too short: {len(series)} points for lags={lags}; use fewer lags or more data")

    x, y = _xy(series, lags, windows)
    split = len(series) - n_test - need
    model = HistGradientBoostingRegressor(learning_rate=float(hp.get("learning_rate", 0.05)),
                                          max_iter=int(hp.get("max_iter", 300)),
                                          max_leaf_nodes=int(hp.get("max_leaf_nodes", 31)), random_state=ctx.seed)
    ctx.check_stop()
    model.fit(x[:split], y[:split])
    ctx.check_stop()
    pred = model.predict(x[split:]).tolist()
    actual = y[split:]
    naive = series[len(series) - n_test - 1:-1]  # one-step naive forecast
    mae = sum(abs(a - p) for a, p in zip(actual, pred)) / len(actual)
    naive_mae = sum(abs(a - p) for a, p in zip(actual, naive)) / len(actual)
    rmse = math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, pred)) / len(actual))
    nonzero = [(a, p) for a, p in zip(actual, pred) if a != 0]
    mape = sum(abs((a - p) / a) for a, p in nonzero) / len(nonzero) if nonzero else float("nan")
    for i, (a, p) in enumerate(zip(actual, pred)):
        ctx.log_metric("abs_error", abs(a - p), step=i)
    ctx.log(f"{len(series)} points, last {n_test} held out; MAE {mae:.4g} vs naive {naive_mae:.4g}")

    model.fit(x, y)  # refit on everything for serving
    out = Path(ctx.output_dir)
    joblib.dump({"model": model, "lags": lags, "windows": windows, "tail": series[-need:]}, out / "model.joblib")
    (out / "io.json").write_text(json.dumps({"features": [
        {"name": "history", "dtype": "any", "required": False,
         "description": f"recent values, oldest first (>= {need}); omit to continue from the training data"},
        {"name": "horizon", "dtype": "numeric", "required": False, "description": "steps to forecast (default 1)"}]}))
    metrics = {"mae": mae, "rmse": rmse, "naive_mae": naive_mae, "skill_vs_naive": 1 - mae / naive_mae if naive_mae else 0.0}
    if not math.isnan(mape):
        metrics["mape"] = mape
    return {"metrics": metrics, "model_path": str(out / "model.joblib")}


def load_model(model_dir):
    import joblib

    return joblib.load(Path(model_dir) / "model.joblib")


def predict(m, rows: list[dict]) -> list[dict]:
    need = max(m["lags"], *m["windows"])
    out = []
    for row in rows:
        history = [float(v) for v in (row.get("history") or m["tail"])]
        if len(history) < need:
            raise ValueError(f"history needs at least {need} values, got {len(history)}")
        values, forecast = list(history), []
        for _ in range(int(row.get("horizon") or 1)):  # recursive multi-step forecast
            nxt = float(m["model"].predict([_features(values, m["lags"], m["windows"])])[0])
            forecast.append(nxt)
            values.append(nxt)
        out.append({"prediction": forecast})
    return out
