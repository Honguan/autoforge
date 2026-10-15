"""Isolation Forest on tabular data. task.target (optional) = a column of known anomaly labels used
only for evaluation (1/true/yes/anomaly = anomaly); it is never used for training."""

from __future__ import annotations

import json
from pathlib import Path

POSITIVE = {"1", "true", "yes", "anomaly", "fraud", "fault", "outlier"}


def _pipeline(df, hp, seed):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import IsolationForest
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = df.select_dtypes("number").columns.tolist()
    categorical = [c for c in df.columns if c not in numeric]
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("cat", make_pipeline(SimpleImputer(missing_values="", strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)), categorical)])
    forest = IsolationForest(n_estimators=int(hp.get("n_estimators", 300)), max_samples=hp.get("max_samples", "auto"),
                             contamination=float(hp.get("contamination", 0.02)), random_state=seed, n_jobs=-1)
    return make_pipeline(pre, forest), numeric, categorical


def train(ctx) -> dict:
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn import metrics

    hp = ctx.hyperparameters
    df = pd.read_csv(ctx.data_path)
    labels = None
    if ctx.target and ctx.target in df.columns:
        labels = df.pop(ctx.target).astype(str).str.strip().str.lower().isin(POSITIVE).astype(int).to_numpy()
    df = df.drop(columns=[c for c in hp.get("exclude", []) if c in df.columns])
    model, numeric, categorical = _pipeline(df, hp, ctx.seed)
    df[categorical] = df[categorical].fillna("").astype(str)
    ctx.check_stop()
    model.fit(df)
    ctx.check_stop()
    score = -model.score_samples(df)  # higher = more anomalous
    flagged = (model.predict(df) == -1).astype(int)
    m = {"anomaly_rate": float(flagged.mean())}
    if labels is not None and 0 < labels.sum() < len(labels):
        m |= {"precision": float(metrics.precision_score(labels, flagged, zero_division=0)),
              "recall": float(metrics.recall_score(labels, flagged, zero_division=0)),
              "f1": float(metrics.f1_score(labels, flagged, zero_division=0)),
              "roc_auc": float(metrics.roc_auc_score(labels, score))}
    ctx.log(f"{len(df)} rows, flagged {flagged.sum()} ({m['anomaly_rate']:.2%})"
            + (f", f1 vs labels {m['f1']:.3f}" if "f1" in m else " (no labels: unsupervised)"))
    out = Path(ctx.output_dir)
    joblib.dump({"model": model, "numeric": numeric, "categorical": categorical,
                 "threshold": float(np.quantile(score, 1 - float(hp.get("contamination", 0.02))))}, out / "model.joblib")
    feats = [{"name": c, "dtype": "numeric"} for c in numeric] + [{"name": c, "dtype": "categorical"} for c in categorical]
    (out / "io.json").write_text(json.dumps({"features": feats, "labels": ["anomaly", "normal"]}))
    return {"metrics": m, "model_path": str(out / "model.joblib")}


def load_model(model_dir):
    import joblib

    return joblib.load(Path(model_dir) / "model.joblib")


def predict(m, rows: list[dict]) -> list[dict]:
    import pandas as pd

    cols = m["numeric"] + m["categorical"]
    df = pd.DataFrame([{c: r.get(c) for c in cols} for r in rows], columns=cols)
    df[m["numeric"]] = df[m["numeric"]].apply(pd.to_numeric, errors="coerce")
    df[m["categorical"]] = df[m["categorical"]].astype(object).where(df[m["categorical"]].notna(), "").astype(str)
    scores = -m["model"].score_samples(df)
    return [{"prediction": "anomaly" if s >= m["threshold"] else "normal", "score": round(float(s), 6)} for s in scores]
