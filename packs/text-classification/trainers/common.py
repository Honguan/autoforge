"""Shared helpers: load the CSV, split, metrics, io.json."""

from __future__ import annotations

import json
from pathlib import Path


def load_split(ctx, text_column: str, seed: int):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(ctx.data_path)
    if text_column not in df.columns:
        raise KeyError(text_column)
    df = df[[text_column, ctx.target]].dropna()
    texts, labels = df[text_column].astype(str).tolist(), df[ctx.target].astype(str).tolist()
    counts = pd.Series(labels).value_counts()
    stratify = labels if counts.min() >= 2 else None
    return train_test_split(texts, labels, test_size=0.2, random_state=seed, stratify=stratify)


def scores(y_true, y_pred) -> dict:
    from sklearn import metrics

    return {"accuracy": float(metrics.accuracy_score(y_true, y_pred)),
            "f1_macro": float(metrics.f1_score(y_true, y_pred, average="macro"))}


def write_io(out_dir: Path, labels: list[str]) -> None:
    io = {"features": [{"name": "text", "dtype": "text", "description": "the text to classify"}],
          "labels": sorted(labels)}
    (Path(out_dir) / "io.json").write_text(json.dumps(io, ensure_ascii=False))
