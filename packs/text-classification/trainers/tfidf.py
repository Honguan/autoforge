"""TF-IDF + logistic regression. analyzer=char_wb works for languages without spaces (Chinese, Japanese)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import load_split, scores, write_io  # noqa: E402


def train(ctx) -> dict:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    hp = ctx.hyperparameters
    x_train, x_val, y_train, y_val = load_split(ctx, hp.get("text_column", "text"), ctx.seed)
    ctx.log(f"{len(x_train)} train / {len(x_val)} validation texts, {len(set(y_train))} labels")
    ctx.check_stop()
    model = make_pipeline(
        TfidfVectorizer(analyzer=hp.get("analyzer", "char_wb"), ngram_range=(1, int(hp.get("ngram_max", 4))),
                        min_df=int(hp.get("min_df", 2)) if len(x_train) > 50 else 1,
                        max_features=int(hp.get("max_features", 200000)), sublinear_tf=True),
        LogisticRegression(C=float(hp.get("C", 4.0)), max_iter=2000, class_weight=hp.get("class_weight")),
    )
    model.fit(x_train, y_train)
    ctx.check_stop()
    metrics = scores(y_val, model.predict(x_val))
    out = Path(ctx.output_dir)
    joblib.dump(model, out / "model.joblib")
    write_io(out, sorted(set(y_train)))
    return {"metrics": metrics, "model_path": str(out / "model.joblib")}


def load_model(model_dir):
    import joblib

    return joblib.load(Path(model_dir) / "model.joblib")


def predict(model, rows: list[dict]) -> list[dict]:
    texts = [str(r.get("text", "")) for r in rows]
    classes = [str(c) for c in model.classes_]
    return [{"prediction": str(p), "probabilities": {c: round(float(x), 6) for c, x in zip(classes, probs)}}
            for p, probs in zip(model.predict(texts), model.predict_proba(texts))]
