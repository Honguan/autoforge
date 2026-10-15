"""scikit-learn trainers for the tabular-classic pack.

Trainer contract v1: train(ctx: TrainContext) -> {"metrics": {...}, "model_path": str}
See autoforge/engine/context.py. Dict-style ctx (ctx["data_path"]) is also accepted.
"""

from __future__ import annotations

from pathlib import Path


def _log(ctx, message: str) -> None:
    if hasattr(ctx, "log"):
        ctx.log(message)


def _check_stop(ctx) -> None:
    if hasattr(ctx, "check_stop"):
        ctx.check_stop()


def _prepare(ctx):
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    df = pd.read_csv(ctx["data_path"])
    _log(ctx, f"loaded {len(df)} rows, {df.shape[1]} columns")
    y = df.pop(ctx["target"])
    numeric = df.select_dtypes("number").columns.tolist()
    categorical = [c for c in df.columns if c not in numeric]
    # Categorical values as strings with "" for missing: identical at predict time and ONNX-exportable.
    df[categorical] = df[categorical].fillna("").astype(str)
    pre = ColumnTransformer(
        [
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
            ("cat", make_pipeline(SimpleImputer(missing_values="", strategy="most_frequent"),
                                  OneHotEncoder(handle_unknown="ignore")), categorical),
        ]
    )
    is_clf = ctx["task_type"] == "tabular-classification"
    split = train_test_split(
        df, y, test_size=0.2, random_state=ctx.get("seed", 0), stratify=y if is_clf else None
    )
    return pre, split, is_clf


def _fit_and_score(ctx, pre, estimator, split, is_clf: bool) -> dict:
    import joblib
    from sklearn import metrics
    from sklearn.pipeline import make_pipeline

    x_train, x_val, y_train, y_val = split
    _check_stop(ctx)
    _log(ctx, f"training {type(estimator).__name__} on {len(x_train)} rows")
    model = make_pipeline(pre, estimator).fit(x_train, y_train)
    _check_stop(ctx)
    pred = model.predict(x_val)
    if is_clf:
        scores = {
            "accuracy": metrics.accuracy_score(y_val, pred),
            "f1_macro": metrics.f1_score(y_val, pred, average="macro"),
        }
    else:
        scores = {
            "rmse": metrics.root_mean_squared_error(y_val, pred),
            "mae": metrics.mean_absolute_error(y_val, pred),
            "r2": metrics.r2_score(y_val, pred),
        }
    out = Path(ctx["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "model.joblib"
    joblib.dump(model, model_path)
    return {"metrics": {k: float(v) for k, v in scores.items()}, "model_path": str(model_path)}


def train_logistic_regression(ctx) -> dict:
    from sklearn.linear_model import LogisticRegression

    hp = {"C": 1.0, "max_iter": 1000, "class_weight": None} | ctx.get("hyperparameters", {})
    pre, split, is_clf = _prepare(ctx)
    if not is_clf:
        raise ValueError("logistic_regression only supports tabular-classification")
    return _fit_and_score(ctx, pre, LogisticRegression(**hp), split, is_clf)


def train_random_forest(ctx) -> dict:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    hp = {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1} | ctx.get("hyperparameters", {})
    pre, split, is_clf = _prepare(ctx)
    if is_clf:
        est = RandomForestClassifier(random_state=ctx.get("seed", 0), n_jobs=-1, **hp)
    else:
        hp.pop("class_weight", None)
        est = RandomForestRegressor(random_state=ctx.get("seed", 0), n_jobs=-1, **hp)
    return _fit_and_score(ctx, pre, est, split, is_clf)


# ---------------------------------------------------------------- serving (predictor contract)


def _py(v):
    """numpy scalars -> plain Python, so results serialise to JSON."""
    return v.item() if hasattr(v, "item") else v


def load_model(model_dir):
    import joblib

    return joblib.load(Path(model_dir) / "model.joblib")


def predict(model, rows: list[dict]) -> list[dict]:
    """rows: [{feature: value}] -> [{"prediction": ..., "probabilities": {label: p}}]."""
    import pandas as pd

    columns = list(model.feature_names_in_)
    df = pd.DataFrame([{c: r.get(c) for c in columns} for r in rows], columns=columns)
    pre = model.steps[0][1]
    categorical = next((cols for name, _, cols in pre.transformers_ if name == "cat"), [])
    if len(categorical):
        df[categorical] = df[categorical].astype(object).where(df[categorical].notna(), "").astype(str)
    numeric = [c for c in columns if c not in set(categorical)]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    preds = model.predict(df)
    out = [{"prediction": _py(p)} for p in preds]
    if hasattr(model, "predict_proba"):
        classes = [_py(c) for c in model.classes_]
        for item, probs in zip(out, model.predict_proba(df)):
            item["probabilities"] = {str(c): round(float(p), 6) for c, p in zip(classes, probs)}
    return out


def export_onnx(model_dir, out_path, features: dict) -> str:
    """Export to ONNX (needs skl2onnx). features: {name: 'numeric' | 'categorical'}, in input order.
    The ONNX graph takes one [N, 1] input per feature."""
    import joblib
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType, StringTensorType

    model = joblib.load(Path(model_dir) / "model.joblib")
    initial_types = [
        (name, FloatTensorType([None, 1]) if kind == "numeric" else StringTensorType([None, 1]))
        for name, kind in features.items()
    ]
    onx = convert_sklearn(model, initial_types=initial_types, options={"zipmap": False}
                          if hasattr(model, "predict_proba") else None)
    Path(out_path).write_bytes(onx.SerializeToString())
    return str(out_path)
