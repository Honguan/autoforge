"""scikit-learn trainers for the tabular-classic pack.

Trainer contract (provisional, finalized in P3):
    train(ctx: dict) -> dict
    ctx keys: data_path (CSV), target, task_type, hyperparameters, output_dir, seed
    returns:  {"metrics": {...}, "model_path": str}
"""

from __future__ import annotations

from pathlib import Path


def _prepare(ctx: dict):
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    df = pd.read_csv(ctx["data_path"])
    y = df.pop(ctx["target"])
    numeric = df.select_dtypes("number").columns.tolist()
    categorical = [c for c in df.columns if c not in numeric]
    pre = ColumnTransformer(
        [
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
            ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                  OneHotEncoder(handle_unknown="ignore")), categorical),
        ]
    )
    is_clf = ctx["task_type"] == "tabular-classification"
    split = train_test_split(
        df, y, test_size=0.2, random_state=ctx.get("seed", 0), stratify=y if is_clf else None
    )
    return pre, split, is_clf


def _fit_and_score(ctx: dict, pre, estimator, split, is_clf: bool) -> dict:
    import joblib
    from sklearn import metrics
    from sklearn.pipeline import make_pipeline

    x_train, x_val, y_train, y_val = split
    model = make_pipeline(pre, estimator).fit(x_train, y_train)
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


def train_logistic_regression(ctx: dict) -> dict:
    from sklearn.linear_model import LogisticRegression

    hp = {"C": 1.0, "max_iter": 1000, "class_weight": None} | ctx.get("hyperparameters", {})
    pre, split, is_clf = _prepare(ctx)
    if not is_clf:
        raise ValueError("logistic_regression only supports tabular-classification")
    return _fit_and_score(ctx, pre, LogisticRegression(**hp), split, is_clf)


def train_random_forest(ctx: dict) -> dict:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    hp = {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1} | ctx.get("hyperparameters", {})
    pre, split, is_clf = _prepare(ctx)
    if is_clf:
        est = RandomForestClassifier(random_state=ctx.get("seed", 0), n_jobs=-1, **hp)
    else:
        hp.pop("class_weight", None)
        est = RandomForestRegressor(random_state=ctx.get("seed", 0), n_jobs=-1, **hp)
    return _fit_and_score(ctx, pre, est, split, is_clf)
