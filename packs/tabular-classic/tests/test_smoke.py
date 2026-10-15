"""Smoke test: must finish on CPU in < 30 s. Run with `pytest packs/tabular-classic/tests`."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("sklearn")
pd = pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("trainers", ROOT / "trainers" / "sklearn_trainers.py")
trainers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainers)


def _data(tmp_path, regression=False):
    import numpy as np

    rng = np.random.default_rng(0)
    n = 200
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    color = rng.choice(["red", "blue"], size=n)
    y = 2 * x1 - x2 + rng.normal(scale=0.1, size=n) if regression else (x1 + x2 > 0).astype(int)
    path = tmp_path / "data.csv"
    pd.DataFrame({"x1": x1, "x2": x2, "color": color, "y": y}).to_csv(path, index=False)
    return path


def _ctx(tmp_path, task, regression=False, hp=None):
    return {"data_path": _data(tmp_path, regression), "target": "y", "task_type": task,
            "hyperparameters": hp or {}, "output_dir": tmp_path / "out", "seed": 0}


def test_logistic_regression(tmp_path):
    res = trainers.train_logistic_regression(_ctx(tmp_path, "tabular-classification"))
    assert res["metrics"]["accuracy"] > 0.85
    assert Path(res["model_path"]).is_file()


def test_random_forest_regression(tmp_path):
    res = trainers.train_random_forest(_ctx(tmp_path, "tabular-regression", True, {"n_estimators": 50}))
    assert res["metrics"]["r2"] > 0.8
