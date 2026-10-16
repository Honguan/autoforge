"""Smoke test on CPU: 60 users in 3 taste groups x 40 items; both recipes beat (ALS) or run vs popularity."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("pandas")
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("recsys", ROOT / "trainers" / "recsys.py")
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)


def _data(tmp_path, cols=("user_id", "item_id", "clicks", "timestamp")):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    rows, t = [], 0
    for u in range(60):
        group = u % 3
        own = [i for i in range(40) if i % 3 == group]          # 13-14 items per taste group
        for i in rng.choice(own, size=8, replace=False):
            rows.append((f"u{u}", f"item{i}", int(rng.integers(1, 4)), t))
            t += 1
        rows.append((f"u{u}", f"item{int(rng.integers(0, 40))}", 1, t))  # a little noise
        t += 1
    rows += [(f"u{u}", "item0", 1, t + u) for u in range(0, 60, 2)]  # a globally popular item
    order = rng.permutation(len(rows))  # shuffle timestamps so the held-out "last" item is a random one
    rows = [(u, i, w, int(order[n])) for n, (u, i, w, _) in enumerate(rows)]
    p = tmp_path / "events.csv"
    pd.DataFrame(rows, columns=list(cols)).to_csv(p, index=False)
    return p


def _ctx(tmp_path, data, hp, name="run"):
    from autoforge.engine.context import TrainContext

    return TrainContext(tmp_path / name, str(data), None, "recommendation", hp, "cpu", seed=0)


def test_als_beats_popularity_and_serves(tmp_path):
    data = _data(tmp_path)
    hp = {"weight_col": "clicks", "time_col": "timestamp", "k": 5, "factors": 8, "regularization": 0.1,
          "alpha": 10.0, "iterations": 8}
    res = rs.train(_ctx(tmp_path, data, hp))
    m = res["metrics"]
    assert 0 <= m["ndcg_at_k"] <= m["recall_at_k"] <= 1 and 0 < m["coverage"] <= 1
    assert m["recall_at_k"] > m["popularity_recall_at_k"]
    assert (tmp_path / "run" / "checkpoints" / "state.json").is_file()
    model = rs.load_model(Path(res["model_path"]).parent)
    known, cold, sim = rs.predict(model, [{"user": "u0"}, {"user": "nobody", "k": 3}, {"item": "item3"}])
    assert len(known["prediction"]) == 5 and known["cold_start"] is False
    assert sum(int(i[4:]) % 3 == 0 for i in known["prediction"]) >= 3     # u0 is in taste group 0
    assert cold["cold_start"] is True and len(cold["prediction"]) == 3 and cold["prediction"][0] == "item0"
    assert "item3" not in sim["prediction"] and len(sim["scores"]) == 10 // 2


def test_item_knn_autodetects_columns(tmp_path):
    data = _data(tmp_path, cols=("customer_id", "product_id", "rating", "ts"))
    res = rs.train_knn(_ctx(tmp_path, data, {"k": 5, "neighbors": 20, "shrinkage": 5.0}))
    m = res["metrics"]
    assert 0 <= m["recall_at_k"] <= 1 and m["recall_at_k"] >= m["popularity_recall_at_k"]
    model = rs.load_model(Path(res["model_path"]).parent)
    out = rs.predict(model, [{"user": "u1", "exclude_seen": False, "k": 3}, {"item": "item1"}])
    assert len(out[0]["prediction"]) == 3 and out[1]["similar_to"] == "item1"


def test_bad_columns_are_explained(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="no user column found"):
        rs.train(_ctx(tmp_path, p, {}))
    data = _data(tmp_path)
    with pytest.raises(ValueError, match="weight column 'stars' not found"):
        rs.train(_ctx(tmp_path, data, {"weight_col": "stars"}))
