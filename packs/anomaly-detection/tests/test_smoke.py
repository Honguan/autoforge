import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("sklearn")
pd = pytest.importorskip("pandas")
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("iforest", ROOT / "trainers" / "iforest.py")
iforest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(iforest)


def _data(tmp_path):
    import numpy as np
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({"amount": rng.normal(50, 10, n), "items": rng.integers(1, 5, n),
                       "channel": rng.choice(["web", "store"], n), "txn_id": range(n), "is_fraud": 0})
    df.loc[:9, "amount"] = rng.normal(900, 50, 10)  # 10 obvious anomalies
    df.loc[:9, "is_fraud"] = 1
    p = tmp_path / "tx.csv"
    df.to_csv(p, index=False)
    return p


def test_finds_planted_anomalies(tmp_path):
    from autoforge.engine.context import TrainContext

    ctx = TrainContext(tmp_path / "run", str(_data(tmp_path)), "is_fraud", "anomaly-detection",
                       {"contamination": 0.02, "exclude": ["txn_id"]}, "cpu")
    res = iforest.train(ctx)
    assert res["metrics"]["roc_auc"] > 0.95 and res["metrics"]["recall"] > 0.8
    m = iforest.load_model(Path(res["model_path"]).parent)
    out = iforest.predict(m, [{"amount": 1000, "items": 2, "channel": "web"}, {"amount": 50, "items": 2, "channel": "web"}])
    assert [o["prediction"] for o in out] == ["anomaly", "normal"]
    assert out[0]["score"] > out[1]["score"]


def test_unsupervised_without_labels(tmp_path):
    from autoforge.engine.context import TrainContext

    ctx = TrainContext(tmp_path / "run", str(_data(tmp_path)), None, "anomaly-detection", {"exclude": ["txn_id", "is_fraud"]}, "cpu")
    assert set(iforest.train(ctx)["metrics"]) == {"anomaly_rate"}
