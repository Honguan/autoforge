import importlib.util
import math
from pathlib import Path

import pytest

pytest.importorskip("sklearn")
pd = pytest.importorskip("pandas")
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("forecast", ROOT / "trainers" / "forecast.py")
fc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fc)


def _series(tmp_path, n=300):
    import random
    rng = random.Random(0)
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    values = [100 + 20 * math.sin(2 * math.pi * i / 7) + rng.gauss(0, 2) for i in range(n)]  # weekly season
    p = tmp_path / "sales.csv"
    pd.DataFrame({"date": dates, "sales": values}).to_csv(p, index=False)
    return p


def test_beats_naive_and_forecasts(tmp_path):
    from autoforge.engine.context import TrainContext

    ctx = TrainContext(tmp_path / "run", str(_series(tmp_path)), "sales", "timeseries-forecast", {"lags": 14}, "cpu")
    res = fc.train(ctx)
    m = res["metrics"]
    assert m["skill_vs_naive"] > 0.3, m  # weekly pattern: the model must clearly beat "same as yesterday"
    model = fc.load_model(Path(res["model_path"]).parent)
    out = fc.predict(model, [{"horizon": 7}, {"history": [100.0] * 14, "horizon": 2}])
    assert len(out[0]["prediction"]) == 7 and len(out[1]["prediction"]) == 2
    with pytest.raises(ValueError, match="at least"):
        fc.predict(model, [{"history": [1, 2, 3]}])


def test_too_short(tmp_path):
    from autoforge.engine.context import TrainContext

    ctx = TrainContext(tmp_path / "run", str(_series(tmp_path, n=20)), "sales", "timeseries-forecast", {"lags": 14}, "cpu")
    with pytest.raises(ValueError, match="too short"):
        fc.train(ctx)


def _many(tmp_path):
    import random
    rng = random.Random(1)
    rows = []
    for k, (level, n) in enumerate([(100, 200), (10, 200), (1000, 200), (50, 40)]):
        for i, d in enumerate(pd.date_range("2025-01-01", periods=n, freq="D")):
            rows.append({"date": d, "store": f"s{k}", "sales": level * (1 + 0.3 * math.sin(2 * math.pi * i / 7))
                         + rng.gauss(0, level * 0.02)})
    p = tmp_path / "stores.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_global_model_many_series(tmp_path):
    from autoforge.engine.context import TrainContext

    hp = {"series_column": "store", "lags": 14}
    ctx = TrainContext(tmp_path / "run", str(_many(tmp_path)), "sales", "timeseries-forecast", hp, "cpu")
    res = fc.train_global(ctx)
    m = res["metrics"]
    assert m["n_series"] == 4 and m["skill_vs_naive"] > 0.3 and 0 <= m["wape"] < 0.1, m
    model = fc.load_model(Path(res["model_path"]).parent)
    big, small, new = fc.predict(model, [{"series": "s2", "horizon": 3}, {"series": "s1", "horizon": 3},
                                         {"history": [5.0] * 14, "horizon": 2}])
    assert 600 < big["prediction"][0] < 1400 and 6 < small["prediction"][0] < 14 and len(new["prediction"]) == 2
    with pytest.raises(ValueError, match="unknown series"):
        fc.predict(model, [{"series": "nope"}])
    bad = TrainContext(tmp_path / "run2", str(_many(tmp_path)), "sales", "timeseries-forecast", {}, "cpu")
    with pytest.raises(ValueError, match="series column"):
        fc.train_global(bad)
