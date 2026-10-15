"""Smoke tests: CPU, < 30 s. transformer_finetune uses a tiny random model (needs network once)."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("sklearn")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "trainers" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _data(tmp_path, n=60):
    import csv
    path = tmp_path / "texts.csv"
    good = ["great product, love it", "excellent service, very happy", "works perfectly, recommended", "很好用，非常滿意"]
    bad = ["terrible, broke after a day", "awful support, very angry", "waste of money, do not buy", "很差，完全不能用"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        for i in range(n):
            w.writerow([good[(i // 2) % 4] + f" #{i}", "pos"] if i % 2 else [bad[(i // 2) % 4] + f" #{i}", "neg"])
    return path


def _ctx(tmp_path, hp):
    from autoforge.engine.context import TrainContext
    return TrainContext(tmp_path / "run", str(_data(tmp_path)), "label", "text-classification", hp, "cpu")


def test_tfidf(tmp_path):
    m = _load("tfidf")
    res = m.train(_ctx(tmp_path, {"analyzer": "char_wb", "ngram_max": 4, "C": 4.0}))
    assert res["metrics"]["accuracy"] > 0.9
    model = m.load_model(Path(res["model_path"]).parent)
    out = m.predict(model, [{"text": "very happy, great"}, {"text": "很差，完全不能用"}])
    assert [o["prediction"] for o in out] == ["pos", "neg"]


def test_transformer_tiny(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    m = _load("transformer")
    ctx = _ctx(tmp_path, {"model_name": "hf-internal-testing/tiny-random-DistilBertForSequenceClassification",
                          "epochs": 2, "batch_size": 16, "max_length": 32, "learning_rate": 0.001})
    res = m.train(ctx)
    assert 0 <= res["metrics"]["f1_macro"] <= 1
    assert (Path(res["model_path"]) / "io.json").is_file()
    names = {e.split('"name": "')[1].split('"')[0] for e in (ctx.run_dir / "events.jsonl").read_text().splitlines() if '"metric"' in e}
    assert {"train_loss", "val_loss", "val_f1_macro"} <= names
    preds = m.predict(m.load_model(res["model_path"]), [{"text": "great"}])
    assert preds[0]["prediction"] in ("pos", "neg")
