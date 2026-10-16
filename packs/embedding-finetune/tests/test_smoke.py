"""Smoke test on CPU with a tiny random BERT (downloads < 1 MB once)."""

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("contrastive", ROOT / "trainers" / "contrastive.py")
ct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ct)
TINY = "hf-internal-testing/tiny-random-BertModel"
TOPICS = ["cat", "dog", "car", "rain", "pizza", "guitar", "river", "phone", "train", "apple", "chess", "tennis"]


def _data(tmp_path):
    p = tmp_path / "pairs.jsonl"
    rows = [{"query": f"what about the {t}", "document": f"the {t} is described here",
             "negative": f"the {TOPICS[(i + 1) % len(TOPICS)]} is described here" if i % 2 else ""}
            for i, t in enumerate(TOPICS)]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


def test_train_predict_export(tmp_path):
    from autoforge.engine.context import TrainContext

    hp = {"model_name": TINY, "epochs": 2, "batch_size": 4, "learning_rate": 0.001, "max_length": 16,
          "val_fraction": 0.34, "temperature": 0.05, "warmup_ratio": 0.1, "pooling": "mean"}
    ctx = TrainContext(tmp_path / "run", str(_data(tmp_path)), None, "text-embedding", hp, "cpu")
    res = ct.train(ctx)
    for k in ("recall_at_1", "recall_at_5", "mrr"):
        assert 0 <= res["metrics"][k] <= 1
    events = [json.loads(x) for x in (tmp_path / "run" / "events.jsonl").read_text().splitlines()]
    names = {e.get("name") for e in events}
    assert {"baseline_recall_at_1", "train_loss", "recall_at_1"} <= names
    out = Path(res["model_path"])
    assert (out / "config.json").is_file() and (out / "pooling.json").is_file() and (out / "io.json").is_file()
    assert json.loads((tmp_path / "run" / "checkpoints" / "state.json").read_text())["epoch"] == 1
    again = ct.train(TrainContext(tmp_path / "run", ctx.data_path, None, "text-embedding", dict(hp), "cpu"))
    assert again["metrics"] == res["metrics"]  # `run retry` after the last epoch: resumes, nothing to train

    m = ct.load_model(out)
    pred = ct.predict(m, [{"text": "what about the cat"}, {"text": "a", "compare": "a"}])
    assert pred[0]["dim"] == len(pred[0]["prediction"]) == 32
    assert "similarity" not in pred[0] and pred[1]["similarity"] == pytest.approx(1.0, abs=1e-4)

    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    ct.export_onnx(out, tmp_path / "m.onnx", {})
    assert (tmp_path / "m.onnx").stat().st_size > 1000
    ort = pytest.importorskip("onnxruntime")
    import numpy as np

    enc = m["tok"](["a much longer sentence than the export sample used"], truncation=True,
                  max_length=m["cfg"]["max_length"], return_tensors="np")
    sess = ort.InferenceSession(str(tmp_path / "m.onnx"))
    got = sess.run(None, {"input_ids": enc["input_ids"].astype(np.int64),
                          "attention_mask": enc["attention_mask"].astype(np.int64)})[0][0]
    want = ct.predict(m, [{"text": "a much longer sentence than the export sample used"}])[0]["prediction"]
    assert np.allclose(got, want, atol=1e-4)


def test_bad_columns_are_explained(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("foo,bar\na,b\nc,d\n")
    with pytest.raises(ValueError, match="found columns"):
        ct.read_pairs(str(p))
