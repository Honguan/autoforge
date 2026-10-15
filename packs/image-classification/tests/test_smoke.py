"""Smoke test on CPU: red vs blue squares, tiny resnet from scratch (no download)."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("timm")
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("timm_train", ROOT / "trainers" / "timm_train.py")
tt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tt)


def _images(tmp_path):
    from PIL import Image

    root = tmp_path / "imgs"
    for cls, color in [("red", (220, 20, 20)), ("blue", (20, 20, 220))]:
        (root / cls).mkdir(parents=True)
        for i in range(12):
            shade = tuple(max(0, min(255, c + (i * 7) % 30 - 15)) for c in color)
            Image.new("RGB", (40, 40), shade).save(root / cls / f"{i}.png")
    return root


def test_train_predict_export(tmp_path):
    from autoforge.engine.context import TrainContext

    data = _images(tmp_path)
    hp = {"model_name": "resnet18", "pretrained": False, "image_size": 32, "epochs": 3, "batch_size": 8,
          "learning_rate": 0.003, "freeze_backbone_epochs": 0}
    ctx = TrainContext(tmp_path / "run", str(data), None, "image-classification", hp, "cpu")
    res = tt.train(ctx)
    assert 0 <= res["metrics"]["accuracy"] <= 1
    out = Path(res["model_path"]).parent
    m = tt.load_model(out)
    pred = tt.predict(m, [{"image": str(data / "red" / "0.png")}, {"image": str(data / "blue" / "0.png")}])
    assert {p["prediction"] for p in pred} <= {"red", "blue"}
    import base64
    b64 = "data:image/png;base64," + base64.b64encode((data / "red" / "1.png").read_bytes()).decode()
    assert tt.predict(m, [{"image": b64}])[0]["prediction"] in ("red", "blue")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    tt.export_onnx(out, tmp_path / "m.onnx", {})
    assert (tmp_path / "m.onnx").stat().st_size > 1000


def test_empty_folder_is_explained(tmp_path):
    from autoforge.engine.context import TrainContext

    (tmp_path / "empty").mkdir()
    ctx = TrainContext(tmp_path / "run", str(tmp_path / "empty"), None, "image-classification", {}, "cpu")
    with pytest.raises(FileNotFoundError, match="no images"):
        tt.train(ctx)
