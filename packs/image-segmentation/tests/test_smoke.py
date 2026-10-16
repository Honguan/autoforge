"""Smoke test on CPU: squares (class 1) and circles (class 2) on 48x48 images, LR-ASPP from scratch (no download)."""

import base64
import importlib.util
import io
import json
import random
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("segmenter", ROOT / "trainers" / "segmenter.py")
sg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sg)

HP = {"model_name": "lraspp_mobilenet_v3_large", "pretrained": False, "image_size": 48, "epochs": 2,
      "batch_size": 4, "learning_rate": 0.003}


def _data(root: Path, n=12, seed=0, classes_txt=True):
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    for i in range(n):
        img = Image.new("RGB", (48, 48), (90, 90, 90))
        mask = Image.new("L", (48, 48), 0)
        x, y, r = rng.randint(4, 24), rng.randint(4, 24), rng.randint(8, 18)
        shape = "rectangle" if i % 2 == 0 else "ellipse"
        for d, fill in ((ImageDraw.Draw(img), (230, 200, 40) if shape == "rectangle" else (40, 200, 230)),
                        (ImageDraw.Draw(mask), 1 if shape == "rectangle" else 2)):
            getattr(d, shape)([x, y, x + r, y + r], fill=fill)
        img.save(root / "images" / f"s{i}.jpg")
        mask.save(root / "masks" / f"s{i}.png")
    if classes_txt:
        (root / "classes.txt").write_text("background\nsquare\ncircle\n")
    return root


def test_train_predict_export(tmp_path):
    from PIL import Image

    from autoforge.engine.context import TrainContext

    data = _data(tmp_path / "data")
    ctx = TrainContext(tmp_path / "run", str(data), None, "image-segmentation", dict(HP), "cpu")
    res = sg.train(ctx)
    for k in ("miou", "pixel_accuracy"):
        assert 0 <= res["metrics"][k] <= 1
    assert res["metrics"]["train_loss"] > 0
    out = Path(res["model_path"]).parent
    assert json.loads((out / "io.json").read_text())["labels"] == ["background", "square", "circle"]
    m = sg.load_model(out)
    Image.open(data / "images" / "s1.jpg").resize((60, 40)).save(tmp_path / "odd.png")
    b64 = "data:image/png;base64," + base64.b64encode((tmp_path / "odd.png").read_bytes()).decode()
    preds = sg.predict(m, [{"image": str(data / "images" / "s0.jpg")}, {"image": b64}])
    assert preds[1]["prediction"].startswith("data:image/png;base64,")
    mask = Image.open(io.BytesIO(base64.b64decode(preds[1]["prediction"].split(",", 1)[1])))
    assert mask.size == (60, 40) and mask.mode == "L"
    assert set(preds[0]["class_fractions"]) == {"background", "square", "circle"}
    assert abs(sum(preds[0]["class_fractions"].values()) - 1) < 1e-3
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    sg.export_onnx(out, tmp_path / "m.onnx", {})
    assert (tmp_path / "m.onnx").stat().st_size > 1000


def test_train_val_folders_and_default_names(tmp_path):
    from autoforge.engine.context import TrainContext

    _data(tmp_path / "data" / "train", 6, 1, classes_txt=False)
    _data(tmp_path / "data" / "val", 3, 2, classes_txt=False)
    res = sg.train(TrainContext(tmp_path / "run", str(tmp_path / "data"), None, "image-segmentation",
                                dict(HP, epochs=1), "cpu"))
    assert json.loads((Path(res["model_path"]).parent / "io.json").read_text())["labels"] == \
        ["class_0", "class_1", "class_2"]


def test_metrics_are_correct():
    import numpy as np

    t = np.array([[0, 1], [2, 255]])
    cm = sg.confusion(np.array([[0, 1], [1, 0]]), t, 3, 255)
    s = sg.scores(cm)
    assert s["per_class"] == [1.0, 0.5, 0.0] and s["pixel_accuracy"] == pytest.approx(2 / 3)
    assert s["miou"] == pytest.approx(0.5)


def test_bad_data_is_explained(tmp_path):
    from PIL import Image

    from autoforge.engine.context import TrainContext

    data = _data(tmp_path / "data", n=3)
    (data / "masks" / "s2.png").unlink()
    with pytest.raises(FileNotFoundError, match="no mask for image"):
        sg.train(TrainContext(tmp_path / "r1", str(data), None, "image-segmentation", dict(HP), "cpu"))
    Image.new("L", (48, 48), 7).save(data / "masks" / "s2.png")
    with pytest.raises(ValueError, match="contains value 7"):
        sg.train(TrainContext(tmp_path / "r2", str(data), None, "image-segmentation", dict(HP), "cpu"))
