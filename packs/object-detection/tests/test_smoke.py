"""Smoke test on CPU: red / blue rectangles on grey 64x64 images, Faster R-CNN from scratch (no download)."""

import base64
import importlib.util
import json
import random
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("detector", ROOT / "trainers" / "detector.py")
det = importlib.util.module_from_spec(spec)
spec.loader.exec_module(det)

HP = {"model_name": "fasterrcnn_mobilenet_v3_large_fpn", "pretrained": False, "epochs": 2, "batch_size": 4,
      "learning_rate": 0.005, "min_size": 96, "max_size": 96, "score_threshold": 0.05}


def _draw(n, seed):
    """Yield (filename, image, [(xmin, ymin, xmax, ymax, label), ...])."""
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    for i in range(n):
        img = Image.new("RGB", (64, 64), (128, 128, 128))
        d, boxes = ImageDraw.Draw(img), []
        for _ in range(0 if i == 0 else rng.randint(1, 2)):
            label = rng.choice(["red", "blue"])
            x, y = rng.randint(0, 36), rng.randint(0, 36)
            w, h = rng.randint(14, 26), rng.randint(14, 26)
            x2, y2 = min(63, x + w), min(63, y + h)
            d.rectangle([x, y, x2, y2], fill=(220, 30, 30) if label == "red" else (30, 30, 220))
            boxes.append((x, y, x2, y2, label))
        yield f"img{i}.png", img, boxes


def _csv_data(root: Path, n=12, seed=0):
    root.mkdir(parents=True)
    lines = ["image,xmin,ymin,xmax,ymax,label"]
    for name, img, boxes in _draw(n, seed):
        img.save(root / name)
        lines += [f"{name},{x},{y},{x2},{y2},{lab}" for x, y, x2, y2, lab in boxes] or [f"{name},,,,,"]
    (root / "annotations.csv").write_text("\n".join(lines) + "\n")
    return root


def _coco_data(root: Path, n, seed):
    root.mkdir(parents=True)
    cats = {"red": 1, "blue": 2}
    images, anns = [], []
    for i, (name, img, boxes) in enumerate(_draw(n, seed)):
        img.save(root / name)
        images.append({"id": i, "file_name": name, "width": 64, "height": 64})
        anns += [{"id": len(anns), "image_id": i, "category_id": cats[lab], "bbox": [x, y, x2 - x, y2 - y]}
                 for x, y, x2, y2, lab in boxes]
    (root / "annotations.json").write_text(json.dumps(
        {"images": images, "annotations": anns, "categories": [{"id": v, "name": k} for k, v in cats.items()]}))


def test_train_predict_csv(tmp_path):
    from autoforge.engine.context import TrainContext

    data = _csv_data(tmp_path / "data")
    ctx = TrainContext(tmp_path / "run", str(data), None, "object-detection", dict(HP), "cpu")
    res = det.train(ctx)
    for k in ("map50", "precision", "recall"):
        assert 0 <= res["metrics"][k] <= 1
    assert res["metrics"]["train_loss"] > 0
    out = Path(res["model_path"]).parent
    io = json.loads((out / "io.json").read_text())
    assert io["labels"] == ["blue", "red"] and io["features"][0]["dtype"] == "image"
    m = det.load_model(out)
    b64 = "data:image/png;base64," + base64.b64encode((data / "img1.png").read_bytes()).decode()
    preds = det.predict(m, [{"image": str(data / "img1.png"), "score_threshold": 0.0}, {"image": b64}])
    assert preds[0]["count"] == len(preds[0]["prediction"])
    for p in preds[0]["prediction"]:
        assert p["label"] in ("red", "blue") and len(p["box"]) == 4 and 0 <= p["score"] <= 1


def test_coco_with_train_val_folders(tmp_path):
    from autoforge.engine.context import TrainContext

    _coco_data(tmp_path / "data" / "train", 6, 1)
    _coco_data(tmp_path / "data" / "val", 3, 2)
    hp = dict(HP, epochs=1)
    res = det.train(TrainContext(tmp_path / "run", str(tmp_path / "data"), None, "object-detection", hp, "cpu"))
    assert 0 <= res["metrics"]["map50"] <= 1


def test_map50_is_correct():
    import numpy as np

    gt = [(np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float), np.array([1, 2]), None)]
    perfect = [(np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float), np.array([1, 2]), np.array([0.9, 0.8]))]
    assert det.evaluate(perfect, gt, 2, 0.5) == {"map50": 1.0, "precision": 1.0, "recall": 1.0}
    half = [(np.array([[0, 0, 10, 10], [50, 50, 60, 60]], float), np.array([1, 2]), np.array([0.9, 0.8]))]
    assert det.evaluate(half, gt, 2, 0.5) == {"map50": 0.5, "precision": 0.5, "recall": 0.5}


def test_bad_data_is_explained(tmp_path):
    from autoforge.engine.context import TrainContext

    (tmp_path / "empty").mkdir()
    ctx = TrainContext(tmp_path / "run", str(tmp_path / "empty"), None, "object-detection", {}, "cpu")
    with pytest.raises(FileNotFoundError, match="annotations.csv"):
        det.train(ctx)
    data = _csv_data(tmp_path / "bad", n=2)
    (data / "annotations.csv").write_text("image,xmin,ymin,xmax,ymax,label\nimg1.png,30,5,10,20,red\n")
    with pytest.raises(ValueError, match="xmax > xmin"):
        det.train(TrainContext(tmp_path / "run2", str(data), None, "object-detection", {}, "cpu"))
